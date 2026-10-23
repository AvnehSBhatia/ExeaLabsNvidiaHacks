"""LeRobot v2 writer for sim episodes (real joint states; ego_view + wrist videos)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from arrival.export.lerobot import (CHUNKS_SIZE, MODALITY, SCHEMA, _stats, check_video_size, chunk_of, dataset_stats,
                                    video_info, write_video)
from arrival.sim import menagerie, robot


STATE_DIM = 11  # 7 arm joints, gripper, base x, y, yaw (room frame); same layout for action
BASE_SLICE = {"start": 8, "end": STATE_DIM}


class SimDatasetWriter:
    def __init__(self, root: str | Path, room_id: str, fps: int, res: int, cameras: tuple[str, ...],
                 provenance: dict | None = None, base: str = "fixed", resume: bool = False,
                 arm: str = "procedural"):
        """resume=True keeps what is on disk (the caller then calls `restore`) instead of wiping the folder."""
        check_video_size(res, res)  # before the rmtree below: a bad res must not wipe an existing dataset
        self.root, self.room_id, self.fps, self.res, self.cameras = Path(root), room_id, fps, res, cameras
        self.provenance = provenance  # export.provenance.run_record: input scan hash, code sha, seeds, generators
        self.base = base
        self.arm = arm
        self.finger_open = menagerie.FINGER_OPEN if arm == "menagerie" else robot.FINGER_OPEN
        if self.root.exists() and not resume:
            shutil.rmtree(self.root)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        self.task_index: dict[str, int] = {}
        self.episodes: list[dict] = []
        self.stats: list[dict] = []
        self.total_frames = 0

    def restore(self, episodes: list[dict], stats: list[dict]) -> None:
        """Continue a dataset a killed run left behind: `episodes` and `stats` are the rows of the episodes that were
        fully written (the caller's journal says which), in order. Their files must be there; files of any later
        episode are an unfinished write and are removed so the next episode can take that index."""
        self.episodes, self.stats, self.task_index = list(episodes), list(stats), {}
        for e in self.episodes:
            self.task_index.setdefault(e["tasks"][0], len(self.task_index))
        self.total_frames = sum(e["length"] for e in self.episodes)
        for i in range(len(self.episodes)):
            files = [self._data_path(i)] + [self._video_path(c, i) for c in self.cameras]
            gone = [f.name for f in files if not f.is_file()]
            if gone:
                raise FileNotFoundError(f"episode {i} was journaled as written but {', '.join(gone)} is missing")
        keep = len(self.episodes)
        stale = list(self.root.glob("data/chunk-*/episode_*.parquet")) + list(self.root.glob("videos/chunk-*/*/episode_*.mp4"))
        for f in stale:
            if int(f.stem.split("_")[1]) >= keep:
                f.unlink()

    def _data_path(self, i: int) -> Path:
        return self.root / "data" / f"chunk-{chunk_of(i):03d}" / f"episode_{i:06d}.parquet"

    def _video_path(self, cam: str, i: int) -> Path:
        return self.root / "videos" / f"chunk-{chunk_of(i):03d}" / f"observation.images.{cam}" / f"episode_{i:06d}.mp4"

    def add_episode(self, ep: dict, task: dict) -> int:
        n = len(ep["states"])
        for cam in self.cameras:
            frames = np.asarray(ep["videos"].get(cam))
            if frames.shape != (n, self.res, self.res, 3) or frames.dtype != np.uint8:
                raise ValueError(f"camera {cam}: expected synchronized uint8 video ({n}, {self.res}, {self.res}, 3), "
                                 f"got {frames.shape} {frames.dtype}")
        i = self._write_arrays(ep["states"], ep["actions"], task["instruction"])
        for cam in self.cameras:
            write_video(self._video_path(cam, i), ep["videos"][cam], self.fps)
        self.episodes.append({"episode_index": i, "tasks": [task["instruction"]], "length": n,
                              "arrival_family": task["family"], "arrival_lighting": task["lighting"],
                              "arrival_people": task.get("people", 0),
                              **({"arrival_task_id": task["task_id"]} if "task_id" in task else {}),
                              **({"arrival_planned_by": task["planned_by"]} if "planned_by" in task else {}),
                              "arrival_success_detail": ep["detail"],
                              "arrival_min_human_distance_m": ep["safety"]["min_human_distance_m"],
                              "arrival_human_contact": ep["safety"]["human_contact"],
                              "arrival_people_xy": ep["people_xy"][:: max(1, n // 20)]})
        return i

    def add_copied(self, states: np.ndarray, actions: np.ndarray, row: dict, videos: dict[str, Path]) -> int:
        """Append an episode written elsewhere (merge-shards): its arrays are re-indexed into this dataset and its
        mp4s copied. `row` is its episodes.jsonl row, kept as it is except for `episode_index`."""
        i = self._write_arrays(states, actions, row["tasks"][0])
        for cam in self.cameras:
            dst = self._video_path(cam, i)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(videos[cam], dst)
        self.episodes.append({**row, "episode_index": i})
        return i

    def _write_arrays(self, states: np.ndarray, actions: np.ndarray, instr: str) -> int:
        """Write the next episode's parquet and add its stats row (not its episodes.jsonl row); returns its index."""
        i = len(self.episodes)
        ti = self.task_index.setdefault(instr, len(self.task_index))
        s, a = np.asarray(states).astype(np.float32), np.asarray(actions).astype(np.float32)
        n = len(s)
        reward = np.zeros(n, dtype=np.float32)
        reward[-1] = 1.0
        done = np.zeros(n, dtype=bool)
        done[-1] = True
        table = pa.Table.from_arrays([
            pa.array(list(s), type=pa.list_(pa.float32())), pa.array(list(a), type=pa.list_(pa.float32())),
            pa.array(np.arange(n, dtype=np.float32) / self.fps), pa.array(np.full(n, ti, dtype=np.int64)),
            pa.array(np.full(n, ti, dtype=np.int64)), pa.array(np.full(n, i, dtype=np.int64)),
            pa.array(np.arange(self.total_frames, self.total_frames + n, dtype=np.int64)),
            pa.array(reward), pa.array(done)], schema=SCHEMA)
        self._data_path(i).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, self._data_path(i))
        cols = {"observation.state": s, "action": a, "timestamp": np.arange(n, dtype=np.float32) / self.fps,
                "annotation.human.task_description": np.full(n, ti), "task_index": np.full(n, ti),
                "episode_index": np.full(n, i), "index": np.arange(self.total_frames, self.total_frames + n),
                "next.reward": reward, "next.done": done.astype(np.float64)}
        self.stats.append({"episode_index": i, "stats": {k: _stats(v) for k, v in cols.items()}})
        self.total_frames += n
        return i

    def finish(self, attempts: int, skipped: dict | None = None, extra: dict | None = None) -> dict:
        """`attempts` ran to the physics check; `skipped` (when given) says what never ran and why, from the
        pre-flight or a build or rollout error. Neither is written as an episode. `extra` is more info.json fields,
        written with the `arrival_` prefix (the shard a run covered, what a merge came from)."""
        meta = self.root / "meta"
        with open(meta / "tasks.jsonl", "w") as f:
            for instr, ti in self.task_index.items():
                f.write(json.dumps({"task_index": ti, "task": instr}) + "\n")
        for name, rows in (("episodes.jsonl", self.episodes), ("episodes_stats.jsonl", self.stats)):
            with open(meta / name, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        if self.stats:
            (meta / "stats.json").write_text(json.dumps(dataset_stats(self.root, self.stats), indent=2))
        modality = json.loads(json.dumps(MODALITY))
        modality["state"]["base"] = dict(BASE_SLICE)
        modality["action"]["base"] = {**BASE_SLICE, "absolute": True}
        modality["video"] = {c: {"original_key": f"observation.images.{c}"} for c in self.cameras}
        (meta / "modality.json").write_text(json.dumps(modality, indent=2))
        (meta / "embodiment.json").write_text(json.dumps({
            "embodiment": "arrival_sim_panda", "dof_arm": 7, "dof_gripper": 1, "dof_base": 3,
            "arm": self.arm, "gripper_open": self.finger_open,
            "base": "planar", "base_frame": "room", "base_actuated": self.base == "mobile",
            "control": "absolute_joint_position", "joints_are_placeholder": False,
            "note": ("Procedural Panda-like arm (Franka link offsets, capsule geometry)" if self.arm == "procedural"
                     else "MuJoCo Menagerie Franka Emika Panda (Apache-2.0 meshes, fetched, not committed)")
                    + f" in MuJoCo. Gripper state = mean finger opening in meters (0..{self.finger_open}). "
                    "state[8:11] and action[8:11] are the base x, y, yaw "
                    "in the room frame (meters, radians); with base_actuated false they hold the parked pose for "
                    "the whole episode. Not a real-robot dataset.",
        }, indent=2))
        n = len(self.episodes)
        vid = {f"observation.images.{c}": {"dtype": "video", "shape": [self.res, self.res, 3],
                                            "names": ["height", "width", "channel"],
                                            "info": video_info(self.res, self.res, 3, self.fps)}
               for c in self.cameras}
        info = {
            "codebase_version": "v2.1", "robot_type": "arrival_sim_panda", "fps": self.fps,
            "total_episodes": n, "total_frames": self.total_frames, "total_tasks": len(self.task_index),
            "total_videos": n * len(self.cameras), "total_chunks": max(1, -(-n // CHUNKS_SIZE)), "chunks_size": CHUNKS_SIZE,
            "splits": {"train": f"0:{n}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "arrival_generator": "mujoco_scripted_ik", "arrival_physically_valid": True, "arrival_grasp": "friction",
            "arrival_success_check": "physics", "arrival_successes": n, "arrival_attempts": attempts,
            "arrival_camera_capture": {"phase": "pre_action", "timestamp": "frame_index / fps",
                                       "roles": {c: {"ego_view": "egocentric", "wrist": "wrist",
                                                     "overview": "exocentric"}.get(c, "unspecified") for c in self.cameras}},
            "arrival_video_policy": "all", "arrival_appearance": "mujoco_raster", "arrival_room_id": self.room_id,
            "arrival_families": sorted({e["arrival_family"] for e in self.episodes}),
            **{f"arrival_{k}": v for k, v in {**(skipped or {}), **(extra or {})}.items()},
            **({"arrival_provenance": self.provenance} if self.provenance else {}),
            "features": {"observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
                         "action": {"dtype": "float32", "shape": [STATE_DIM]}, **vid,
                         "timestamp": {"dtype": "float32", "shape": [1]},
                         "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
                         "task_index": {"dtype": "int64", "shape": [1]},
                         "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
                         "next.reward": {"dtype": "float32", "shape": [1]}, "next.done": {"dtype": "bool", "shape": [1]}},
        }
        (meta / "info.json").write_text(json.dumps(info, indent=2))
        return info
