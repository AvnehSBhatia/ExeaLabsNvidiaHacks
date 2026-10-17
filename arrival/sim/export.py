"""LeRobot v2 writer for sim episodes (real joint states; ego_view + wrist videos)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from arrival.export.lerobot import MODALITY, SCHEMA, write_video


def _stats(x: np.ndarray) -> dict:
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


class SimDatasetWriter:
    def __init__(self, root: str | Path, room_id: str, fps: int, res: int, cameras: tuple[str, ...], mobile: bool = False):
        self.root, self.room_id, self.fps, self.res, self.cameras = Path(root), room_id, fps, res, cameras
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "meta").mkdir(parents=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True)
        self.task_index: dict[str, int] = {}
        self.episodes: list[dict] = []
        self.stats: list[dict] = []
        self.total_frames = 0
        if mobile:
            self.state_dim = 9

    def add_episode(self, ep: dict, task: dict) -> int:
        i = len(self.episodes)
        instr = task["instruction"]
        ti = self.task_index.setdefault(instr, len(self.task_index))
        s, a = ep["states"].astype(np.float32), ep["actions"].astype(np.float32)
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
        pq.write_table(table, self.root / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
        for cam in self.cameras:
            write_video(self.root / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{i:06d}.mp4",
                        ep["videos"][cam], self.fps)
        self.episodes.append({"episode_index": i, "tasks": [instr], "length": n, "arrival_family": task["family"],
                              "arrival_lighting": task["lighting"], "arrival_people": task.get("people", 0),
                              "arrival_success_detail": ep["detail"],
                              "arrival_min_human_distance_m": ep["safety"]["min_human_distance_m"],
                              "arrival_human_contact": ep["safety"]["human_contact"],
                              "arrival_people_xy": ep["people_xy"][:: max(1, n // 20)]})
        self.stats.append({"episode_index": i, "stats": {"observation.state": _stats(s), "action": _stats(a)}})
        self.state_dim = s.shape[1]
        self.total_frames += n
        return i

    def finish(self, attempts: int) -> dict:
        meta = self.root / "meta"
        with open(meta / "tasks.jsonl", "w") as f:
            for instr, ti in self.task_index.items():
                f.write(json.dumps({"task_index": ti, "task": instr}) + "\n")
        for name, rows in (("episodes.jsonl", self.episodes), ("episodes_stats.jsonl", self.stats)):
            with open(meta / name, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        modality = json.loads(json.dumps(MODALITY))
        modality["video"] = {c: {"original_key": f"observation.images.{c}"} for c in self.cameras}
        state_dim = getattr(self, "state_dim", 8)
        robot_type = "arrival_sim_panda"
        emb_note = ("Procedural Panda-like arm (Franka link offsets, capsule geometry) in MuJoCo. Gripper state = "
                    "mean finger opening in meters (0..0.05). Base is fixed per episode; pose in episodes.jsonl is "
                    "not recorded in state. Not a real-robot dataset.")
        dof_base = None
        if state_dim == 9:
            modality["state"]["base"] = {"start": 8, "end": 9}
            modality["action"]["base"] = {"start": 8, "end": 9, "absolute": True}
            robot_type = "arrival_sim_mobile_panda"
            dof_base = 1
            emb_note = ("Procedural Panda-like arm on actuated mobile base (slide joint along counter run) in MuJoCo. "
                        "Gripper state = mean finger opening in meters (0..0.05). Base state = base_x position.")
        (meta / "modality.json").write_text(json.dumps(modality, indent=2))
        emb = {
            "embodiment": robot_type, "dof_arm": 7, "dof_gripper": 1,
            "control": "absolute_joint_position", "joints_are_placeholder": False,
            "note": emb_note,
        }
        if dof_base is not None:
            emb["dof_base"] = dof_base
        (meta / "embodiment.json").write_text(json.dumps(emb, indent=2))
        n = len(self.episodes)
        vid = {f"observation.images.{c}": {"dtype": "video", "shape": [self.res, self.res, 3],
                                            "names": ["height", "width", "channel"],
                                            "video_info": {"video.fps": self.fps, "video.codec": "h264"}}
               for c in self.cameras}
        info = {
            "codebase_version": "v2.1", "robot_type": robot_type, "fps": self.fps,
            "total_episodes": n, "total_frames": self.total_frames, "total_tasks": len(self.task_index),
            "total_videos": n * len(self.cameras), "total_chunks": 1, "chunks_size": 1000,
            "splits": {"train": f"0:{n}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "arrival_generator": "mujoco_scripted_ik", "arrival_physically_valid": True, "arrival_grasp": "friction",
            "arrival_success_check": "physics", "arrival_successes": n, "arrival_attempts": attempts,
            "arrival_video_policy": "all", "arrival_appearance": "mujoco_raster", "arrival_room_id": self.room_id,
            "arrival_families": sorted({e["arrival_family"] for e in self.episodes}),
            "features": {"observation.state": {"dtype": "float32", "shape": [state_dim]},
                         "action": {"dtype": "float32", "shape": [state_dim]}, **vid,
                         "timestamp": {"dtype": "float32", "shape": [1]},
                         "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
                         "task_index": {"dtype": "int64", "shape": [1]},
                         "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
                         "next.reward": {"dtype": "float32", "shape": [1]}, "next.done": {"dtype": "bool", "shape": [1]}},
        }
        (meta / "info.json").write_text(json.dumps(info, indent=2))
        return info
