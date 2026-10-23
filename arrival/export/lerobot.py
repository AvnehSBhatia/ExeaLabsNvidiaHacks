"""GR00T LeRobot v2.1 writer: parquet per episode, mp4 per episode per camera, meta/*.

Generic API (reused by the sim package):

    w = LeRobotWriter(root, room_id, video_policy="all", fps=20,
                      state_dim=S, action_dim=A, modality={...},     # modality.json state/action/annotation
                      video_keys=["ego_view", ...],                  # camera names; frames supplied by caller
                      embodiment={...}, extra_info={"arrival_generator": ..., ...})
    for ...:
        videos = {"ego_view": uint8[T,H,W,3]} if w.wants_video(w.next_episode) else None
        w.add_episode(states[T,S], actions[T,A], timestamps[T], instruction, success,
                      videos=videos, extra={"arrival_family": ...},   # extra -> episodes.jsonl row
                      sidecar={...})                                 # optional row in meta/arrival_cartesian.jsonl
    info = w.finish(attempts)

video_policy "all": every episode must supply every camera; the video features are declared.
"sparse"/"none": no video feature is declared (GR00T never looks for missing mp4s); sparse episodes'
frames go to <root>/../<room>_previews/<camera>/episode_XXXXXX.mp4 for humans only.
`add(attempt)` is the scripted-generator path and delegates to add_episode.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from arrival.render import sim_texture

VIDEO_KEY = "observation.images.ego_view"
SPARSE_EVERY = 125
CHUNKS_SIZE = 1000
SIDECAR = "arrival_cartesian.jsonl"
QUANTILES = (0.01, 0.99)  # the q01/q99 GR00T normalises with
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

MODALITY = {
    "state": {"single_arm": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
    "action": {
        "single_arm": {"start": 0, "end": 7, "absolute": True},
        "gripper": {"start": 7, "end": 8, "absolute": True},
    },
    "video": {"ego_view": {"original_key": VIDEO_KEY}},
    "annotation": {"human.task_description": {}},
}

SCRIPTED_EMBODIMENT = {
    "embodiment": "robocasa_panda",
    "dof_arm": 7,
    "dof_gripper": 1,
    "control": "absolute_joint_position",
    "joints_are_placeholder": True,
    "note": "single_arm values are zeros until a physics generator fills them. "
            "Do not fine-tune on this dataset while joints_are_placeholder is true.",
}

SCHEMA = pa.schema([
    ("observation.state", pa.list_(pa.float32())),
    ("action", pa.list_(pa.float32())),
    ("timestamp", pa.float32()),
    ("annotation.human.task_description", pa.int64()),
    ("task_index", pa.int64()),
    ("episode_index", pa.int64()),
    ("index", pa.int64()),
    ("next.reward", pa.float32()),
    ("next.done", pa.bool_()),
])
SCALAR_FEATURES = {
    "timestamp": "float32", "annotation.human.task_description": "int64", "task_index": "int64",
    "episode_index": "int64", "index": "int64", "next.reward": "float32", "next.done": "bool",
}


def video_key(name: str) -> str:
    return name if name.startswith("observation.images.") else f"observation.images.{name}"


def _check_camera_name(name: str) -> None:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"camera name {name!r} must be a single safe path component")


def chunk_of(episode_index: int, chunks_size: int = CHUNKS_SIZE) -> int:
    return episode_index // chunks_size


def wants_video(policy: str, episode_index: int) -> bool:
    if policy == "all":
        return True
    if policy == "sparse":
        return episode_index % SPARSE_EVERY == 0
    return False


def check_video_size(h: int, w: int) -> None:
    """imageio's macro_block_size=16 silently resizes any other size, so info.features would lie about the mp4."""
    if h % 16 or w % 16:
        raise ValueError(f"video frames are {h}x{w}; height and width must be multiples of 16 or the mp4 "
                         f"is resized and no longer matches info.features")


def video_info(h: int, w: int, c: int, fps: int) -> dict:
    return {"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
            "video.fps": fps, "video.channels": c, "video.is_depth_map": False, "has_audio": False}


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, format="FFMPEG", codec="libx264", pixelformat="yuv420p",
                            macro_block_size=16, ffmpeg_log_level="error") as w:
        for f in frames:
            w.append_data(f)


def _stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64).reshape(len(x), -1)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def list_column(col) -> np.ndarray:
    """A list<number> or scalar parquet column as a (rows, width) float64 array, read from the Arrow buffers. Rows of
    unequal width or null rows raise ValueError."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if arr.null_count:
        raise ValueError("column has null rows")
    if pa.types.is_list(arr.type):
        widths = pc.list_value_length(arr).to_numpy()
        if len(widths) and (widths != widths[0]).any():
            raise ValueError("column rows differ in width")
        arr = arr.flatten()
    return arr.to_numpy(zero_copy_only=False).astype(np.float64).reshape(len(col), -1)


def dataset_stats(root: Path, ep_stats: list[dict], data_glob: str = "data/chunk-*/episode_*.parquet") -> dict:
    """stats.json: the per-episode stats merged, plus q01/q99 over every frame. GR00T's loader expects both
    quantiles and recomputes the whole file when they are missing; quantiles do not merge across episodes, so they
    are read back from the parquet files."""
    stats = {k: _merge_stats([s["stats"][k] for s in ep_stats]) for k in ep_stats[0]["stats"]}
    tables = [pq.read_table(f) for f in sorted(Path(root).glob(data_glob))]
    for k, st in stats.items():
        cols = [list_column(t.column(k)) for t in tables if k in t.column_names and t.num_rows]
        if not cols:
            continue
        x = np.vstack(cols).reshape(-1, len(st["mean"]))
        q01, q99 = np.quantile(x, QUANTILES, axis=0)
        st.update({"q01": q01.tolist(), "q99": q99.tolist()})
    return stats


def _merge_stats(items: list[dict]) -> dict:
    n = np.array([s["count"][0] for s in items], dtype=np.float64)
    mean = np.array([s["mean"] for s in items])
    var = np.array([s["std"] for s in items]) ** 2
    tot = n.sum()
    gmean = (mean * n[:, None]).sum(0) / tot
    gvar = ((var + (mean - gmean) ** 2) * n[:, None]).sum(0) / tot
    return {"min": np.min([s["min"] for s in items], 0).tolist(), "max": np.max([s["max"] for s in items], 0).tolist(),
            "mean": gmean.tolist(), "std": np.sqrt(gvar).tolist(), "count": [int(tot)]}


class LeRobotWriter:
    def __init__(self, root: str | Path, room_id: str, video_policy: str, fps: int = 20,
                 room_center_x: float = 2.0, state_dim: int = 8, action_dim: int = 8,
                 modality: dict | None = None, video_keys: list[str] | None = None,
                 embodiment: dict | None = None, extra_info: dict | None = None,
                 robot_type: str = "robocasa_panda", chunks_size: int = CHUNKS_SIZE):
        if video_policy not in ("all", "sparse", "none"):
            raise ValueError(f"unknown video policy {video_policy!r}")
        if type(fps) is not int or fps < 1:
            raise ValueError("fps must be a positive integer")

        if type(chunks_size) is not int or chunks_size < 1:
            raise ValueError("chunks_size must be a positive integer")

        if video_keys is not None and not isinstance(video_keys, (list, tuple)):
            raise ValueError("video_keys must be a list of camera names")
        names = list(video_keys if video_keys is not None else ["ego_view"])
        for name in names:
            _check_camera_name(name)
        if len(names) != len(set(names)) or (video_policy == "all" and not names):
            raise ValueError("video_keys must contain distinct camera names (at least one for 'all')")
        self.root = Path(root)
        self.room_id = room_id
        self.video_policy = video_policy
        self.fps = fps
        self.room_center_x = room_center_x
        self.state_dim, self.action_dim = state_dim, action_dim
        self.video_keys = names
        base = modality if modality is not None else MODALITY
        self.modality = {k: v for k, v in base.items() if k != "video"}
        self.embodiment = dict(embodiment if embodiment is not None else SCRIPTED_EMBODIMENT)
        self.extra_info = {"arrival_generator": "scripted_retarget", "arrival_physically_valid": False,
                           "arrival_appearance": sim_texture.APPEARANCE, **(extra_info or {})}
        self.robot_type = robot_type
        self.chunks_size = chunks_size
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "meta").mkdir(parents=True)
        self.previews_dir = self.root.parent / f"{room_id}_previews"
        if self.previews_dir.exists():
            shutil.rmtree(self.previews_dir)
        self.sidecar_path = self.root / "meta" / SIDECAR
        self._sidecar = open(self.sidecar_path, "w")
        self.task_index: dict[str, int] = {}
        self.episodes: list[dict] = []
        self.ep_stats: list[dict] = []
        self.families: set[str] = set()
        self.video_shapes: dict[str, list[int]] = {}
        self.total_frames = 0
        self.videos = 0
        self.previews = 0

    @property
    def next_episode(self) -> int:
        return len(self.episodes)

    def wants_video(self, episode_index: int) -> bool:
        return wants_video(self.video_policy, episode_index)

    def add_episode(self, states: np.ndarray, actions: np.ndarray, timestamps, instruction: str,
                    success: bool, videos: dict[str, np.ndarray] | None = None, extra: dict | None = None,
                    sidecar: dict | None = None) -> int:
        ep = len(self.episodes)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"episode {ep}: instruction must be a nonempty string")
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        n = len(states)
        if states.shape != (n, self.state_dim) or actions.shape != (n, self.action_dim) or n == 0:
            raise ValueError(f"episode {ep}: states {states.shape} / actions {actions.shape} "
                             f"!= (T,{self.state_dim}) / (T,{self.action_dim})")
        ts = np.asarray(timestamps, dtype=np.float32)
        if ts.shape != (n,):
            raise ValueError(f"episode {ep}: timestamps shape {ts.shape} != ({n},)")
        for what, v in (("states", states), ("actions", actions), ("timestamps", ts)):
            if not np.isfinite(v).all():  # NaN would also land in stats.json as a bare NaN token (invalid JSON)
                raise ValueError(f"episode {ep}: {what} contain NaN or inf")
        # every check before the first write: a rejected episode leaves no orphan parquet or mp4
        if self.video_policy == "all":
            missing = [k for k in self.video_keys if not videos or k not in videos]
            if missing:
                raise ValueError(f"episode {ep}: video policy 'all' needs frames for {missing}")
        clips = {}
        for name, frames in (videos or {}).items():
            frames = np.asarray(frames)
            if frames.dtype != np.uint8:
                raise ValueError(f"episode {ep}: video {name} must contain uint8 RGB pixels, got {frames.dtype}")

            _check_camera_name(name)
            if frames.ndim != 4 or frames.shape[0] != n or frames.shape[3] != 3:
                raise ValueError(f"episode {ep}: video {name} shape {frames.shape} != ({n},H,W,3)")
            if self.video_policy == "all" or self.wants_video(ep):
                check_video_size(*frames.shape[1:3])
            if self.video_policy == "all":
                shape = list(frames.shape[1:])
                if self.video_shapes.get(name, shape) != shape:
                    raise ValueError(f"episode {ep}: video {name} shape changed to {shape}")
            clips[name] = frames
        ti = self.task_index.setdefault(instruction, len(self.task_index))
        reward = np.zeros(n, dtype=np.float32)
        if success:
            reward[-1] = 1.0
        done = np.zeros(n, dtype=bool)
        done[-1] = True
        cols = {
            "observation.state": states, "action": actions, "timestamp": ts,
            "annotation.human.task_description": np.full(n, ti, dtype=np.int64),
            "task_index": np.full(n, ti, dtype=np.int64), "episode_index": np.full(n, ep, dtype=np.int64),
            "index": np.arange(self.total_frames, self.total_frames + n, dtype=np.int64),
            "next.reward": reward, "next.done": done,
        }
        table = pa.Table.from_arrays([
            pa.array(list(states), type=pa.list_(pa.float32())),
            pa.array(list(actions), type=pa.list_(pa.float32())),
            *[pa.array(cols[c]) for c in SCHEMA.names[2:]],
        ], schema=SCHEMA)
        chunk = chunk_of(ep, self.chunks_size)
        path = self.root / DATA_PATH.format(episode_chunk=chunk, episode_index=ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

        for name, frames in clips.items():
            if self.video_policy == "all":
                self.video_shapes.setdefault(name, list(frames.shape[1:]))
                write_video(self.root / VIDEO_PATH.format(episode_chunk=chunk, video_key=video_key(name),
                                                          episode_index=ep), frames, self.fps)
                self.videos += 1
            elif self.wants_video(ep):
                write_video(self.previews_dir / name / f"episode_{ep:06d}.mp4", frames, self.fps)
                self.previews += 1

        self.ep_stats.append({"episode_index": ep, "stats": {
            k: _stats(v.astype(np.float64) if v.dtype == bool else v) for k, v in cols.items()}})
        self._sidecar.write(json.dumps({"episode_index": ep, **(sidecar or {})}) + "\n")
        row = {"episode_index": ep, "tasks": [instruction], "length": n, **(extra or {})}
        self.episodes.append(row)
        if "arrival_family" in row:
            self.families.add(row["arrival_family"])
        self.total_frames += n
        return ep

    def add(self, attempt: dict) -> int:
        """Scripted path: absolute joint targets, action[i] == state[i]."""
        task, frames = attempt["task"], attempt["frames"]
        state = np.array([f["joint"] + [f["gripper"]] for f in frames], dtype=np.float32)
        ep = self.next_episode
        videos = None
        if self.wants_video(ep):
            videos = {"ego_view": sim_texture.render_episode(frames, task["lighting"], self.room_center_x)}
        return self.add_episode(state, state, [f["t"] for f in frames], task["instruction"], attempt["success"],
                                videos=videos,
                                extra={"arrival_family": task["family"], "arrival_lighting": task["lighting"]},
                                sidecar={"task_id": task["task_id"], "frames": frames})

    def features(self) -> dict:
        feats = {
            "observation.state": {"dtype": "float32", "shape": [self.state_dim]},
            "action": {"dtype": "float32", "shape": [self.action_dim]},
        }
        if self.video_policy == "all":
            for name in self.video_keys:
                h, w, c = self.video_shapes.get(name, [sim_texture.SIZE, sim_texture.SIZE, 3])
                feats[video_key(name)] = {
                    "dtype": "video", "shape": [h, w, c], "names": ["height", "width", "channel"],
                    "info": video_info(h, w, c, self.fps),
                }
        for k, dt in SCALAR_FEATURES.items():
            feats[k] = {"dtype": dt, "shape": [1]}
        return feats

    def close(self) -> None:
        """Release the sidecar handle; idempotent, so an aborted run can call it without finish()."""
        self._sidecar.close()

    def __enter__(self) -> "LeRobotWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def finish(self, attempts: int) -> dict:
        self.close()
        meta = self.root / "meta"
        with open(meta / "tasks.jsonl", "w") as f:
            for instr, ti in self.task_index.items():
                f.write(json.dumps({"task_index": ti, "task": instr}) + "\n")
        with open(meta / "episodes.jsonl", "w") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")
        with open(meta / "episodes_stats.jsonl", "w") as f:
            for s in self.ep_stats:
                f.write(json.dumps(s) + "\n")
        if self.ep_stats:
            (meta / "stats.json").write_text(json.dumps(dataset_stats(self.root, self.ep_stats), indent=2))
        modality = dict(self.modality)
        if self.video_policy == "all":
            modality["video"] = {name: {"original_key": video_key(name)} for name in self.video_keys}
        (meta / "modality.json").write_text(json.dumps(modality, indent=2))
        (meta / "embodiment.json").write_text(json.dumps({**self.embodiment, "cartesian_sidecar": SIDECAR},
                                                         indent=2))
        n = len(self.episodes)
        info = {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": n,
            "total_frames": self.total_frames,
            "total_tasks": len(self.task_index),
            "total_videos": self.videos,
            "total_chunks": max(1, math.ceil(n / self.chunks_size)),
            "chunks_size": self.chunks_size,
            "splits": {"train": f"0:{n}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self.video_policy == "all" else None,
            **self.extra_info,
            "arrival_successes": n,
            "arrival_attempts": attempts,
            "arrival_video_policy": self.video_policy,
            "arrival_previews": self.previews,
            "arrival_room_id": self.room_id,
            "arrival_families": sorted(self.families),
            "features": self.features(),
        }
        (meta / "info.json").write_text(json.dumps(info, indent=2))
        return info
