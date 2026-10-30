"""Structural checker for a GR00T LeRobot v2.1 folder. Does not import GR00T.

Reads slices from the dataset's own modality.json and checks them against info.features shapes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from arrival.export.lerobot import read_jsonl

META = (
    "info.json",
    "tasks.jsonl",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "stats.json",
    "modality.json",
    "embodiment.json",
)
ROW_KEYS = {"episodes.jsonl": ("episode_index", "length"), "tasks.jsonl": ("task_index", "task")}
DTYPES = {
    "observation.state": pa.list_(pa.float32()),
    "action": pa.list_(pa.float32()),
    "timestamp": pa.float32(),
    "annotation.human.task_description": pa.int64(),
    "task_index": pa.int64(),
    "episode_index": pa.int64(),
    "index": pa.int64(),
    "next.reward": pa.float32(),
    "next.done": pa.bool_(),
}
TS_TOL = 1e-4  # seconds; LeRobot's timestamp sync check uses the same tolerance
VIDEO_SAMPLE = 3  # mp4s fully decoded (first, last, middle); the rest are existence-checked


def meta_shape_error(name: str, rows: list) -> str | None:
    """Why a parsed meta file (rows: one per JSONL line, or [the object]) is not the shape the checks read."""
    for i, row in enumerate(rows):
        where = f"meta/{name} line {i + 1}" if name.endswith(".jsonl") else f"meta/{name}"
        if not isinstance(row, dict):
            return f"{where} is a JSON {type(row).__name__}, not an object"
        missing = [k for k in ROW_KEYS.get(name, ()) if k not in row]
        if missing:
            return f"{where} has no {', '.join(missing)}"
    return None


def _check_slices(modality: dict, features: dict, errors: list[str]) -> None:
    for group, feat in (("state", "observation.state"), ("action", "action")):
        parts = modality.get(group)
        if not isinstance(parts, dict) or not parts:
            errors.append(f"modality.json has no {group} slices")
            continue
        dim = (features.get(feat, {}).get("shape") or [None])[0]
        invalid = [
            k
            for k, part in parts.items()
            if not isinstance(part, dict) or type(part.get("start")) is not int or type(part.get("end")) is not int
        ]
        if invalid:
            errors.extend(f"modality {group}.{k} slice needs integer start/end bounds" for k in invalid)
            continue
        spans = sorted((p.get("start"), p.get("end"), k) for k, p in parts.items())
        cursor = 0
        for start, end, k in spans:
            if start != cursor or end <= start:
                errors.append(f"modality {group}.{k} slice [{start}, {end}) is not contiguous from {cursor}")
                break
            cursor = end
        if dim is not None and cursor != dim:
            errors.append(f"modality {group} slices cover {cursor} dims, info.features.{feat}.shape={dim}")


def _check_video(path: Path, frames: int, shape: list[int]) -> str | None:
    import imageio.v2 as imageio

    try:
        with imageio.get_reader(str(path)) as r:  # closed on every path: an open ffmpeg blocks deletes on Windows
            n = 0
            for f in r:
                if n == 0 and list(f.shape) != list(shape):
                    return f"{path.name}: frame shape {list(f.shape)} != {shape}"
                n += 1
    except Exception as e:  # noqa: BLE001 - any decode failure is a dataset error
        return f"{path.name}: cannot decode ({e})"
    return None if n == frames else f"{path.name}: {n} frames, episodes.jsonl length {frames}"


def _episode_errors(
    name: Path, t: pa.Table, e: dict, cursor: int, s_dim: int, a_dim: int, task_ids: dict, fps
) -> list[str]:
    """Value checks on one episode's parquet table (whose schema was already checked): row count, state and action
    widths, the global index, episode and task indexes, next.done, finite values and timestamps at row / fps."""
    errs: list[str] = []
    ep, length = e["episode_index"], e["length"]
    if t.num_rows != length:
        errs.append(f"{name}: {t.num_rows} rows, episodes.jsonl length {length}")
    # Value checks assume the schema above; malformed columns already have a dtype error.
    cols = {
        c: t.column(c).to_pylist() for c, dt in DTYPES.items() if c in t.column_names and t.schema.field(c).type == dt
    }
    if any(v is None or len(v) != s_dim for v in cols.get("observation.state", [])):
        errs.append(f"{name}: observation.state length != {s_dim}")
    if any(v is None or len(v) != a_dim for v in cols.get("action", [])):
        errs.append(f"{name}: action length != {a_dim}")
    if cols.get("index") != list(range(cursor, cursor + t.num_rows)):
        errs.append(f"{name}: index is not continuous from {cursor}")
    if set(cols.get("episode_index", [ep])) != {ep}:
        errs.append(f"{name}: episode_index column != {ep}")
    for column in ("task_index", "annotation.human.task_description"):
        tis = set(cols.get(column, []))
        if not tis <= set(task_ids):
            errs.append(f"{name}: {column} {sorted(tis - set(task_ids))} not in tasks.jsonl")
        elif any(task_ids[i] not in e.get("tasks", []) for i in tis):
            errs.append(f"{name}: {column} does not match episodes.jsonl tasks")
    done = cols.get("next.done", [])
    if done and (done[-1] is not True or any(done[:-1])):
        errs.append(f"{name}: next.done must be true on the last row only")
    for c in ("observation.state", "action"):
        v = cols.get(c)
        if (
            v
            and all(r is not None and len(r) == len(v[0]) for r in v)
            and not np.isfinite(np.asarray(v, dtype=np.float64)).all()
        ):
            errs.append(f"{name}: {c} has NaN or inf")
    ts = np.asarray(cols.get("timestamp", []), dtype=np.float64)
    if ts.size and not np.isfinite(ts).all():
        errs.append(f"{name}: timestamp has NaN or inf")
    elif ts.size and fps:
        off = np.abs(ts - np.arange(ts.size) / fps)
        if off.max() > TS_TOL:
            k = int(np.argmax(off > TS_TOL))
            errs.append(f"{name}: timestamp row {k} is {ts[k]:.6f}s, expected {k / fps:.6f}s (row / fps={fps})")
    return errs


def _video_errors(root: Path, episodes: list[dict], video_feats: dict, video_tpl: str, size: int) -> list[str]:
    """Every episode needs an mp4 per declared video feature (the first, middle and last are decoded and checked),
    and mp4s without a declared feature are an error too."""
    errs: list[str] = []
    if video_feats:
        if not video_tpl:
            errs.append("video features declared but info.video_path is empty")
        else:
            sample = {0, len(episodes) - 1, len(episodes) // 2}
            for key, feat in video_feats.items():
                for e in episodes:
                    ep = e["episode_index"]
                    path = root / video_tpl.format(episode_chunk=ep // size, video_key=key, episode_index=ep)
                    if not path.is_file():
                        errs.append(f"missing {path.relative_to(root)}")
                    elif ep in sample:
                        err = _check_video(path, e["length"], feat.get("shape", []))
                        if err:
                            errs.append(err)
    elif list(root.glob("videos/**/*.mp4")):
        errs.append("mp4s under videos/ but no video feature is declared in info.features")
    return errs


def validate(root: str | Path) -> list[str]:
    """Structural problems with the LeRobot v2.1 dataset at `root`, as one sentence each (empty when it is valid):
    meta files present and well formed, info, modality and embodiment consistent with each other, chunk layout,
    every episode's parquet schema and values, and the declared videos."""
    root = Path(root)
    errors: list[str] = []
    if not root.is_dir():
        return [f"{root} is not a directory"]
    missing = [m for m in META if not (root / "meta" / m).is_file()]
    if missing:
        return [f"missing meta/{m}" for m in missing]
    for m in META:  # a run killed while writing meta leaves truncated JSON; report it instead of a traceback
        try:
            rows = (
                read_jsonl(root / "meta" / m) if m.endswith(".jsonl") else [json.loads((root / "meta" / m).read_text())]
            )
        except ValueError as e:
            return [f"meta/{m} is not valid JSON ({e})"]
        if err := meta_shape_error(m, rows):
            return [err]
    info = json.loads((root / "meta" / "info.json").read_text())
    modality = json.loads((root / "meta" / "modality.json").read_text())
    features = info.get("features", {})
    if type(info.get("total_videos")) is not int or info["total_videos"] < 0:
        errors.append("info.total_videos must be a nonnegative integer")
    if not isinstance(info.get("arrival_physically_valid"), bool):
        errors.append("info.arrival_physically_valid must be a bool")
    for c in DTYPES:
        if c not in features:
            errors.append(f"info.features missing {c}")
    _check_slices(modality, features, errors)
    video_feats = {k: v for k, v in features.items() if v.get("dtype") == "video"}
    declared = {v.get("original_key") for v in modality.get("video", {}).values()}
    if declared != set(video_feats):
        errors.append(
            f"modality.json video keys {sorted(k for k in declared if k)} != "
            f"info.features video keys {sorted(video_feats)}"
        )
    emb = json.loads((root / "meta" / "embodiment.json").read_text())
    if emb.get("cartesian_sidecar") and not (root / "meta" / emb["cartesian_sidecar"]).is_file():
        errors.append(f"embodiment cartesian_sidecar {emb['cartesian_sidecar']} not found in meta/")

    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    tasks = read_jsonl(root / "meta" / "tasks.jsonl")
    # A string length crashes the sum (int + str). A string index crashes episode_index // chunks_size.
    # true is an int there, so it is formatted as episode 1 and the real parquet is reported missing.
    bad_ints = [e for e in episodes if type(e.get("episode_index")) is not int or type(e.get("length")) is not int]
    if bad_ints:
        for e in bad_ints:
            errors.append(
                "episodes.jsonl episode_index and length must be integers, "
                f"got {e.get('episode_index')!r} and {e.get('length')!r}"
            )
        return errors
    n = info.get("total_episodes")
    if len(episodes) != n:
        errors.append(f"episodes.jsonl has {len(episodes)} rows, info.total_episodes={n}")
    if [e.get("episode_index") for e in episodes] != list(range(len(episodes))):
        errors.append("episodes.jsonl episode_index is not 0..N-1")
    if sum(e["length"] for e in episodes) != info.get("total_frames"):
        errors.append("sum of episode lengths != info.total_frames")
    if len(tasks) != info.get("total_tasks"):
        errors.append(f"tasks.jsonl has {len(tasks)} rows, info.total_tasks={info.get('total_tasks')}")
    task_ids = {t["task_index"]: t["task"] for t in tasks}
    if len(read_jsonl(root / "meta" / "episodes_stats.jsonl")) != len(episodes):
        errors.append("episodes_stats.jsonl row count != episodes.jsonl")

    size = info.get("chunks_size", 1000)
    if type(size) is not int or size <= 0:
        errors.append(f"info.chunks_size must be a positive integer, got {size!r}")
        return errors
    want_chunks = max(1, -(-len(episodes) // size))
    if info.get("total_chunks") != want_chunks:
        errors.append(f"info.total_chunks={info.get('total_chunks')}, expected {want_chunks}")
    data_tpl, video_tpl = info.get("data_path", ""), info.get("video_path") or ""
    expected = {
        (root / data_tpl.format(episode_chunk=e["episode_index"] // size, episode_index=e["episode_index"])).resolve()
        for e in episodes
    }
    found = {p.resolve() for p in root.glob("data/chunk-*/*.parquet")}
    for p in sorted(found - expected)[:5]:
        errors.append(f"unexpected parquet {p.relative_to(root.resolve())}")
    s_dim = (features.get("observation.state", {}).get("shape") or [0])[0]
    a_dim = (features.get("action", {}).get("shape") or [0])[0]
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        errors.append(f"info.fps must be a positive number, got {fps!r}")
        fps = None
    cursor = 0
    for e in episodes:
        ep, length = e["episode_index"], e["length"]
        path = root / data_tpl.format(episode_chunk=ep // size, episode_index=ep)
        name = path.relative_to(root)
        if not path.is_file():
            errors.append(f"missing {name}")
            cursor += length
            continue
        try:
            t = pq.read_table(path)
        except (OSError, pa.ArrowException) as exc:
            errors.append(f"{name}: cannot read parquet ({exc})")
            cursor += length
            continue
        for c, dt in DTYPES.items():
            if c not in t.column_names:
                errors.append(f"{name}: missing column {c}")
            elif t.schema.field(c).type != dt:
                errors.append(f"{name}: column {c} dtype {t.schema.field(c).type} != {dt}")
        if errors and len(errors) > 50:
            return errors
        errors += _episode_errors(name, t, e, cursor, s_dim, a_dim, task_ids, fps)
        cursor += t.num_rows

    errors += _video_errors(root, episodes, video_feats, video_tpl, size)
    if info.get("arrival_video_policy") not in ("all", "sparse", "none"):
        errors.append(f"unknown arrival_video_policy {info.get('arrival_video_policy')!r}")
    return errors
