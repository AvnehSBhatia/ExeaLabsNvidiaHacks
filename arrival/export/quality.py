"""Per-episode quality checks for a LeRobot v2.1 folder, and a ``--drop`` that writes a cleaned copy.

``validate.validate`` answers "is this folder structurally a dataset?". This module answers "is each episode worth
training on?". It reads only the parquet files (and, with ``frames=True``, the mp4s), never GR00T.

Checks, every threshold is a module constant and is echoed into the report:

* ``unreadable``     parquet missing, unreadable, or row count != episodes.jsonl length
* ``nonfinite``      NaN or inf in observation.state or action
* ``zero_motion``    the whole state trace moves less than ``ZERO_MOTION_RAD``
* ``stuck_arm``      the arm does not follow its commanded joints (large action-state gap with a frozen state) for
                     more than ``STUCK_STEPS`` consecutive steps
* ``joint_at_limit`` a joint sits within ``LIMIT_FRAC`` of its range limit for more than ``LIMIT_STEPS`` consecutive
                     steps (``arrival_sim_panda`` only: other datasets carry placeholder joints without real limits)
* ``length_outlier`` episode length far from the median of its task family (groups of at least ``MIN_GROUP``)
* ``blank_video``    (``frames=True``) most frames of a camera are near-uniform, i.e. carry no image content

Not checked, because the dataset does not record what it needs: carried-object tilt (object pose is not stored) and
whether a wrist frame shows the right thing (only uniformity is measured, not "useful pixels").
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from arrival.export.lerobot import _stats, dataset_stats, list_column

LIMIT_FRAC = 0.03       # fraction of a joint's range counted as "at the limit"
LIMIT_STEPS = 40        # consecutive steps at the limit before it is flagged (2 s at 20 fps)
STUCK_GAP_RAD = 0.15    # |action - state| above this on some arm joint means the arm is not following the command
STUCK_STEP_RAD = 1e-3   # per-step state change below this on every arm joint means the arm is frozen
STUCK_STEPS = 40        # consecutive steps, both conditions, before it is flagged
ZERO_MOTION_RAD = 1e-3  # peak-to-peak of every state dim below this over the whole episode
LENGTH_RATIO = 2.5      # outlier when length > median * ratio or < median / ratio
MIN_GROUP = 4           # episodes needed in a family before lengths are compared
MIN_LENGTH = 5          # fewer steps than this is never a demonstration
BLANK_STD = 2.0         # a frame whose pixel std is below this (0..255) is blank
BLANK_FRACTION = 0.5    # fraction of blank frames that flags the camera
ARM_DOF = 7
NOT_CHECKED = {
    "carried_object_tilt": "object pose is not recorded in the dataset",
    "wrist_useful_pixels": "only frame uniformity is measured (blank_video), not what the wrist camera sees",
}
THRESHOLDS = {"limit_frac": LIMIT_FRAC, "limit_steps": LIMIT_STEPS, "stuck_gap_rad": STUCK_GAP_RAD,
              "stuck_step_rad": STUCK_STEP_RAD, "stuck_steps": STUCK_STEPS, "zero_motion_rad": ZERO_MOTION_RAD,
              "length_ratio": LENGTH_RATIO, "min_group": MIN_GROUP, "min_length": MIN_LENGTH,
              "blank_std": BLANK_STD, "blank_fraction": BLANK_FRACTION}


class QualityError(ValueError):
    """The folder cannot be checked at all (not a dataset), or a drop would leave nothing."""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    best = cur = 0
    for v in np.asarray(mask, dtype=bool):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _flag(check: str, detail: str) -> dict:
    return {"check": check, "detail": detail}


def check_trace(state: np.ndarray, action: np.ndarray, limits: list[tuple[float, float]] | None = None) -> list[dict]:
    """Flags for one episode's (T, S) state and (T, A) action arrays. Empty list means the episode passed."""
    flags: list[dict] = []
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        return [_flag("nonfinite", "NaN or inf in observation.state or action")]
    if len(state) < MIN_LENGTH:
        flags.append(_flag("length_outlier", f"{len(state)} steps, fewer than {MIN_LENGTH}"))
    if len(state) == 0:
        return flags
    if float(np.ptp(state, axis=0).max()) < ZERO_MOTION_RAD:
        flags.append(_flag("zero_motion", f"state never moves more than {ZERO_MOTION_RAD} over {len(state)} steps"))
    arm = min(ARM_DOF, state.shape[1], action.shape[1])
    if arm and len(state) > 1:
        gap = np.abs(action[:, :arm] - state[:, :arm]).max(1) > STUCK_GAP_RAD
        frozen = np.concatenate([[False], np.abs(np.diff(state[:, :arm], axis=0)).max(1) < STUCK_STEP_RAD])
        run = longest_run(gap & frozen)
        if run > STUCK_STEPS:
            flags.append(_flag("stuck_arm", f"arm frozen {run} consecutive steps while commanded more than "
                                            f"{STUCK_GAP_RAD} rad away (limit {STUCK_STEPS})"))
    if limits:
        for j, (lo, hi) in enumerate(limits[:state.shape[1]]):
            band = LIMIT_FRAC * (hi - lo)
            run = longest_run((state[:, j] <= lo + band) | (state[:, j] >= hi - band))
            if run > LIMIT_STEPS:
                flags.append(_flag("joint_at_limit", f"joint {j + 1} within {LIMIT_FRAC:.0%} of its limit for {run} "
                                                     f"consecutive steps (limit {LIMIT_STEPS})"))
    return flags


def length_outliers(lengths: dict[int, int], groups: dict[int, str]) -> dict[int, str]:
    """Episode index -> detail for lengths far from their family's median (families smaller than MIN_GROUP skipped)."""
    out: dict[int, str] = {}
    by_group: dict[str, list[int]] = {}
    for ep, g in groups.items():
        by_group.setdefault(g, []).append(ep)
    for g, eps in by_group.items():
        if len(eps) < MIN_GROUP:
            continue
        med = float(np.median([lengths[e] for e in eps]))
        for e in eps:
            if lengths[e] > med * LENGTH_RATIO or lengths[e] < med / LENGTH_RATIO:
                out[e] = f"{lengths[e]} steps vs median {med:g} for {g!r} (ratio limit {LENGTH_RATIO}x)"
    return out


def _blank_video(path: Path) -> str | None:
    import imageio.v2 as imageio

    blank = total = 0
    try:
        with imageio.get_reader(str(path)) as r:
            for f in r:
                total += 1
                blank += float(np.asarray(f, dtype=np.float32).std()) < BLANK_STD
    except Exception as e:  # noqa: BLE001 - an undecodable video is a bad episode, not a crash
        return f"{path.name}: cannot decode ({e})"
    if total == 0:
        return f"{path.name}: video has no frames"
    if blank / total > BLANK_FRACTION:
        return f"{path.parent.name}: {blank}/{total} frames are near-uniform (pixel std < {BLANK_STD})"
    return None


def _load(root: Path) -> tuple[dict, list[dict]]:
    meta = root / "meta"
    if not (root.is_dir() and (meta / "info.json").is_file() and (meta / "episodes.jsonl").is_file()):
        raise QualityError(f"{root} is not a LeRobot dataset folder (needs meta/info.json and meta/episodes.jsonl)")
    return json.loads((meta / "info.json").read_text()), _jsonl(meta / "episodes.jsonl")


def check_episodes(root: str | Path, frames: bool = False) -> dict:
    """Run every per-episode check; returns the report dict (see module docstring). Never raises on a bad episode."""
    root = Path(root)
    info, episodes = _load(root)
    size = info.get("chunks_size", 1000)
    data_tpl, video_tpl = info.get("data_path", ""), info.get("video_path") or ""
    limits = None
    if info.get("robot_type") == "arrival_sim_panda":
        from arrival.sim.robot import RANGES
        limits = list(RANGES)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    rows, lengths, groups = [], {}, {}
    for e in episodes:
        ep = e["episode_index"]
        flags: list[dict] = []
        n = None
        path = root / data_tpl.format(episode_chunk=ep // size, episode_index=ep)
        try:
            t = pq.read_table(path, columns=["observation.state", "action"])
            n = t.num_rows
            if n != e.get("length"):
                flags.append(_flag("unreadable", f"{n} rows, episodes.jsonl length {e.get('length')}"))
            state = list_column(t.column("observation.state"))
            action = list_column(t.column("action"))
            flags += check_trace(state, action, limits)
        except Exception as ex:  # noqa: BLE001 - a missing or corrupt parquet flags this episode only
            flags.append(_flag("unreadable", f"{path.relative_to(root)}: {type(ex).__name__}: {ex}"))
        if frames and video_tpl:
            for key in video_keys:
                vp = root / video_tpl.format(episode_chunk=ep // size, video_key=key, episode_index=ep)
                msg = _blank_video(vp) if vp.is_file() else f"missing {vp.relative_to(root)}"
                if msg:
                    flags.append(_flag("blank_video", msg))
        lengths[ep] = n if n is not None else int(e.get("length", 0))
        groups[ep] = e.get("arrival_family") or "all"
        rows.append({"episode_index": ep, "length": lengths[ep], "family": groups[ep], "flags": flags})
    for ep, detail in length_outliers(lengths, groups).items():
        next(r for r in rows if r["episode_index"] == ep)["flags"].append(_flag("length_outlier", detail))
    for r in rows:
        r["ok"] = not r["flags"]
    bad = [r["episode_index"] for r in rows if not r["ok"]]
    by_check: dict[str, int] = {}
    for r in rows:
        for f in r["flags"]:
            by_check[f["check"]] = by_check.get(f["check"], 0) + 1
    checks = ["unreadable", "nonfinite", "zero_motion", "stuck_arm", "length_outlier"]
    checks += ["joint_at_limit"] if limits else []
    checks += ["blank_video"] if frames and video_keys else []
    return {"dataset": str(root), "robot_type": info.get("robot_type"), "episodes": len(rows), "flagged": len(bad),
            "kept": len(rows) - len(bad), "flagged_episodes": bad, "flags_by_check": by_check, "checks_run": checks,
            "not_checked": NOT_CHECKED, "thresholds": THRESHOLDS, "results": rows}


def _rewrite_table(t: pa.Table, new_ep: int, cursor: int) -> pa.Table:
    n = t.num_rows
    for name, vals in (("episode_index", np.full(n, new_ep, dtype=np.int64)),
                       ("index", np.arange(cursor, cursor + n, dtype=np.int64))):
        t = t.set_column(t.schema.get_field_index(name), t.schema.field(name), pa.array(vals))
    return t


def drop_flagged(root: str | Path, out: str | Path, report: dict | None = None, frames: bool = False) -> dict:
    """Write a cleaned copy of ``root`` at ``out`` without the flagged episodes (re-indexed 0..K-1, stats
    recomputed, videos renamed). Returns the report with a ``cleaned`` section. ``out`` must not already exist."""
    root, out = Path(root), Path(out)
    report = report or check_episodes(root, frames=frames)
    if out.exists():
        raise QualityError(f"{out} already exists; refusing to overwrite")
    if out.resolve() == root.resolve() or root.resolve() in out.resolve().parents:
        raise QualityError("--drop must write outside the source dataset")
    info, episodes = _load(root)
    bad = set(report["flagged_episodes"])
    kept = [e for e in episodes if e["episode_index"] not in bad]
    if not kept:
        raise QualityError("every episode was flagged; nothing to write")
    size = info.get("chunks_size", 1000)
    data_tpl, video_tpl = info["data_path"], info.get("video_path") or ""
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    shutil.copytree(root / "meta", out / "meta")
    old_stats = {r["episode_index"]: r for r in _jsonl(root / "meta" / "episodes_stats.jsonl")}
    new_eps, new_stats, cursor, mapping = [], [], 0, {}
    for j, e in enumerate(kept):
        i = e["episode_index"]
        mapping[i] = j
        t = _rewrite_table(pq.read_table(root / data_tpl.format(episode_chunk=i // size, episode_index=i)), j, cursor)
        dst = out / data_tpl.format(episode_chunk=j // size, episode_index=j)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst)
        stats = dict(old_stats.get(i, {}).get("stats", {}))
        for col in list(stats):
            if col in t.column_names:
                stats[col] = _stats(np.array(t.column(col).to_pylist(), dtype=np.float64))
        new_stats.append({"episode_index": j, "stats": stats})
        new_eps.append({**e, "episode_index": j, "length": t.num_rows})
        if video_tpl:
            for key in video_keys:
                src = root / video_tpl.format(episode_chunk=i // size, video_key=key, episode_index=i)
                if src.is_file():
                    vdst = out / video_tpl.format(episode_chunk=j // size, video_key=key, episode_index=j)
                    vdst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, vdst)
        cursor += t.num_rows
    meta = out / "meta"
    for name, rows in (("episodes.jsonl", new_eps), ("episodes_stats.jsonl", new_stats)):
        (meta / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    if all(r["stats"] for r in new_stats):
        (meta / "stats.json").write_text(json.dumps(dataset_stats(out, new_stats), indent=2))
    side = meta / "arrival_cartesian.jsonl"
    if side.is_file():
        rows = [{**r, "episode_index": mapping[r["episode_index"]]} for r in _jsonl(side)
                if r.get("episode_index") in mapping]
        side.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(kept)
    info.update({"total_episodes": n, "total_frames": cursor, "splits": {"train": f"0:{n}"},
                 "total_chunks": max(1, -(-n // size)),
                 "arrival_dropped_by_validator": len(bad)})
    if info.get("total_videos") is not None:
        info["total_videos"] = n * len(video_keys)
    if "arrival_successes" in info:
        info["arrival_successes"] = n  # successes the dataset still holds; arrival_attempts is kept as collected
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    cleaned = {"output": str(out), "kept": n, "dropped": sorted(bad), "index_map": {str(k): v for k, v in mapping.items()},
               "dropped_reasons": {str(r["episode_index"]): r["flags"] for r in report["results"] if not r["ok"]}}
    (meta / "dropped_episodes.json").write_text(json.dumps(cleaned, indent=2))
    return {**report, "cleaned": cleaned}


def format_report(report: dict) -> str:
    lines = [f"{report['dataset']}: {report['episodes']} episodes, {report['flagged']} flagged, {report['kept']} ok "
             f"(checks: {', '.join(report['checks_run'])})"]
    for r in report["results"]:
        for f in r["flags"]:
            lines.append(f"  episode {r['episode_index']} [{r['family']}] {f['check']}: {f['detail']}")
    lines.append("  not checked: " + "; ".join(f"{k} ({v})" for k, v in report["not_checked"].items()))
    return "\n".join(lines)
