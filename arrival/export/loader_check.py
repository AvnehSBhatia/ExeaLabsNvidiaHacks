"""Offline conformance check for what the GR00T/LeRobot loader reads from a dataset. Does not import GR00T.

Goes beyond validate.py's structure checks: it recomputes the statistics the loader normalises with,
decodes sample videos for frame count / fps / shape, and reproduces the 16-step action window the training
config asks for. Refuses placeholder datasets: arrival_physically_valid must be true and
joints_are_placeholder false. Parquet is streamed per episode; only the first, middle and last episode's
mp4 per camera is decoded.

    res = check_loader(root)      # {"ok", "errors", "warnings", "info", "stats"}; each finding is {"code", "msg"}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from arrival.export.lerobot import read_jsonl
from arrival.export.validate import meta_shape_error
from arrival.sim.robot import BASE_SPEED, BASE_TURN

HORIZON = 16  # len(delta_indices) of the action modality in configs/gr00t/arrival_panda_config.py
TS_TOL = 1e-4  # seconds, same as validate.py and LeRobot's timestamp sync check
STAT_RTOL, STAT_ATOL = 1e-3, 1e-4
CONST_STD = 1e-8
ARROW = {"float32": pa.float32(), "int64": pa.int64(), "bool": pa.bool_()}
NEEDED_META = (
    "info.json",
    "modality.json",
    "tasks.jsonl",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "stats.json",
    "embodiment.json",
)
STAT_KEYS = ("mean", "std", "min", "max")
Q = (0.01, 0.99)
MAX_ERRORS = 60
BASE_SLICE = (8, 11)  # base x, y, yaw in the room frame: the third slice of state and action in sim datasets (MAN-3)
BASE_HOLD_TOL = 1e-4  # a fixed base's state and action must not move more than this (m, rad) in an episode
BASE_SPEED_TOL = 0.01  # slack over BASE_SPEED / BASE_TURN for float32 rounding of the written columns


class _Findings:
    def __init__(self):
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self._seen: set[tuple] = set()

    def err(self, code: str, msg: str) -> None:
        """Record an error once (the same code and message are kept only the first time), up to MAX_ERRORS."""
        if (code, msg) not in self._seen and len(self.errors) < MAX_ERRORS:
            self._seen.add((code, msg))
            self.errors.append({"code": code, "msg": msg})

    def warn(self, code: str, msg: str) -> None:
        self.warnings.append({"code": code, "msg": msg})


def _arrow_type(feat: dict) -> pa.DataType | None:
    base = ARROW.get(feat.get("dtype"))
    if base is None:
        return None
    return pa.list_(base) if feat.get("shape", [1]) != [1] else base


def _decode(path: Path):
    """(frames, fps, first-frame shape) or an error string."""
    import imageio.v2 as imageio

    try:
        with imageio.get_reader(str(path)) as r:  # closed on every path: an open ffmpeg blocks deletes on Windows
            fps = r.get_meta_data().get("fps")
            n, shape = 0, None
            for fr in r:
                if n == 0:
                    shape = list(fr.shape)
                n += 1
    except Exception as e:  # noqa: BLE001 - any decode failure is a dataset error
        return f"cannot decode ({e})"
    return n, fps, shape


def _close(a, b) -> bool:
    return np.shape(a) == np.shape(b) and bool(np.allclose(a, b, rtol=STAT_RTOL, atol=STAT_ATOL))


def _read_meta(meta: Path, f: _Findings) -> dict | None:
    """Every meta file the loader reads, parsed and keyed by file name (JSONL files as row lists); None, with the
    problems recorded in f, when one is missing, not JSON, or not the shape the checks read."""
    missing = [m for m in NEEDED_META if not (meta / m).is_file()]
    for m in missing:
        f.err("missing_meta", f"missing meta/{m}")
    if missing:
        return None
    parsed = {}
    for m in NEEDED_META:
        try:
            rows = read_jsonl(meta / m) if m.endswith(".jsonl") else [json.loads((meta / m).read_text())]
        except ValueError as e:
            f.err("meta_unreadable", f"meta/{m} is not valid JSON ({e})")
            continue
        if err := meta_shape_error(m, rows):
            f.err("meta_unreadable", err)
        parsed[m] = rows if m.endswith(".jsonl") else rows[0]
    return None if f.errors else parsed


def _decode_videos(
    root: Path, cams: dict, episodes: list[dict], sample: set, video_tpl: str, size: int, fps, f: _Findings
) -> int:
    """Check that every episode has an mp4 per camera, and decode the sampled episodes' videos to compare frame
    count, fps and frame shape with the metadata. Returns how many videos were decoded."""
    n_decoded = 0
    for cv in cams.values():
        for e in episodes:
            ep = e["episode_index"]
            path = root / video_tpl.format(episode_chunk=ep // size, video_key=cv["key"], episode_index=ep)
            rel_path = path.relative_to(root)
            if not path.is_file():
                f.err("missing_video", f"missing {rel_path}")
            elif ep in sample:
                r = _decode(path)
                n_decoded += 1
                if isinstance(r, str):
                    f.err("video_decode", f"{rel_path}: {r}")
                    continue
                n, vfps, shape = r
                if n != e["length"]:
                    f.err("video_frames", f"{rel_path}: {n} frames, episode length {e['length']}")
                if vfps is None or abs(float(vfps) - fps) > 1e-3:
                    f.err("video_fps", f"{rel_path}: mp4 fps {vfps} != info.fps {fps}")
                if shape != list(cv["shape"] or []):
                    f.err("video_shape", f"{rel_path}: frame shape {shape} != info.json {cv['shape']}")
    return n_decoded


def _check_modality(modality: dict, emb: dict, s_dim: int, a_dim: int, f: _Findings) -> tuple:
    """The modality.json slices against the state and action widths, and the base slice against the embodiment.
    Returns (arm state slice, arm action slice, has_base, base_ok, base_actuated); an arm slice is None when
    the relative arm action cannot be built."""
    # modality.json slices against the state/action dims
    arm_s = arm_a = None
    base_s = base_a = None
    for group, dim in (("state", s_dim), ("action", a_dim)):
        parts = modality.get(group) or {}
        if not parts:
            f.err("modality_slices", f"modality.json has no {group} slices")
        for k, p in parts.items():
            st, en = p.get("start"), p.get("end")
            if not (isinstance(st, int) and isinstance(en, int) and 0 <= st < en <= dim):
                f.err("modality_slices", f"modality {group}.{k} slice [{st}, {en}) outside 0..{dim}")
            elif k == "single_arm":
                if group == "state":
                    arm_s = (st, en)
                else:
                    arm_a = (st, en)
            elif k == "base":
                if group == "state":
                    base_s = (st, en)
                else:
                    base_a = (st, en)
    if arm_s is None or arm_a is None:
        f.err(
            "modality_slices", "modality.json needs state.single_arm and action.single_arm for the relative arm action"
        )
    elif arm_s[1] - arm_s[0] != arm_a[1] - arm_a[0]:
        f.err("modality_slices", "state.single_arm and action.single_arm have different widths")
        arm_s = arm_a = None

    # the base pose (MAN-3): both slices or neither, at [8, 11), and the embodiment must agree with the dims
    has_base = base_s is not None or base_a is not None
    base_ok = False
    if has_base:
        if base_s != BASE_SLICE or base_a != BASE_SLICE or s_dim != BASE_SLICE[1] or a_dim != BASE_SLICE[1]:
            f.err(
                "base_slice",
                f"modality base slices must be [{BASE_SLICE[0]}, {BASE_SLICE[1]}) in both state and action "
                f"with 11-float rows, got state {base_s}, action {base_a}, dims {s_dim}/{a_dim}",
            )
        else:
            base_ok = True
    elif emb.get("dof_base"):
        f.err("base_slice", f"embodiment.json says dof_base {emb.get('dof_base')} but modality.json has no base slice")
    base_actuated = emb.get("base_actuated")
    if has_base and not isinstance(base_actuated, bool):
        f.err("base_slice", "embodiment.json needs base_actuated true or false when modality.json has a base slice")
        base_ok = False
    return arm_s, arm_a, has_base, base_ok, base_actuated


def _check_table(t, name: str, features: dict, length: int, task_ids: set, ann_cols: list, fps, f: _Findings) -> None:
    """One episode's parquet against the metadata: every info.features column present with its dtype, the row
    count, task and annotation indices that resolve in tasks.jsonl, and timestamps on the frame_index / fps grid."""
    for c, feat in features.items():
        want = _arrow_type(feat)
        if want is None:
            continue
        if c not in t.column_names:
            f.err("schema", f"{name}: info.features column {c} not in parquet")
        elif t.schema.field(c).type != want:
            f.err("schema", f"{name}: column {c} dtype {t.schema.field(c).type} != info.features {want}")
    if t.num_rows != length:
        f.err("episode_length", f"{name}: {t.num_rows} rows, episodes.jsonl length {length}")
    for c in ["task_index", *ann_cols]:
        if c in t.column_names:
            bad = sorted(set(t.column(c).to_pylist()) - task_ids)
            if bad:
                f.err("task_index_unresolved", f"{name}: {c} {bad} has no row in tasks.jsonl")
    if "timestamp" in t.column_names and t.num_rows:
        ts = np.asarray(t.column("timestamp").to_pylist(), dtype=np.float64)
        off = np.abs(ts - np.arange(ts.size) / fps)
        if not (off <= TS_TOL).all():
            k = int(np.argmax(~(off <= TS_TOL)))
            f.err(
                "timestamp_fps",
                f"{name}: timestamp row {k} is {ts[k]:.6f}s, expected {k / fps:.6f}s (frame_index / info.fps={fps})",
            )


def _check_base_motion(s: np.ndarray, a: np.ndarray, name: str, base_actuated: bool, fps, f: _Findings) -> None:
    """A fixed base must hold still; a driven base must stay within the speed and turn rate the writer allows."""
    for c, x in (("observation.state", s), ("action", a)):
        b = x[:, BASE_SLICE[0] : BASE_SLICE[1]]
        if not base_actuated:
            if len(b) and float(np.ptp(b, axis=0).max()) > BASE_HOLD_TOL:
                f.err(
                    "base_fixed_moves",
                    f"{name}: {c} base moves (range "
                    f"{float(np.ptp(b, axis=0).max()):.4f}) but embodiment.base_actuated is false",
                )
        elif len(b) > 1:
            v = np.linalg.norm(np.diff(b[:, :2], axis=0), axis=1) * fps
            w = np.abs(np.diff(b[:, 2])) * fps
            if v.max() > BASE_SPEED + BASE_SPEED_TOL or w.max() > BASE_TURN + BASE_SPEED_TOL:
                f.err(
                    "base_speed",
                    f"{name}: {c} base moves {v.max():.3f} m/s and {w.max():.3f} rad/s, over the "
                    f"{BASE_SPEED} m/s and {BASE_TURN} rad/s the writer limits it to (a teleport, not a drive)",
                )


def _compare_stats(
    acc: dict, nonfinite_cols: set, stats: dict, ep_stats: list, ep_local: dict, n_ep: int, base_ok, base_actuated, f
) -> dict:
    """Stats recomputed from the parquet accumulators, compared with stats.json and episodes_stats.jsonl; constant
    dimensions are warned about (a parked base once, as such). Returns the recomputed stats per column."""
    out_stats = {}
    for c, d in acc.items():
        if not d["n"] or c in nonfinite_cols:
            continue
        mean = d["sum"] / d["n"]
        rows = np.concatenate(d["rows"]).astype(np.float64)
        # Two-pass std, as the writer stores it. E[x^2] - mean^2 on a constant column is rounding
        # noise (1500 frames of float32 2.1 give 2.98e-8), which is above CONST_STD, so a parked
        # base dim looks like it moves and normalisation is told to divide by zero.
        std = rows.std(0)
        q = np.quantile(rows, Q, axis=0)
        mine = {"mean": mean, "std": std, "min": d["min"], "max": d["max"], "q01": q[0], "q99": q[1]}
        out_stats[c] = {k: v.tolist() for k, v in mine.items()}
        const = np.flatnonzero(std <= CONST_STD)
        if (
            base_ok and base_actuated is False
        ):  # a parked base is constant by construction: say so once, not as a defect
            parked = [i for i in const.tolist() if BASE_SLICE[0] <= i < BASE_SLICE[1]]
            const = np.asarray([i for i in const.tolist() if i not in parked], dtype=int)
            if parked:
                f.warn(
                    "fixed_base_constant",
                    f"{c} dims {parked} are the parked base pose and constant (std 0): train "
                    f"with configs/gr00t/arrival_panda_config.py (arm and gripper only), not the mobile config",
                )
        if len(const):
            f.warn(
                "constant_dim",
                f"{c} dims {const.tolist()} are constant (std 0); std normalisation divides by it, "
                f"so the loader must clamp it",
            )
        got = stats.get(c)
        if not got:
            f.err("stats_missing", f"stats.json has no entry for {c}")
            continue
        if not {"q01", "q99"} <= set(got):
            f.warn(
                "stats_no_quantiles",
                f"stats.json {c} has no q01/q99 (written before the writers stored them); "
                f"GR00T normalises with them, so regenerate with gr00t/data/stats.py or re-run the sim; the "
                f"recomputed values are in the --json report",
            )
        for k in (*STAT_KEYS, "q01", "q99"):
            if k not in got:
                if k not in ("q01", "q99"):
                    f.err("stats_missing", f"stats.json {c} has no {k}")
                continue
            g = np.asarray(got[k], dtype=np.float64)
            if g.shape != mine[k].shape or not np.isfinite(g).all():
                f.err("stats_mismatch", f"stats.json {c}.{k} has the wrong shape or is not finite")
                continue
            zero = np.flatnonzero((g <= CONST_STD) & (std > CONST_STD)) if k == "std" else []
            if len(zero):
                f.err(
                    "stats_std_zero",
                    f"stats.json {c}.std is 0 on dims {zero.tolist()} that move in the parquet "
                    f"(normalisation would divide by zero)",
                )
            elif not _close(g, mine[k]):
                f.err(
                    "stats_mismatch",
                    f"stats.json {c}.{k} differs from the parquet recompute "
                    f"(max abs diff {float(np.abs(g - mine[k]).max()):.3g})",
                )
    if len(ep_stats) != n_ep:
        f.err("stats_mismatch", f"episodes_stats.jsonl has {len(ep_stats)} rows, episodes.jsonl {n_ep}")
    for row in ep_stats:
        for c, loc in ep_local.get(row.get("episode_index"), {}).items():
            for k in STAT_KEYS:
                g = (row.get("stats", {}).get(c) or {}).get(k)
                if g is None or not _close(np.asarray(g, dtype=np.float64), loc[k]):
                    f.err(
                        "stats_mismatch",
                        f"episodes_stats.jsonl episode {row.get('episode_index')} {c}.{k} "
                        f"differs from the parquet recompute",
                    )
    return out_stats


def check_loader(root: str | Path) -> dict:
    """What GR00T's LeRobot loader would read from the dataset at `root`, checked offline: meta files, the
    physically-valid labels, modality slices, video keys, annotations, every episode's parquet (shapes, finite
    values, base motion, action windows), stats recomputed from the parquet, and decoded sample videos.
    Returns {"ok", "errors", "warnings", "info", "stats"}; errors are {"code", "msg"} dicts."""
    root = Path(root)
    f = _Findings()
    out = {"ok": False, "errors": f.errors, "warnings": f.warnings, "info": {}, "stats": {}}
    meta = _read_meta(root / "meta", f)
    if meta is None:
        return out
    info, modality, emb = meta["info.json"], meta["modality.json"], meta["embodiment.json"]
    tasks, episodes, ep_stats = meta["tasks.jsonl"], meta["episodes.jsonl"], meta["episodes_stats.jsonl"]
    stats = meta["stats.json"]
    features = info.get("features", {})

    # what the loader would train on must be physics-checked, not scripted placeholder motion
    if info.get("arrival_physically_valid") is not True:
        f.err(
            "placeholder",
            "refusing: info.arrival_physically_valid is not true; this is the scripted placeholder "
            "demo dataset, not physics-checked sim data (generate one with 'python -m arrival sim')",
        )
    if emb.get("joints_are_placeholder") is not False:
        f.err(
            "placeholder",
            "refusing: embodiment.joints_are_placeholder is not false; the arm joints are "
            "placeholder zeros, do not fine-tune on this dataset",
        )
    if f.errors:
        return out

    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        f.err("fps_invalid", f"info.fps must be a positive number, got {fps!r}")
        return out
    s_dim = (features.get("observation.state", {}).get("shape") or [0])[0]
    a_dim = (features.get("action", {}).get("shape") or [0])[0]

    arm_s, arm_a, has_base, base_ok, base_actuated = _check_modality(modality, emb, s_dim, a_dim, f)

    # video keys: original_key must be a video feature in info.json; every episode needs an mp4 per key
    video_tpl = info.get("video_path") or ""
    size = info.get("chunks_size", 1000)
    cams: dict[str, dict] = {}
    for name, v in (modality.get("video") or {}).items():
        ok_key = v.get("original_key")
        feat = features.get(ok_key)
        if not feat or feat.get("dtype") != "video":
            f.err("video_key", f"modality video.{name} original_key {ok_key!r} is not a video feature in info.json")
            continue
        cams[name] = {"key": ok_key, "shape": feat.get("shape")}
        decl = (feat.get("info") or {}).get("video.fps")
        if decl != fps:
            f.err("video_fps", f"info.features.{ok_key}.info video.fps={decl} != info.fps={fps}")
    if not cams:
        f.err("video_key", "modality.json declares no usable video keys")

    # annotation columns must exist and resolve in tasks.jsonl
    task_ids = {t["task_index"] for t in tasks}
    ann_cols = []
    for name, a in (modality.get("annotation") or {}).items():
        col = (a or {}).get("original_key") or f"annotation.{name}"
        if col not in features:
            f.err("annotation_key", f"modality annotation.{name} column {col} is not in info.features")
        else:
            ann_cols.append(col)

    data_tpl = info.get("data_path", "")
    n_ep = len(episodes)
    sample = {0, n_ep - 1, n_ep // 2} if n_ep else set()
    stat_cols = ("observation.state", "action")
    acc = {c: {"n": 0, "sum": 0.0, "min": None, "max": None, "rows": []} for c in stat_cols}
    ep_local: dict[int, dict] = {}
    rel_min, rel_max = np.inf, -np.inf
    padded = windows = short = frames = 0
    nonfinite_cols: set[str] = set()
    for e in episodes:
        ep, length = e["episode_index"], e["length"]
        path = root / data_tpl.format(episode_chunk=ep // size, episode_index=ep)
        name = str(path.relative_to(root))
        if not path.is_file():
            f.err("missing_parquet", f"missing {name}")
            continue
        try:
            t = pq.read_table(path)
        except Exception as ex:  # noqa: BLE001 - any read failure is a dataset error
            f.err("parquet_unreadable", f"{name}: cannot read ({ex})")
            continue
        _check_table(t, name, features, length, task_ids, ann_cols, fps, f)
        frames += t.num_rows
        # action horizon: starts in the last HORIZON-1 frames read past the end and are padded
        windows += t.num_rows
        if t.num_rows < HORIZON:
            short += 1
            padded += t.num_rows
            f.err("short_episode", f"{name}: {t.num_rows} frames < {HORIZON}-step action horizon")
        else:
            padded += HORIZON - 1
        if any(c not in t.column_names for c in stat_cols) or not t.num_rows:
            continue
        s = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float64)
        a = np.asarray(t.column("action").to_pylist(), dtype=np.float64)
        if s.ndim != 2 or a.ndim != 2 or s.shape[1] != s_dim or a.shape[1] != a_dim:
            f.err(
                "schema",
                f"{name}: state {s.shape} / action {a.shape} do not match info.features dims ({s_dim}, {a_dim})",
            )
            continue
        fin = True
        for c, x in (("observation.state", s), ("action", a)):
            if not np.isfinite(x).all():
                fin = False
                nonfinite_cols.add(c)
                f.err("nonfinite", f"{name}: {c} has NaN or inf")
        if not fin:
            continue
        if base_ok:
            _check_base_motion(s, a, name, base_actuated, fps, f)
        if arm_s and arm_a:
            # relative arm action as the loader builds it: action[t+k] - state[t], tail padded with the last action
            idx = np.minimum(np.arange(len(s))[:, None] + np.arange(HORIZON)[None, :], len(s) - 1)
            rel = a[idx][:, :, arm_a[0] : arm_a[1]] - s[:, None, arm_s[0] : arm_s[1]]
            rel_min, rel_max = min(rel_min, float(rel.min())), max(rel_max, float(rel.max()))
        for c, x in (("observation.state", s), ("action", a)):
            d = acc[c]
            d["n"] += len(x)
            d["sum"] = d["sum"] + x.sum(0)
            d["min"] = x.min(0) if d["min"] is None else np.minimum(d["min"], x.min(0))
            d["max"] = x.max(0) if d["max"] is None else np.maximum(d["max"], x.max(0))
            d["rows"].append(x.astype(np.float32))
            ep_local.setdefault(ep, {})[c] = {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0)}

    out_stats = _compare_stats(acc, nonfinite_cols, stats, ep_stats, ep_local, n_ep, base_ok, base_actuated, f)

    # videos: an mp4 per episode and camera; decode the first, middle and last episode only
    n_decoded = _decode_videos(root, cams, episodes, sample, video_tpl, size, fps, f)

    out["stats"] = out_stats
    out["info"] = {
        "episodes": n_ep,
        "frames": frames,
        "fps": fps,
        "state_dim": s_dim,
        "action_dim": a_dim,
        "horizon": HORIZON,
        "windows": windows,
        "padded_windows": padded,
        "short_episodes": short,
        "videos_decoded": n_decoded,
        "cameras": sorted(cams),
        "base": None if not has_base else ("mobile" if base_actuated else "fixed"),
        "relative_action_range": [rel_min, rel_max] if np.isfinite(rel_min) else None,
    }
    out["ok"] = not f.errors
    return out


def format_report(root, res: dict) -> str:
    """The loader check as text: PASS or FAIL, a summary of what was read (episodes, frames, cameras, action windows,
    relative action range), then every warning and error."""
    i = res["info"]
    lines = [f"loader check: {root}: {'PASS' if res['ok'] else 'FAIL'}"]
    if i:
        lines.append(
            f"  {i['episodes']} episodes, {i['frames']} frames @ {i['fps']} fps, state {i['state_dim']}d / "
            f"action {i['action_dim']}d, cameras {', '.join(i['cameras'])}, {i['videos_decoded']} mp4s decoded"
            + (f", {i['base']} base in state/action[8:11]" if i.get("base") else "")
        )
        pct = 100 * i["padded_windows"] / max(i["windows"], 1)
        lines.append(
            f"  {i['horizon']}-step action windows: {i['padded_windows']}/{i['windows']} padded at the tail "
            f"({pct:.0f}%), {i['short_episodes']} episode(s) shorter than the horizon"
        )
        if i["relative_action_range"]:
            lo, hi = i["relative_action_range"]
            lines.append(f"  relative arm action (action - state): finite, range [{lo:.3f}, {hi:.3f}] rad")
    lines += [f"  WARN {w['code']}: {w['msg']}" for w in res["warnings"]]
    lines += [f"  FAIL {e['code']}: {e['msg']}" for e in res["errors"]]
    return "\n".join(lines)
