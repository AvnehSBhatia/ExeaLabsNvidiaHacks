"""CAP-2a: RGB-D scan-frame manifest (frames_manifest.json), strict loader, depth unprojection and a
synthetic bundle generator for tests. The iOS app's recorder (ios/ArrivalCapture/ArrivalCapture/FrameRecorder.swift)
writes these bundles; it compiles and its writer is tested on a Mac, but it has not run on a phone.

Axes: an ARKit camera is x right, y up, z backward (it looks down -z). Image pixel (u, v) is column,
row with v growing downward, so a pixel at depth d (metres along the optical axis) is the camera point
((u - cx) / fx * d, -(v - cy) / fy * d, -d). The room frame is Z-up metres, the twin's frame."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA_VERSION = 1
MANIFEST_NAME = "frames_manifest.json"
CAMERA_AXES = "arkit_x_right_y_up_z_backward"
ROOM_AXES = "z_up_metres"
DEPTH_UNITS = {"m": ".npy", "mm": ".png"}  # float32 metres in .npy, uint16 millimetres in PNG
SPLITS = ("train", "heldout")
ROT_TOL = 1e-3


class ManifestError(ValueError):
    """A frames_manifest.json that must not be used; `code` names the rule that failed."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class Frame:
    id: str
    timestamp: float
    rgb: Path
    depth: Path
    confidence: Path | None
    K: np.ndarray  # 3x3 for the RGB image
    camera_to_room: np.ndarray  # 4x4, ARKit camera axes -> room frame
    split: str


@dataclass(frozen=True)
class Manifest:
    room_id: str
    root: Path
    rgb_size: tuple[int, int]  # (width, height)
    depth_size: tuple[int, int]
    depth_unit: str
    arkit_world_to_room: np.ndarray
    frames: list[Frame]


def _fail(code: str, message: str):
    raise ManifestError(code, message)


def _matrix(value, what: str) -> np.ndarray:
    try:
        m = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        _fail("bad_matrix", f"{what} is not numeric")
    if m.size != 16:
        _fail("bad_matrix", f"{what} needs 16 row-major numbers, got {m.size}")
    m = m.reshape(4, 4)
    if not np.isfinite(m).all():
        _fail("nonfinite_transform", f"{what} has NaN or inf")
    if not np.allclose(m[3], [0, 0, 0, 1]):
        _fail("bad_matrix", f"{what} bottom row must be 0 0 0 1")
    r = m[:3, :3]
    if np.abs(r @ r.T - np.eye(3)).max() > ROT_TOL or np.linalg.det(r) < 0:
        _fail("non_orthonormal_rotation", f"{what} rotation is not orthonormal within {ROT_TOL}")
    return m


def _size(value, what: str) -> tuple[int, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and all(type(x) is int and x > 0 for x in value)):
        _fail("bad_size", f"{what} must be [width, height] positive ints")
    return int(value[0]), int(value[1])


def load_manifest(directory) -> Manifest:
    root = Path(directory)
    resolved_root = root.resolve()
    path = root / MANIFEST_NAME
    if not path.is_file():
        _fail("missing_manifest", f"{path} not found")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _fail("bad_json", str(e))
    if not isinstance(raw, dict):
        _fail("bad_json", "manifest must be a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        _fail("schema_version", f"expected {SCHEMA_VERSION}, got {raw.get('schema_version')!r}")
    room_id = raw.get("room_id")
    if not isinstance(room_id, str) or not room_id:
        _fail("room_id", "room_id must be a non-empty string")
    axes = raw.get("axes")
    if not isinstance(axes, dict):
        _fail("axis_convention", "axes must be an object with camera and room")
    if axes.get("camera") != CAMERA_AXES or axes.get("room") != ROOM_AXES:
        _fail("axis_convention", f"axes must be camera={CAMERA_AXES!r} room={ROOM_AXES!r}, got {axes}")
    unit = raw.get("depth_unit")
    if not isinstance(unit, str) or unit not in DEPTH_UNITS:
        _fail("depth_unit", f"depth_unit must be one of {sorted(DEPTH_UNITS)}, got {unit!r}")
    rgb_size = _size(raw.get("rgb_size"), "rgb_size")
    depth_size = _size(raw.get("depth_size"), "depth_size")
    world = _matrix(raw.get("arkit_world_to_room"), "arkit_world_to_room")
    entries = raw.get("frames")
    if not isinstance(entries, list) or not entries:
        _fail("frames", "frames must be a non-empty list")

    frames: list[Frame] = []
    seen: set[str] = set()
    last_t = -math.inf
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            _fail("frames", f"frame {i} must be an object")
        fid = e.get("id")
        if not isinstance(fid, str) or not fid:
            _fail("frame_id", f"frame {i} has no id")
        if fid in seen:
            _fail("duplicate_frame_id", f"frame id {fid!r} appears twice")
        seen.add(fid)
        t = e.get("timestamp")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t):
            _fail("timestamp", f"frame {fid} timestamp is not a finite number")
        if t <= last_t:
            _fail("non_monotonic_timestamps", f"frame {fid} timestamp {t} is not after {last_t}")
        last_t = float(t)
        k = e.get("intrinsics")
        if not isinstance(k, dict):
            _fail("intrinsics", f"frame {fid} needs numeric fx fy cx cy")
        try:
            values = [k[n] for n in ("fx", "fy", "cx", "cy")]
        except (KeyError, TypeError, ValueError):
            _fail("intrinsics", f"frame {fid} needs numeric fx fy cx cy")
        if any(type(value) not in (int, float) for value in values):
            _fail("intrinsics", f"frame {fid} needs numeric fx fy cx cy")
        fx, fy, cx, cy = map(float, values)
        if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            _fail("intrinsics", f"frame {fid} intrinsics must be finite with fx, fy > 0")
        T = _matrix(e.get("camera_to_room"), f"frame {fid} camera_to_room")
        split = e.get("split")
        if split not in SPLITS:
            _fail("split", f"frame {fid} split must be one of {SPLITS}, got {split!r}")
        paths = {}
        for key in ("rgb", "depth", "confidence"):
            rel = e.get(key)
            if rel is None and key == "confidence":
                paths[key] = None
                continue
            if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
                _fail("bad_path", f"frame {fid} {key} must be a relative path inside the bundle")
            if not (root / rel).resolve().is_relative_to(resolved_root):
                _fail("bad_path", f"frame {fid} {key} resolves outside the bundle")
            if not (root / rel).is_file():
                _fail("missing_file", f"frame {fid} {key} file {rel} does not exist")
            paths[key] = root / rel
        if paths["depth"].suffix.lower() != DEPTH_UNITS[unit]:
            _fail("depth_unit", f"frame {fid} depth {paths['depth'].name} does not match depth_unit {unit!r} "
                  f"(expects {DEPTH_UNITS[unit]})")
        frames.append(Frame(fid, last_t, paths["rgb"], paths["depth"], paths["confidence"],
                            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), T, split))

    split_ids = raw.get("splits")
    if split_ids is not None and not isinstance(split_ids, dict):
        _fail("split", "splits must be an object with train and heldout id lists")
    if split_ids is not None:  # optional explicit lists must agree with the per-frame field
        for name in SPLITS:
            ids = split_ids.get(name)
            if not isinstance(ids, list) or not all(isinstance(fid, str) for fid in ids):
                _fail("split", f"splits.{name} must be a list of frame ids")
            if len(ids) != len(set(ids)):
                _fail("split", f"splits.{name} contains duplicate frame ids")
        both = set(split_ids["train"]) & set(split_ids["heldout"])
        if both:
            _fail("frame_in_both_splits", f"frames in both train and heldout: {sorted(both)}")
        for name in SPLITS:
            expected = {f.id for f in frames if f.split == name}
            listed = set(split_ids[name])
            if listed != expected:
                _fail("split", f"splits.{name} disagrees with frame entries: missing {sorted(expected - listed)}, "
                      f"unknown or mislabelled {sorted(listed - expected)}")
    return Manifest(room_id, root, rgb_size, depth_size, unit, world, frames)


def depth_intrinsics(m: Manifest, frame: Frame) -> np.ndarray:
    """The RGB intrinsics scaled to the depth image's resolution."""
    sx, sy = m.depth_size[0] / m.rgb_size[0], m.depth_size[1] / m.rgb_size[1]
    K = frame.K.copy()
    K[0] *= sx
    K[1] *= sy
    return K


def load_depth(m: Manifest, frame: Frame) -> np.ndarray:
    """Depth in metres, float32, shape (h, w); 0 or non-finite marks no reading."""
    if m.depth_unit == "m":
        raw = np.load(frame.depth)
        if raw.dtype != np.float32:
            _fail("depth_dtype", f"frame {frame.id} metric depth must be float32, got {raw.dtype}")
        d = raw
    else:
        raw = np.asarray(Image.open(frame.depth))
        if raw.dtype != np.uint16:
            _fail("depth_dtype", f"frame {frame.id} millimetre depth must be uint16, got {raw.dtype}")
        d = raw.astype(np.float32) / 1000.0
    if d.shape != (m.depth_size[1], m.depth_size[0]):
        _fail("depth_shape", f"frame {frame.id} depth is {d.shape[::-1]}, manifest says {m.depth_size}")
    return d


def unproject(depth: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Depth image (metres along the optical axis) -> Nx3 points in the room frame; no-reading pixels dropped."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(depth) & (depth > 0)
    d = depth[ok].astype(float)
    x = (u[ok] - K[0, 2]) / K[0, 0] * d
    y = -(v[ok] - K[1, 2]) / K[1, 1] * d
    cam = np.stack([x, y, -d, np.ones_like(d)], axis=1)
    return (cam @ np.asarray(T, float).T)[:, :3]


def project(points: np.ndarray, K: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Room-frame Nx3 points -> (Nx2 pixel (u, v), N depth in metres). Points behind the camera get depth <= 0."""
    p = np.asarray(points, float).reshape(-1, 3)
    cam = (np.c_[p, np.ones(len(p))] @ np.linalg.inv(np.asarray(T, float)).T)[:, :3]
    d = -cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam[:, 0] / d * K[0, 0] + K[0, 2]
        v = -cam[:, 1] / d * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=1), d


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-room 4x4 (ARKit axes) for a camera at `eye` looking at `target`, room Z up."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (eye, target, up)):
        raise ValueError("eye, target and up must be finite 3D vectors")
    f = target - eye
    distance = np.linalg.norm(f)
    if distance < 1e-9:
        raise ValueError("eye and target must be distinct")
    f /= distance
    right = np.cross(f, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        raise ValueError("up must not be parallel to the viewing direction")
    right /= right_norm
    true_up = np.cross(right, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, true_up, -f, eye
    return T


def write_synthetic_bundle(directory, depth_unit: str = "m", size=(64, 48)) -> Path:
    """A flat floor (z = 0) and a wall (x = 3) seen by two cameras; one frame is held out. Synthetic geometry
    and flat-colour images only, never real imagery."""
    root = Path(directory)
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    w, h = size
    fx = fy = 0.9 * w
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1.0]])
    poses = [look_at((0.5, 1.0, 1.5), (2.5, 1.2, 0.3)), look_at((0.6, 2.5, 1.4), (2.6, 1.0, 0.4))]
    ext = DEPTH_UNITS[depth_unit]
    v, u = np.mgrid[0:h, 0:w]
    frames = []
    for i, T in enumerate(poses):
        ray = np.stack([(u - K[0, 2]) / fx, -(v - K[1, 2]) / fy, -np.ones(u.shape)], axis=-1)
        dirs, o = ray @ T[:3, :3].T, T[:3, 3]  # ray.z = -1, so the ray parameter t is the optical-axis depth
        with np.errstate(divide="ignore", invalid="ignore"):
            t_floor = np.where(dirs[..., 2] < 0, -o[2] / dirs[..., 2], np.inf)
            t_wall = np.where(dirs[..., 0] > 0, (3.0 - o[0]) / dirs[..., 0], np.inf)
        depth = np.minimum(t_floor, t_wall)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
        name = f"f{i:03d}"
        if depth_unit == "m":
            np.save(root / "depth" / f"{name}{ext}", depth)
        else:
            Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(root / "depth" / f"{name}{ext}")
        Image.new("RGB", (w, h), (120 + 60 * i, 130, 140)).save(root / "rgb" / f"{name}.png")
        frames.append({"id": name, "timestamp": 1.0 + 0.5 * i, "rgb": f"rgb/{name}.png",
                       "depth": f"depth/{name}{ext}",
                       "intrinsics": {"fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2},
                       "camera_to_room": T.ravel().tolist(), "split": "train" if i == 0 else "heldout"})
    manifest = {"schema_version": SCHEMA_VERSION, "room_id": "synthetic_room",
                "axes": {"camera": CAMERA_AXES, "room": ROOM_AXES},
                "arkit_world_to_room": np.eye(4).ravel().tolist(),
                "rgb_size": [w, h], "depth_size": [w, h], "depth_unit": depth_unit, "frames": frames}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return root
