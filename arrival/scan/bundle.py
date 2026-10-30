"""Load and validate a ScanBundle (scans/<room_id>/scan.json)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from arrival.provenance import sha256_file

GEOMETRY_SOURCES = ("authored_sample", "roomplan", "four_photo_unmeasured")
OPTIONAL_LISTS = ("doors", "windows", "openings", "floors")


MAX_DIM_M = 100.0  # no single fixture, wall or floor edge is longer than this
MAX_COORD_M = 1000.0  # a coordinate beyond this is a corrupt transform, not a big room
MIN_ROOM_M, MAX_ROOM_M = (
    0.5,
    100.0,
)  # bounding box of the walls, per side: a closet to a warehouse, not a speck or a campus


class ScanBundleError(ValueError):
    """A scan that cannot become a twin. The message says what is wrong, in words a person scanning a room can act on."""


def is_finite_number(value) -> bool:
    """True for an int or float that converts to a finite float. False for a bool, any other type, nan, inf,
    and an int too large for float: that raises OverflowError, which is not a ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def numbers(value, n: int | None = None) -> list[float] | None:
    """`value` as a list of finite floats (exactly n of them when given); None when it is anything else.
    An int too large for float is not a finite float. A number written as a string ("2.4") is not a number: float()
    would read it, but the twin keeps a wall's own values, and the wall-span check and the sim then fail on the str."""
    if (
        isinstance(value, (str, bytes, dict))
        or not isinstance(value, (list, tuple))
        or (n is not None and len(value) != n)
    ):
        return None
    if not all(is_finite_number(v) for v in value):
        return None
    return [float(v) for v in value]


def _mujoco_name(label: str, ident: str) -> None:
    """Wall, fixture and object ids are copied into MuJoCo name attributes. A double quote ends the attribute
    and a NUL is not a legal XML character, so the model does not load and the error does not name the id."""
    if '"' in ident or "\x00" in ident:
        raise ScanBundleError(
            f"{label} {ident!r}: id cannot contain a double quote or a NUL character "
            "(the sim uses it as a MuJoCo name, and the model does not load)"
        )


def _check_geometry(bundle: dict) -> None:
    """Finite, positive, plausible numbers everywhere the twin compiler and the scene builder read them."""
    for n, w in enumerate(bundle.get("walls", [])):
        wid = w.get("id") if isinstance(w, dict) else None
        ends = [numbers(w.get(k), 2) if isinstance(w, dict) else None for k in ("start", "end")]
        hv = numbers([w.get("height"), w.get("thickness")] if isinstance(w, dict) else None, 2)
        if None in ends or hv is None or min(hv) <= 0:
            raise ScanBundleError(f"wall {wid}: start and end must be [x, y] numbers and height and thickness positive")
        if max(abs(v) for e in ends for v in e) > MAX_COORD_M or hv[0] > MAX_DIM_M or math.dist(*ends) > MAX_DIM_M:
            raise ScanBundleError(
                f"wall {wid}: longer than {MAX_DIM_M:g} m or beyond {MAX_COORD_M:g} m from the origin"
            )
        if not isinstance(wid, str) or not wid:  # the twin's layout and the sim's wall box are named after it
            raise ScanBundleError(f"wall {n}: id must be a non-empty string, got {wid!r}")
        _mujoco_name("wall", wid)
    for key in ("fixtures", "objects"):
        for item in bundle.get(key, []):
            if not isinstance(item, dict):
                raise ScanBundleError(f"{key} entries must be objects")
            what = f"{key[:-1]} {item.get('id')!r}"
            if not isinstance(item.get("id"), str) or not item["id"]:
                raise ScanBundleError(f"{what}: id must be a non-empty string")
            _mujoco_name(key[:-1], item["id"])
            label = "category" if key == "fixtures" else "label"
            if not isinstance(item.get(label), str) or not item[label]:
                raise ScanBundleError(f"{what}: {label} must be a non-empty string")
            size = numbers(item.get("size"), 3)
            if size is None or min(size) <= 0 or max(size) > MAX_DIM_M:
                raise ScanBundleError(
                    f"{what}: size must be 3 numbers, each above 0 and at most {MAX_DIM_M:g} m, "
                    f"got {item.get('size')!r}"
                )
            pos = numbers(item.get("position"), 3)
            if pos is None or max(abs(v) for v in pos) > MAX_COORD_M:
                raise ScanBundleError(
                    f"{what}: position must be 3 numbers within {MAX_COORD_M:g} m of the origin, "
                    f"got {item.get('position')!r}"
                )
            if numbers([item.get("yaw", 0.0)], 1) is None:
                raise ScanBundleError(f"{what}: yaw must be a finite number")
    for key in ("doors", "windows", "openings"):
        for item in bundle.get(key, []):
            size, pos = numbers(item.get("size"), 3), numbers(item.get("position"), 3)
            if (
                size is None
                or min(size) < 0
                or max(size) > MAX_DIM_M
                or pos is None
                or max(abs(v) for v in pos) > MAX_COORD_M
            ):
                raise ScanBundleError(
                    f"{key[:-1]} {item.get('id')!r}: size and position must be finite and within {MAX_DIM_M:g} m"
                )
    for fl in bundle.get("floors", []):
        if any(numbers(c, 2) is None or max(abs(v) for v in c) > MAX_COORD_M for c in fl["corners"]):
            raise ScanBundleError(
                f"{fl.get('id')}: floor corners must be finite [x, y] points within {MAX_COORD_M:g} m"
            )
    ids = [f["id"] for f in bundle.get("fixtures", [])] + [o["id"] for o in bundle.get("objects", [])]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ScanBundleError(f"duplicate ids {dup}: every fixture and object needs its own id")
    walls = bundle.get("walls", [])
    # a wall's box and an object's geom are both named by id in the sim's MuJoCo model (a fixture is a body, so a
    # wall may share a fixture's id); a repeated name fails the model build of every task
    taken = [w["id"] for w in walls] + [o["id"] for o in bundle.get("objects", [])]
    dup = sorted({w["id"] for w in walls if taken.count(w["id"]) > 1})
    if dup:
        raise ScanBundleError(f"duplicate ids {dup}: every wall needs an id that no other wall and no object has")
    if walls:
        xs = [p[0] for w in walls for p in (w["start"], w["end"])]
        ys = [p[1] for w in walls for p in (w["start"], w["end"])]
        w_x, w_y = max(xs) - min(xs), max(ys) - min(ys)
        if max(w_x, w_y) > MAX_ROOM_M or min(w_x, w_y) < 0 or max(w_x, w_y) < MIN_ROOM_M:
            raise ScanBundleError(
                f"the walls span {w_x:.2f} m by {w_y:.2f} m; a room scan is between {MIN_ROOM_M:g} m "
                f"and {MAX_ROOM_M:g} m across (a unit mix-up, or the wrong scan?)"
            )


def validate(bundle: dict) -> dict:
    """Check a ScanBundle's header (schema, units, frame, geometry source, room id) and that every list and entry has
    the shape the compiler reads; returns the bundle. Raises ScanBundleError on the first problem."""
    if not isinstance(bundle, dict):
        raise ScanBundleError(f"a scan bundle is a JSON object, got {type(bundle).__name__}")
    if bundle.get("schema_version") != "1":
        raise ScanBundleError(f"schema_version must be '1', got {bundle.get('schema_version')!r}")
    if bundle.get("units") != "meters":
        raise ScanBundleError(f"units must be 'meters', got {bundle.get('units')!r}")
    if bundle.get("frame") != "z_up":
        raise ScanBundleError(f"frame must be 'z_up', got {bundle.get('frame')!r}")
    if bundle.get("geometry_source") not in GEOMETRY_SOURCES:
        raise ScanBundleError(f"geometry_source must be one of {GEOMETRY_SOURCES}")
    if not isinstance(bundle.get("blocks_claims", []), list):
        raise ScanBundleError("blocks_claims must be a list")
    if not bundle.get("room_id"):
        raise ScanBundleError("room_id is required")
    for key in ("walls", "fixtures", "objects") + OPTIONAL_LISTS:
        if not isinstance(bundle.get(key, []), list):
            raise ScanBundleError(f"{key} must be a list")
    for key in ("fixtures", "objects", "doors", "windows", "openings"):
        for item in bundle.get(key, []):
            if not isinstance(item, dict):
                raise ScanBundleError(f"{key} entries must be objects")
            if any(not isinstance(item.get(k), (list, tuple)) or len(item[k]) != 3 for k in ("size", "position")):
                raise ScanBundleError(f"{item.get('id')}: size and position must have 3 values")
    for fl in bundle.get("floors", []):
        if not isinstance(fl, dict):
            raise ScanBundleError("floors entries must be objects")
        corners = fl.get("corners")
        if (
            not isinstance(corners, (list, tuple))
            or len(corners) < 3
            or any(not isinstance(c, (list, tuple)) or len(c) != 2 for c in corners)
        ):
            raise ScanBundleError(f"{fl.get('id')}: floor corners must be 3 or more [x, y] points")
    for key in ("story", "story_count"):
        if key in bundle and not isinstance(bundle[key], int):
            raise ScanBundleError(f"{key} must be an integer")
    _check_geometry(bundle)
    bundle.setdefault("blocks_claims", [])
    return bundle


def resolve_scans(scans_dir: str | Path = "scans") -> Path:
    """A relative --scans that does not exist in the cwd falls back to <repo root>/<scans_dir>."""
    p = Path(scans_dir)
    if p.is_absolute() or p.exists():
        return p
    repo = Path(__file__).resolve().parents[2] / p
    return repo if repo.exists() else p


def load(path: str | Path) -> dict:
    """Read and validate scan.json. The result carries `input_sha256`, the sha256 of the file's bytes, for provenance."""
    with open(path, encoding="utf-8") as f:
        bundle = validate(json.load(f))
    bundle["input_sha256"] = sha256_file(path)
    return bundle


def load_room(room_id: str, scans_dir: str | Path = "scans") -> dict:
    """scans/<room_id>/scan.json, named after its folder as `demo`, `summary` and an upload name it. A scan.json copied
    from another room still says that room's room_id, and sim wrote over that room's dataset and reports; the file's
    own room_id is kept as source_room_id when it differs."""
    bundle = load(Path(scans_dir) / room_id / "scan.json")
    if bundle["room_id"] != room_id:
        bundle["source_room_id"], bundle["room_id"] = bundle["room_id"], room_id
    return bundle


FIXTURE_SCANS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "scans"


def rooms_hint(room_id: str, scans_dir: str | Path = "scans") -> str:
    """For a 'room not found' error: the rooms that are under scans_dir, and the --scans a test room needs."""
    scans_dir = Path(scans_dir)
    found = (
        sorted(p.name for p in scans_dir.iterdir() if (p / "scan.json").is_file() or (p / "roomplan.json").is_file())
        if scans_dir.is_dir()
        else []
    )
    hint = f"rooms under {scans_dir}: {', '.join(found)}" if found else f"no rooms under {scans_dir}"
    if (FIXTURE_SCANS / room_id).is_dir() and not (scans_dir.is_dir() and scans_dir.samefile(FIXTURE_SCANS)):
        hint += f"; {room_id} is a test room, add --scans tests/fixtures/scans"
    return hint
