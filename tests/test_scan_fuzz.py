"""L11: fuzzed RoomPlan-like exports go through normalize -> compile -> default objects -> layout -> tasks -> MuJoCo
scene. Every input either becomes a twin that passes sanity checks or is refused with a ScanBundleError whose message
a person can read; nothing may escape as a KeyError, TypeError, IndexError, XML error or a traceback.

The variants are seeded, so a failure names its seed and reproduces with `_fuzz_export(random.Random(seed))`."""

import copy
import json
import math
import os
import random

import pytest
import yaml

from arrival.catalog.fixtures import default_size
from arrival.scan import roomplan
from arrival.scan.bundle import ScanBundleError
from arrival.tasks.propose import propose
from arrival.twin.compile import compile_twin, layout_dict, place_default_objects

CATEGORIES = ("bathtub", "bed", "chair", "dishwasher", "fireplace", "oven", "refrigerator", "sink", "sofa", "stairs",
              "storage", "stove", "table", "television", "toilet", "washerDryer")
FLOOR_Y = -1.4
H = 2.5
N_FUZZ = int(os.environ.get("ARRIVAL_FUZZ_N", 300))  # a longer sweep: ARRIVAL_FUZZ_N=5000 pytest tests/test_scan_fuzz.py


def _t(x, y, z, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y, z, 1]


def _wall(i, x0, z0, x1, z1):
    length = math.hypot(x1 - x0, z1 - z0)
    yaw = -math.atan2(z1 - z0, x1 - x0)
    return {"identifier": f"wall{i}", "category": {"wall": {}}, "confidence": {"high": {}},
            "dimensions": [length, H, 0], "transform": _t((x0 + x1) / 2, FLOOR_Y + H / 2, (z0 + z1) / 2, yaw)}


def _obj(ident, cat, x, z, yaw=0.0, size=None, confidence="high"):
    sx, sy, sz = size or default_size(roomplan._snake(cat))
    return {"identifier": ident, "category": {cat: {}}, "confidence": {confidence: {}},
            "dimensions": [sx, sz, sy], "transform": _t(x, FLOOR_Y + sz / 2, z, yaw)}


def _room(w=5.0, d=4.0, objects=(), open_side=None, extra_walls=()):
    corners = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    walls = [_wall(i, *corners[i], *corners[(i + 1) % 4]) for i in range(4) if i != open_side]
    return {"walls": walls + list(extra_walls), "floors": [], "doors": [], "windows": [], "openings": [],
            "objects": list(objects)}


def _rotate(rp, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    out = copy.deepcopy(rp)
    for k in ("walls", "objects"):
        for item in out[k]:
            m = item["transform"]
            x, z = m[12], m[14]
            m[12], m[14] = c * x - s * z, s * x + c * z
            yaw = math.atan2(-m[2], m[0]) - a
            m[0], m[2], m[8], m[10] = math.cos(yaw), -math.sin(yaw), math.sin(yaw), math.cos(yaw)
    return out


def _mirror(rp):
    out = copy.deepcopy(rp)
    for k in ("walls", "objects"):
        for item in out[k]:
            m = item["transform"]
            m[12] = -m[12]
            yaw = -math.atan2(-m[2], m[0])
            m[0], m[2], m[8], m[10] = math.cos(yaw), -math.sin(yaw), math.sin(yaw), math.cos(yaw)
    return out


def _scale_room(w, d):
    objs = [_obj("sink", "sink", w / 2 - 0.5, 0.0), _obj("fridge", "refrigerator", -w / 2 + 0.5, 0.0)]
    return _room(w, d, objs)


def _u_kitchen():
    """The shape reported in #42: a U of base cabinets on three walls, one wall open."""
    objs = [_obj("sink", "sink", 0.0, -1.5, size=[0.6, 0.5, 0.25]),
            _obj("base_n", "storage", 0.0, -1.7, size=[1.2, 0.6, 0.9]),
            _obj("base_e", "storage", 1.9, 0.0, math.pi / 2, size=[1.4, 0.6, 0.9]),
            _obj("stove", "stove", -1.5, -1.65, size=[0.7, 0.65, 0.9]),
            _obj("fridge", "refrigerator", -2.0, 1.2, size=[0.9, 0.7, 1.8])]
    return _room(5.0, 4.0, objs, open_side=2)


def _island_kitchen():
    objs = [_obj("sink", "sink", 0.0, -1.7, size=[0.6, 0.5, 0.25]),
            _obj("island", "table", 0.0, 0.3, size=[2.0, 0.9, 0.92]),
            _obj("fridge", "refrigerator", -2.0, -1.5, size=[0.9, 0.7, 1.8])]
    return _room(5.0, 4.0, objs)


NAMED = {
    "open_wall": lambda: _room(objects=[_obj("sink", "sink", 0.0, -1.7), _obj("c", "storage", 1.0, -1.7)], open_side=3),
    "rotated_30": lambda: _rotate(_u_kitchen(), 30),
    "rotated_90": lambda: _rotate(_u_kitchen(), 90),
    "rotated_odd": lambda: _rotate(_island_kitchen(), 137),
    "mirrored": lambda: _mirror(_u_kitchen()),
    "no_sink": lambda: _room(objects=[_obj("c", "storage", 0.0, -1.7), _obj("o", "oven", 1.0, -1.7)]),
    "many_sinks": lambda: _room(objects=[_obj(f"s{i}", "sink", -2.0 + 0.9 * i, -1.7) for i in range(5)]),
    "island": _island_kitchen,
    "no_objects": lambda: _room(),
    "tiny_room": lambda: _scale_room(1.2, 1.0),
    "large_room": lambda: _scale_room(14.0, 10.0),
    "u_kitchen": _u_kitchen,
}
# A structure no single-room kitchen sim can use: these must be refused with a readable reason, not compiled.
REFUSED = {
    "huge_room": lambda: _scale_room(900.0, 800.0),
    "specks": lambda: _scale_room(0.05, 0.04),
}


def _surface_items(rp):
    return [i for k in ("walls", "doors", "windows", "openings", "floors", "objects")
            for i in (rp.get(k) if isinstance(rp.get(k), list) else []) if isinstance(i, dict)]


def _mutate(rp, rng):
    """One random corruption of a valid export."""
    if not isinstance(rp, dict) or "rooms" in rp:
        return rp  # already unrecognisable
    kind = rng.choice(["drop_dim", "zero_dim", "neg_dim", "nan_dim", "huge_dim", "short_dim", "drop_transform",
                       "nan_transform", "short_transform", "inf_transform", "string_transform", "drop_walls",
                       "walls_none", "objects_none", "objects_not_list", "item_not_dict", "dup_ids", "bad_category",
                       "no_category", "weird_id", "bad_story", "huge_coord", "same_wall_ends", "top_level_list",
                       "rooms_garbage", "drop_ident", "bad_confidence", "text_dim", "floor_bad_polygon"])
    items = _surface_items(rp)
    pick = rng.choice(items) if items else None
    if kind == "top_level_list":
        return [rp]
    if kind == "drop_walls":
        rp["walls"] = []
    elif kind == "walls_none":
        rp["walls"] = None
    elif kind == "objects_none":
        rp["objects"] = None
    elif kind == "objects_not_list":
        rp["objects"] = {"a": 1}
    elif kind == "rooms_garbage":
        return {"rooms": [None, 3, {"walls": "x"}]}
    elif pick is None:
        return rp
    elif kind in ("nan_transform", "inf_transform", "short_transform", "huge_coord") and not (
            isinstance(pick.get("transform"), list) and len(pick["transform"]) == 16):
        return rp  # the transform was already corrupted by an earlier mutation
    elif kind == "drop_dim":
        pick.pop("dimensions", None)
    elif kind == "zero_dim":
        pick["dimensions"] = [0.0, 0.0, 0.0]
    elif kind == "neg_dim":
        pick["dimensions"] = [-1.0, 0.5, -2.0]
    elif kind == "nan_dim":
        pick["dimensions"] = [float("nan"), 1.0, 1.0]
    elif kind == "huge_dim":
        pick["dimensions"] = [1e6, 1e6, 1e6]
    elif kind == "short_dim":
        pick["dimensions"] = [1.0, 1.0]
    elif kind == "text_dim":
        pick["dimensions"] = ["a", "b", "c"]
    elif kind == "drop_transform":
        pick.pop("transform", None)
    elif kind == "nan_transform":
        pick["transform"][12] = float("nan")
    elif kind == "inf_transform":
        pick["transform"][14] = float("inf")
    elif kind == "short_transform":
        pick["transform"] = pick["transform"][:9]
    elif kind == "string_transform":
        pick["transform"] = "identity"
    elif kind == "item_not_dict":
        rp["objects"] = list(rp.get("objects") or []) + [None, 5]
    elif kind == "dup_ids":
        for item in items:
            item["identifier"] = "same"
    elif kind == "bad_category":
        pick["category"] = rng.choice([{"spaceship": {}}, 7, [], "refrigerator", {}, None])
    elif kind == "no_category":
        pick.pop("category", None)
    elif kind == "weird_id":
        pick["identifier"] = rng.choice(['a"b', "<x>", "with space", "", 5, None, "../etc"])
    elif kind == "drop_ident":
        pick.pop("identifier", None)
    elif kind == "bad_story":
        pick["story"] = rng.choice(["abc", [], 1.5, {"a": 1}])
    elif kind == "huge_coord":
        pick["transform"][12] = 5e7
    elif kind == "same_wall_ends":
        if rp["walls"]:
            rp["walls"][0]["dimensions"][0] = 0.0
    elif kind == "bad_confidence":
        pick["confidence"] = rng.choice(["low", 3, None, [], {"low": {}}])
    elif kind == "floor_bad_polygon":
        rp["floors"] = [{"identifier": "f", "category": {"floor": {}}, "dimensions": [1, 1, 0],
                         "transform": _t(0, FLOOR_Y, 0), "polygonCorners": rng.choice([[[0, 0], [1, 0], [1, 1]], "x",
                                                                                         [[0, 0, 0]] * 3 + [None]])}]
    return rp


def _fuzz_export(rng):
    """A random plausible kitchen, rotated and maybe mirrored, then zero to three corruptions."""
    w, d = rng.uniform(1.5, 9.0), rng.uniform(1.5, 8.0)
    objs = []
    for i in range(rng.randint(0, 9)):
        cat = rng.choice(CATEGORIES)
        objs.append(_obj(f"o{i}", cat, rng.uniform(-w / 2, w / 2), rng.uniform(-d / 2, d / 2),
                         rng.choice([0.0, math.pi / 2, math.pi, rng.uniform(0, 6.28)]),
                         confidence=rng.choice(["high", "high", "medium", "low"])))
    rp = _room(w, d, objs, open_side=rng.choice([None, None, 0, 1, 2, 3]))
    rp = _rotate(rp, rng.choice([0, 0, 90, rng.uniform(0, 360)]))
    if rng.random() < 0.3:
        rp = _mirror(rp)
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
        rp = _mutate(rp, rng)
    return rp


def _all_numbers(x):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _all_numbers(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _all_numbers(v)


def _pipeline(rp, build_scene=True):
    """Everything a worker does before the sim. Returns the twin; ScanBundleError is the one allowed refusal."""
    twin = compile_twin(roomplan.to_bundle(rp, "fuzz"))
    place_default_objects(twin, seed=1)
    yaml.safe_dump(layout_dict(twin))
    json.dumps(twin)
    propose(twin)
    if build_scene:
        mujoco = pytest.importorskip("mujoco")
        from arrival.sim.scene import build_mjcf

        xs = [p[0] for w in twin["walls"] for p in (w["start"], w["end"])] or [0.0]
        ys = [p[1] for w in twin["walls"] for p in (w["start"], w["end"])] or [0.0]
        mujoco.MjModel.from_xml_string(build_mjcf(twin, ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.75)))
    return twin


def _sane(twin):
    assert all(math.isfinite(v) for v in _all_numbers(twin)), "non-finite number in the twin"
    for f in twin["fixtures"] + twin["objects"]:
        assert all(v > 0 for v in f["size"]), f"{f['id']} has a non-positive size {f['size']}"
    ids = [f["id"] for f in twin["fixtures"]]
    assert len(ids) == len(set(ids)), "duplicate fixture ids"
    for f in twin["fixtures"] + twin["objects"]:
        assert f["id"].replace("_", "").replace("-", "").replace(".", "").isalnum(), f"unsafe id {f['id']!r}"


def _check(rp, label):
    try:
        twin = _pipeline(copy.deepcopy(rp))
    except ScanBundleError as e:
        assert str(e).strip() and "Traceback" not in str(e), f"{label}: unreadable refusal {e!r}"
        return None
    except pytest.skip.Exception:
        raise
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"{label}: escaped as {type(e).__name__}: {e}")
    _sane(twin)
    return twin


@pytest.mark.parametrize("name", sorted(NAMED))
def test_named_variants_give_a_valid_twin(name):
    """Open walls, rotated or mirrored rooms, no sink, many sinks, an island, tiny and large rooms all compile."""
    twin = _pipeline(NAMED[name]())
    _sane(twin)
    if name not in ("no_sink", "no_objects", "tiny_room"):  # no sink or no fixtures at all: nothing to propose
        assert propose(twin)["tasks"]


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_implausible_rooms_are_refused_readably(name):
    with pytest.raises(ScanBundleError, match=r"\bm\b|metre|size|extent"):
        roomplan.to_bundle(REFUSED[name](), "fuzz")


@pytest.mark.parametrize("seed", range(N_FUZZ))
def test_fuzzed_export_compiles_or_is_refused_cleanly(seed):
    _check(_fuzz_export(random.Random(seed)), f"seed {seed}")


def test_fuzz_hits_both_outcomes():
    """The sweep is not vacuous: some seeds compile, some are refused."""
    outcomes = {_check(_fuzz_export(random.Random(s)), f"seed {s}") is None for s in range(40)}
    assert outcomes == {True, False}


@pytest.mark.parametrize("mutation,needle", [
    (lambda rp: rp["walls"][0].pop("transform"), "wall0"),
    (lambda rp: rp["walls"][0]["transform"].__setitem__(12, float("nan")), "wall0"),
    (lambda rp: rp["walls"][0].__setitem__("dimensions", [float("inf"), 2.5, 0]), "wall0"),
    (lambda rp: rp["objects"][0].__setitem__("dimensions", [1e6, 1.0, 1.0]), "sink"),
    (lambda rp: rp.__setitem__("walls", "not a list"), "walls"),
])
def test_specific_corruptions_name_the_offender(mutation, needle):
    rp = _room(objects=[_obj("sink", "sink", 0.0, 0.0)])
    mutation(rp)
    with pytest.raises(ScanBundleError, match=needle):
        roomplan.to_bundle(rp, "fuzz")


def test_null_rooms_reads_like_no_rooms():
    rp = _room(objects=[_obj("sink", "sink", 0.0, 0.0)])
    assert roomplan.to_bundle({**rp, "rooms": None}, "fuzz") == roomplan.to_bundle(rp, "fuzz")


def test_zero_size_object_falls_back_to_catalog_default():
    rp = _room(objects=[_obj("sink", "sink", 0.0, 0.0)])
    rp["objects"][0]["dimensions"] = [0.0, 0.0, 0.0]
    b = roomplan.to_bundle(rp, "fuzz")
    assert b["fixtures"][0]["size"] == pytest.approx(default_size("sink"))


def test_duplicate_and_unsafe_ids_are_made_unique_and_safe():
    rp = _room(objects=[_obj('a"b', "storage", 0.0, 0.0), _obj('a"b', "storage", 1.0, 0.0)])
    ids = [f["id"] for f in roomplan.to_bundle(rp, "fuzz")["fixtures"]]
    assert len(set(ids)) == 2 and all('"' not in i for i in ids)
