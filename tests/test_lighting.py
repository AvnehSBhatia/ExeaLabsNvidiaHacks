"""SIM-2: brightness follows the lighting label on every seed, the seed moves the lights, materials are attached."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.scene import LIGHT, build_mjcf  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

BASE = (1.3, 1.0, 0.75)
SEEDS = (0, 1, 2)


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


def _model(twin, **kw):
    return mujoco.MjModel.from_xml_string(build_mjcf(twin, BASE, **kw))


def _mean_brightness(twin, lighting, seed) -> float:
    m = _model(twin, lighting=lighting, n_people=1, seed=seed)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 120, 160)
    r.update_scene(d, "overview")
    return float(r.render().mean())


def test_brightness_follows_the_label_on_every_seed(twin):
    means = {lab: [_mean_brightness(twin, lab, s) for s in SEEDS] for lab in LIGHT}
    assert max(means["dim"]) < min(means["neutral"]) and max(means["neutral"]) < min(means["bright"])
    assert 20 < min(means["dim"]) and max(means["bright"]) < 200
    for vs in means.values():  # the seed changes the lights inside a label
        assert max(vs) - min(vs) > 1.0


def test_seed_jitters_the_lights_and_no_seed_is_nominal(twin):
    m0, m1 = _model(twin, seed=0), _model(twin, seed=1)
    assert not np.allclose(m0.light("key").pos, m1.light("key").pos)
    assert not np.allclose(m0.light("key").diffuse, m1.light("key").diffuse)
    assert build_mjcf(twin, BASE) == build_mjcf(twin, BASE)
    assert m0.nlight == 3 and m0.light("key").castshadow[0] and not m0.light("fill").castshadow[0]


def test_materials_are_attached_by_surface(twin):
    m = _model(twin)
    mats = [(m.geom(i).name, m.material(int(m.geom_matid[i])).name) for i in range(m.ngeom) if m.geom_matid[i] >= 0]
    named = dict(mats)
    assert named["floor"] == "floor"
    assert named[twin["walls"][0]["id"]] == "paint"
    assert {v for k, v in mats if "_top" in k} == {"stone"}
    counts = [v for _, v in mats]
    assert counts.count("wood") >= 4 and "panel" in counts  # fronts and backsplash are unnamed visual geoms
    assert m.ntex == 4 and all(m.tex(i).name for i in range(m.ntex))  # builtin textures only, no files
