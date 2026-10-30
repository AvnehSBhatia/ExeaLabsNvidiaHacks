"""A double quote or a NUL in a wall, fixture or object id is a bad scan.

Those ids are MuJoCo names: a wall geom (`name="{id}"`), a fixture body, an object geom. A
double quote or a NUL is not a legal attribute value, so the model does not load
(`XML parse error`) and the id is not in the message. Validation used to accept the scan.
A space is a legal name and still loads. A door id is not used as a MuJoCo name.
"""

import json

import pytest
from conftest import SCANS

from arrival.jobs.worker import load_upload, retryable
from arrival.scan import bundle as scan_bundle

BAD = {
    "wall double quote": lambda d: d["walls"][0].update(id='wall"north'),
    "fixture double quote": lambda d: d["fixtures"][0].update(id='counter"a'),
    "object double quote": lambda d: d["objects"][0].update(id='mug"1'),
    "wall NUL": lambda d: d["walls"][0].update(id="wall\x00north"),
    "fixture NUL": lambda d: d["fixtures"][0].update(id="counter\x00a"),
    "object NUL": lambda d: d["objects"][0].update(id="mug\x001"),
}


def _hero(mutate=None):
    data = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(data)
    return data


@pytest.mark.parametrize("mutate", BAD.values(), ids=list(BAD))
def test_quote_or_nul_in_an_id_is_a_scan_error(mutate):
    with pytest.raises(scan_bundle.ScanBundleError, match="double quote or a NUL"):
        scan_bundle.validate(_hero(mutate))


@pytest.mark.parametrize("mutate", BAD.values(), ids=list(BAD))
def test_upload_with_a_quote_or_nul_id_is_refused_and_not_retried(mutate, tmp_path):
    upload = tmp_path / "upload.bin"
    upload.write_text(json.dumps(_hero(mutate)), encoding="utf-8")
    with pytest.raises(scan_bundle.ScanBundleError, match="double quote or a NUL") as e:
        load_upload(upload, "room_job")
    assert not retryable(e.value)


def test_a_space_in_a_wall_id_still_loads():
    mujoco = pytest.importorskip("mujoco")
    from arrival.sim.scene import build_mjcf
    from arrival.twin.compile import compile_twin

    bundle = scan_bundle.validate(_hero(lambda d: d["walls"][0].update(id="wall north")))
    model = mujoco.MjModel.from_xml_string(build_mjcf(compile_twin(bundle), (1.3, 1.0, 0.75)))
    assert model.ngeom > 0


def test_a_door_id_with_a_quote_is_not_a_mujoco_name():
    """Portals are named door{k}_part, not the scan id, so a quote there still loads."""
    mujoco = pytest.importorskip("mujoco")
    from arrival.sim.scene import build_mjcf
    from arrival.twin.compile import compile_twin

    def add_door(d):
        d["doors"] = [
            {
                "id": 'door"1',
                "wall_id": "wall_north",
                "size": [0.9, 0.0, 2.0],
                "position": [2.0, 0.0, 1.0],
                "yaw": 0.0,
                "is_open": False,
            }
        ]

    bundle = scan_bundle.validate(_hero(add_door))
    assert bundle["doors"][0]["id"] == 'door"1'
    model = mujoco.MjModel.from_xml_string(build_mjcf(compile_twin(bundle), (1.3, 1.0, 0.75)))
    assert model.ngeom > 0
