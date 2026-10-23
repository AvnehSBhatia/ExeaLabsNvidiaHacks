"""`arrival subset`: drop marginal or unwanted episodes of a sim dataset, and say so."""

import json
import math

import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.export.subset import SubsetError, subset_dataset  # noqa: E402
from arrival.export.validate import validate  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import collect  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    twin = compile_twin(load_room("hero_kitchen", "scans"))
    out = tmp_path_factory.mktemp("src")
    collect(twin, ["counter_to_sink", "open_drawer"], 6, out, people=0, res=32, seed=4, exec_noise=0.02, log=lambda *_: None)
    return out / "datasets" / "hero_kitchen_sim"


def _info(ds):
    return json.loads((ds / "meta" / "info.json").read_text())


def _families(ds):
    return [json.loads(line)["arrival_family"] for line in (ds / "meta" / "episodes.jsonl").read_text().splitlines()]


def test_family_filter_reindexes_validates_and_keeps_the_label(source, tmp_path):
    n_drawer = _families(source).count("open_drawer")
    res = subset_dataset(source, tmp_path, families=["open_drawer"], log=lambda *_: None)
    assert _families(res["dataset"]) == ["open_drawer"] * n_drawer and res["kept"] == n_drawer
    assert not validate(res["dataset"])
    eps = [json.loads(line) for line in (res["dataset"] / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["episode_index"] for e in eps] == list(range(n_drawer))  # contiguous again
    info = _info(res["dataset"])
    assert info["arrival_exec_noise_rad"] == 0.02  # still labelled as perturbed
    assert info["arrival_subset_rule"] == {"families": ["open_drawer"], "min_drawer_opening_m": None}
    assert info["arrival_subset_source_episodes"] == len(_families(source)) and info["arrival_subset_dropped"]["family"] > 0


def test_drawer_opening_floor_drops_only_marginal_drawers(source, tmp_path):
    openings = [(json.loads(line).get("arrival_success_detail") or {}).get("qpos")
                for line in (source / "meta" / "episodes.jsonl").read_text().splitlines()
                if json.loads(line)["arrival_family"] == "open_drawer"]
    floor = sorted(openings)[len(openings) // 2]  # the median: about half the drawer pulls fall under it
    res = subset_dataset(source, tmp_path, min_drawer_opening=floor, log=lambda *_: None)
    dropped = res["dropped"]["drawer_opening"]
    assert dropped == sum(1 for q in openings if q < floor) and dropped > 0
    fams = _families(res["dataset"])
    assert fams.count("counter_to_sink") == _families(source).count("counter_to_sink")  # other families untouched
    assert fams.count("open_drawer") == len(openings) - dropped
    assert not validate(res["dataset"])


def test_refuses_an_empty_result_and_overwriting_the_source(source, tmp_path):
    with pytest.raises(SubsetError, match="keeps none"):
        subset_dataset(source, tmp_path, families=["no_such_family"], log=lambda *_: None)
    with pytest.raises(SubsetError, match="overwrite"):
        subset_dataset(source, source.parent.parent, log=lambda *_: None)

    with pytest.raises(SubsetError, match="inside the source"):
        subset_dataset(source, source, log=lambda *_: None)
    assert not (source / "datasets").exists()


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -0.1, True, "0.28"])
def test_rejects_invalid_drawer_threshold_before_reading_source(tmp_path, threshold):
    with pytest.raises(SubsetError, match="min_drawer_opening"):
        subset_dataset(tmp_path / "missing", tmp_path / "out", min_drawer_opening=threshold)
    assert not (tmp_path / "out").exists()


def test_cli_reports_a_folder_with_no_dataset_instead_of_a_traceback(tmp_path, capsys):
    from arrival.cli import main

    (tmp_path / "empty").mkdir()
    assert main(["subset", str(tmp_path / "empty"), "--out", str(tmp_path / "out")]) == 2
    assert "subset: " in capsys.readouterr().err
