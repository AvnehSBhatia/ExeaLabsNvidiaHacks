"""Multi-room summary table (arrival summary): stages, failure reporting, honest not-measured cells."""

import json
from pathlib import Path

import pytest

from arrival import cli
from arrival.eval import summary

REPO = Path(__file__).resolve().parents[1]
SCANS = REPO / "scans"
FAMS = ("open_drawer",)


def statuses(row):
    return {k: v["status"] for k, v in row["stages"].items()}


def test_resolve_room_by_name_and_by_path():
    assert summary.resolve_room("hero_kitchen", SCANS) == ("hero_kitchen", SCANS / "hero_kitchen" / "scan.json")
    room, path = summary.resolve_room("tests/fixtures/roomplan/amal.json", SCANS)
    assert room == "amal" and path.is_file()
    with pytest.raises(FileNotFoundError):
        summary.resolve_room("no_such_room", SCANS)
    with pytest.raises(FileNotFoundError):
        summary.resolve_room("nope.json", SCANS)



def test_generic_scan_file_names_take_the_folder_name():
    # two rooms given as scan.json files were both "scan", and their datasets shared scan_sim
    a = summary.resolve_room("tests/fixtures/scans/rotated_kitchen/scan.json", SCANS)
    b = summary.resolve_room("tests/fixtures/scans/u_kitchen_42/scan.json", SCANS)
    assert (a[0], b[0]) == ("rotated_kitchen", "u_kitchen_42")


def test_missing_room_is_a_failed_cell_not_a_crash(tmp_path):
    row = summary.run_room("no_such_room", tmp_path, SCANS)
    assert statuses(row) == {"normalize": "failed", **dict.fromkeys(summary.STAGES[1:], "not_run")}
    assert "no_such_room" in row["stages"]["normalize"]["detail"]
    assert row["stages"]["compile"]["detail"] == "normalize failed"


def test_bad_json_fails_at_normalize(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    row = summary.run_room(str(bad), tmp_path / "out", SCANS)
    assert row["stages"]["normalize"]["status"] == "failed" and "not a JSON object" in row["stages"]["normalize"]["detail"]


def test_no_reachable_task_is_skipped_not_failed(tmp_path, monkeypatch):
    import arrival.sim.tasks as tasks

    monkeypatch.setattr(tasks, "make_tasks", lambda *a, **k: [])
    row = summary.run_room("hero_kitchen", tmp_path, SCANS, families=FAMS)
    assert statuses(row)["compile"] == "ok" and statuses(row)["tasks"] == "skipped"
    assert statuses(row)["sim"] == "not_run" == statuses(row)["validate"]


def test_sim_exception_is_recorded_with_reason(tmp_path, monkeypatch):
    import arrival.sim.collect as collect

    def boom(*a, **k):
        raise RuntimeError("mujoco exploded")

    monkeypatch.setattr(collect, "collect", boom)
    row = summary.run_room("hero_kitchen", tmp_path, SCANS, families=FAMS)
    assert statuses(row)["tasks"] == "ok" and statuses(row)["sim"] == "failed"
    assert "RuntimeError: mujoco exploded" in row["stages"]["sim"]["detail"]
    assert statuses(row)["validate"] == "not_run"


def test_zero_successes_is_a_failed_sim_with_counts(tmp_path, monkeypatch):
    import arrival.sim.collect as collect

    rep = {"attempts": 2, "successes": 0, "human_contacts": 0}
    monkeypatch.setattr(collect, "collect", lambda *a, log=print, **k: (
        log("[sim] t0: MISS {'joint': 'd'} ik_err=0.0"), {"report_data": rep, "dataset": tmp_path})[1])
    row = summary.run_room("hero_kitchen", tmp_path, SCANS, families=FAMS)
    sim = row["stages"]["sim"]
    assert sim["status"] == "failed" and (sim["attempts"], sim["successes"]) == (2, 0)
    assert sim["misses"] and "MISS" in sim["misses"][0]
    assert statuses(row)["episodes"] == "not_run"


def test_real_hero_kitchen_row_end_to_end(tmp_path):
    pytest.importorskip("mujoco")
    row = summary.run_room("hero_kitchen", tmp_path, SCANS, episodes=1, res=64, people=0, families=FAMS)
    assert statuses(row) == dict.fromkeys(summary.STAGES, "ok"), row["stages"]
    assert row["stages"]["sim"]["arrival_physically_valid"] is True
    assert row["stages"]["compile"]["objects_default_synthetic"] is False  # hero_kitchen's objects are authored
    assert row["geometry_source"]
    assert row["tape"]["tape_measured"] is False  # tests/fixtures/measurements/hero_kitchen.json is authored
    assert row["realism"] is None and "not measured" in row["realism_note"]


def test_roomplan_rows_flag_invented_objects(tmp_path, monkeypatch):
    import arrival.sim.tasks as tasks

    monkeypatch.setattr(tasks, "make_tasks", lambda *a, **k: [])  # compile is the point; skip the sim
    row = summary.run_room("tests/fixtures/roomplan/shopyseek.json", tmp_path, SCANS)
    c = row["stages"]["compile"]
    assert c["status"] == "ok" and c["objects_default_synthetic"] is True and "invented" in c["detail"]
    assert row["tape"] is None  # no measurements exist for this room


def test_markdown_table_has_a_row_per_room_and_honest_cells():
    row = {"room": "r1", "geometry_source": "roomplan", "tape": None, "realism": None,
           "stages": {s: {"status": "ok", "detail": ""} for s in summary.STAGES}}
    row["stages"]["sim"].update(attempts=2, successes=1)
    row2 = {"room": "r2", "geometry_source": "authored", "tape": {"max_abs_error_m": 0.0, "tape_measured": False,
                                                                 "ok": True, "tolerance_m": 0.05},
            "realism": None, "stages": {s: {"status": "not_run", "detail": "compile failed"} for s in summary.STAGES}}
    row2["stages"]["compile"] = {"status": "failed", "detail": "ValueError: x"}
    md = summary.to_markdown({"settings": {"episodes_per_room": 2, "res": 64, "people": 1}, "rooms": [row, row2]})
    lines = [ln for ln in md.splitlines() if ln.startswith(("| r1", "| r2"))]
    assert len(lines) == 2
    assert "ok 1/2" in lines[0] and "not measured" in lines[0]
    assert "FAILED" in lines[1] and "authored, not tape" in lines[1]
    assert "Realism: not measured" in md


def test_cli_summary_writes_json_and_markdown(tmp_path, capsys):
    pytest.importorskip("mujoco")
    rc = cli.main(["summary", "--rooms", "hero_kitchen", "--episodes", "1", "--people", "0", "--families", "open_drawer",
                   "--out", str(tmp_path), "--scans", str(SCANS)])
    assert rc == cli.EXIT_OK
    data = json.loads((tmp_path / "summary.json").read_text())
    assert [r["room"] for r in data["rooms"]] == ["hero_kitchen"] and data["settings"]["res"] == 64
    assert (tmp_path / "summary.md").read_text().startswith("# Room summary")
    assert cli.main(["summary", "--families", "bogus", "--out", str(tmp_path)]) == cli.EXIT_USAGE
