import http.client
import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import pytest

from arrival.jobs.queue import create_job
from arrival.jobs.worker import load_upload, room_id_for, run_once, work
from arrival.web.app import make_server

REPO = Path(__file__).resolve().parents[1]
QUIET = lambda *_: None  # noqa: E731


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def test_room_id_from_filename():
    assert room_id_for("My Kitchen (2).json", "job_1a2b3c4d") == "my_kitchen_2_job_1a2b3c4d"
    assert room_id_for("", "job_1a2b3c4d") == "job_1a2b3c4d"
    assert room_id_for("../../etc.json", "job_00000000") == "etc_job_00000000"


def test_load_upload_detects_roomplan(tmp_path):
    p = tmp_path / "upload.bin"
    p.write_bytes((REPO / "scans" / "hero_kitchen" / "roomplan.json").read_bytes())
    b = load_upload(p, "r_job_00000000")
    assert b["geometry_source"] == "roomplan" and b["room_id"] == "r_job_00000000"
    p.write_bytes((REPO / "scans" / "hero_kitchen" / "scan.json").read_bytes())
    b = load_upload(p, "s_job_00000000")
    assert b["geometry_source"] == "authored_sample" and b["room_id"] == "s_job_00000000"
    p.write_text('{"room_id": "x"}')
    with pytest.raises(ValueError):
        load_upload(p, "x")


@pytest.mark.parametrize("change", [{"walls": 1}, {"walls": -2.5}, {"walls": [], "objects": True}])
def test_wrong_type_list_in_roomplan_fails_without_retry(tmp_path, change):
    data = json.loads((REPO / "scans" / "hero_kitchen" / "roomplan.json").read_text(encoding="utf-8"))
    data.update(change)
    p = tmp_path / "upload.bin"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_upload(p, "x")
    out = tmp_path / "out"
    create_job(out / "jobs", "kitchen.json", data=json.dumps(data).encode())
    st = run_once(out, log=QUIET)
    assert st["status"] == "failed" and not st.get("retry_errors"), st


def test_corrupt_upload_fails_and_worker_moves_on(tmp_path):
    out = tmp_path / "out"
    a = create_job(out / "jobs", "broken.json", data=b"this is not json")
    b = create_job(out / "jobs", "thin.json", data=b'{"schema_version": "1"}')
    first = run_once(out, log=QUIET)
    second = run_once(out, log=QUIET)
    assert run_once(out, log=QUIET) is None
    by_id = {s["id"]: s for s in (first, second)}
    assert set(by_id) == {a["id"], b["id"]}
    assert by_id[a["id"]]["status"] == "failed" and "JSONDecodeError" in by_id[a["id"]]["error"]
    assert by_id[b["id"]]["status"] == "failed" and "units" in by_id[b["id"]]["error"]
    for s in by_id.values():
        on_disk = json.loads((out / "jobs" / s["id"] / "status.json").read_text())
        assert on_disk["status"] == "failed" and on_disk["stage"] == "normalize"
        assert on_disk["started_at"] and on_disk["finished_at"] and on_disk["zip"] is None
        assert (out / "jobs" / s["id"] / "error.txt").is_file()
    assert work(out, once=True, log=QUIET) == 0


def test_failure_with_non_ascii_message_is_recorded(tmp_path):
    out = tmp_path / "out"
    job = create_job(out / "jobs", "units.json", data='{"schema_version": "1", "units": "мм"}'.encode())
    (out / "jobs" / "junk").mkdir()
    (out / "jobs" / "junk" / "status.json").write_text('{"status": "queued", "created_unix": "abc"}')
    st = run_once(out, log=QUIET)
    assert st["id"] == job["id"] and st["status"] == "failed" and "мм" in st["error"]
    on_disk = json.loads((out / "jobs" / job["id"] / "status.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "failed"
    assert "мм" in (out / "jobs" / job["id"] / "error.txt").read_text(encoding="utf-8")


@pytest.mark.render  # the sim renders in a child process, where --no-render's patch does not reach
def test_scan_json_job_becomes_dataset_zip(tmp_path):
    pytest.importorskip("mujoco")
    from arrival.export.validate import validate

    out = tmp_path / "out"
    scan = (REPO / "scans" / "hero_kitchen" / "scan.json").read_bytes()
    job = create_job(out / "jobs", "hero_kitchen.json", data=scan)
    from arrival.jobs.limits import Limits

    # in a child process under a wall-clock limit, the way `arrival work` runs it
    st = run_once(out, limits=Limits(max_job_seconds=600), isolate=True, episodes=2, families=["counter_to_sink"],
                  people=1, res=64, log=QUIET)
    assert st["status"] == "done", st
    info = json.loads((Path(st["dataset_dir"]) / "meta" / "info.json").read_text())
    u = st["usage"]  # metering: measured units, no price
    assert u["frames"] == info["total_frames"] > 0 and u["episodes_attempted"] == 2
    assert u["episodes_succeeded"] == st["successes"] and u["wall_seconds"] > 0 and set(u["stage_seconds"]) >= {"sim", "zip"}
    assert u["zip_bytes"] == (out / "jobs" / job["id"] / "dataset.zip").stat().st_size and u["dataset_bytes"] > 0
    assert len((out / "usage" / "jobs.jsonl").read_text().splitlines()) == 1
    assert st["room_id"] == f"hero_kitchen_{job['id']}"
    assert st["stage"] == "zip" and st["error"] is None and st["attempts"] == 2 and st["successes"] >= 1
    assert (Path(st["twin_dir"]) / "twin.json").is_file()
    assert Path(st["dataset_dir"]) == out / "datasets" / f"{st['room_id']}_sim"
    assert not validate(st["dataset_dir"])
    zip_path = out / "jobs" / job["id"] / "dataset.zip"
    assert zip_path.is_file() and st["zip"] == str(zip_path)
    names = zipfile.ZipFile(zip_path).namelist()
    assert f"{st['room_id']}_sim/meta/info.json" in names

    srv = make_server(port=0, out_dir=out, scans_dir=REPO / "scans")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        code, body = _get(port, f"/jobs/{job['id']}")
        assert code == 200 and json.loads(body)["status"] == "done"
        code, body = _get(port, f"/jobs/{job['id']}/download")
        assert code == 200 and zipfile.ZipFile(io.BytesIO(body)).namelist() == names
        code, body = _get(port, "/")
        assert code == 200 and job["id"].encode() in body and f"/jobs/{job['id']}/download".encode() in body
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_job_download_not_ready_is_404(tmp_path):
    out = tmp_path / "out"
    job = create_job(out / "jobs", "scan.json", data=b"{}")
    srv = make_server(port=0, out_dir=out, scans_dir=REPO / "scans")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        code, body = _get(port, f"/jobs/{job['id']}/download")
        assert code == 404 and json.loads(body)["status"] == "queued"
        code, body = _get(port, "/")
        assert code == 200 and job["id"].encode() in body and b"queued" in body
        assert _get(port, "/jobs/nope/download")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_status_is_never_read_half_written(tmp_path):
    """The worker rewrites status.json at every stage while the page and the capture app poll it."""
    import time

    from arrival.jobs.queue import read_job
    from arrival.jobs.worker import _write_status

    status = create_job(tmp_path, "k.json", data=b"{}")
    status["error"] = "x" * 20000  # a long traceback-sized field makes a torn read likely
    done = threading.Event()
    writer_errors = []
    writes = [0]

    def writer():
        try:
            while not done.is_set():
                _write_status(tmp_path / status["id"], status)
                writes[0] += 1
        except Exception as exc:  # surface errors from the background thread in the test
            writer_errors.append(exc)

    t = threading.Thread(target=writer)
    t.start()
    try:
        reads, t0 = 0, time.perf_counter()
        while reads <= 100 or time.perf_counter() - t0 < 0.5:
            assert read_job(tmp_path, status["id"])["id"] == status["id"]
            reads += 1
    finally:
        done.set()
        t.join()
    assert not t.is_alive()
    assert not writer_errors, writer_errors
    assert writes[0] > 0
    assert reads > 100
    assert [p.name for p in (tmp_path / status["id"]).iterdir() if p.name.endswith(".tmp")] == []


def test_load_upload_accepts_captured_structure_and_null_lists(tmp_path):
    import json

    flat = json.loads((REPO / "scans" / "hero_kitchen" / "roomplan.json").read_text())
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"rooms": [flat]}))
    assert load_upload(p, "st_job_00000000")["geometry_source"] == "roomplan"
    p.write_text(json.dumps({"walls": None, "objects": None}))
    with pytest.raises(ValueError):
        load_upload(p, "x")


def test_interrupted_job_is_not_left_running(tmp_path, monkeypatch):
    from arrival.jobs import worker

    out = tmp_path / "out"
    job = create_job(out / "jobs", "x.json", data=b"{}")

    def boom(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "load_upload", boom)
    with pytest.raises(KeyboardInterrupt):
        run_once(out, log=QUIET)
    st = json.loads((out / "jobs" / job["id"] / "status.json").read_text())
    assert st["status"] == "failed" and st["error"].startswith("interrupted") and st["finished_at"]


def test_work_requeues_jobs_left_running(tmp_path):
    from arrival.jobs.queue import write_status

    out = tmp_path / "out"
    job = create_job(out / "jobs", "broken.json", data=b"not json")
    write_status(out / "jobs" / job["id"], {**job, "status": "running", "stage": "sim"})
    status_json = out / "jobs" / job["id"] / "status.json"
    long_ago = time.time() - 3600  # no lock and no sign of life for an hour: a killed legacy worker
    os.utime(status_json, (long_ago, long_ago))
    assert work(out, once=True, log=QUIET) == 1  # reclaimed, then failed on the corrupt upload
    st = json.loads((out / "jobs" / job["id"] / "status.json").read_text())
    assert st["status"] == "failed"
