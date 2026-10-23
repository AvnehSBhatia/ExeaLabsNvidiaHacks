"""Worker limits, concurrent claiming, retry, retention and usage metering. Offline, no sim: jobs here are corrupt
uploads, fake child processes or a stand-in for the sim, so each test costs well under a second."""

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from arrival.eval import cost
from arrival.jobs import worker
from arrival.jobs.limits import LimitError, Limits, QueueFull
from arrival.jobs.queue import count_active, create_job, read_job, write_status
from arrival.jobs.retention import JobBusy, delete_job, gc
from arrival.jobs.worker import (CANCEL, LOCK, Heartbeat, UnsupportedLayout, UnusableDataset, build_usage, cancel_job,
                                 claim_next, process, reap_stale, release, requeue_failed, retryable, run_once, work)
from arrival.web.app import make_server

REPO = Path(__file__).resolve().parents[1]
QUIET = lambda *_: None  # noqa: E731
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="pid liveness is not probed on Windows")


def jobs_of(out):
    return out / "jobs"


def status_of(out, job):
    return json.loads((jobs_of(out) / job["id"] / "status.json").read_text(encoding="utf-8"))


def fake_running(out, job, pid=None, age=0.0):
    """Mark a job running as a worker would, with its claim.lock written by ``pid`` and last touched ``age`` s ago."""
    d = jobs_of(out) / job["id"]
    write_status(d, {**job, "status": "running", "stage": "sim", "worker": "ghost"})
    lock = d / LOCK
    lock.write_text(json.dumps({"worker": "ghost", "host": socket.gethostname(), "pid": pid or os.getpid()}))
    t = time.time() - age
    os.utime(lock, (t, t))
    return d


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# ---- limits ----------------------------------------------------------------------------------------------------

def test_limits_defaults_env_and_ceilings():
    lim = Limits.from_env({})
    assert lim.max_upload_bytes == 200 * 1024 * 1024 and lim.max_episodes == 100 and lim.max_people == 4
    assert lim.max_job_seconds == 1800 and lim.max_queue == 20 and lim.retention_hours is None and lim.job_retries == 2
    assert lim.job_retry_backoff == 5
    env = {"ARRIVAL_MAX_UPLOAD_MB": "1.5", "ARRIVAL_MAX_EPISODES": "3", "ARRIVAL_MAX_PEOPLE": "2",
           "ARRIVAL_MAX_JOB_SECONDS": "9.5", "ARRIVAL_MAX_QUEUE": "7", "ARRIVAL_RETENTION_HOURS": "24",
           "ARRIVAL_JOB_RETRIES": "0", "ARRIVAL_JOB_RETRY_BACKOFF": "8"}
    lim = Limits.from_env(env)
    assert (lim.max_upload_bytes, lim.max_episodes, lim.max_people) == (int(1.5 * 1024 * 1024), 3, 2)
    assert (lim.max_job_seconds, lim.max_queue, lim.retention_hours, lim.job_retries) == (9.5, 7, 24.0, 0)
    assert lim.job_retry_backoff == 8
    lim.check_request(3, 2)
    with pytest.raises(LimitError, match="episodes 4"):
        lim.check_request(4, 1)
    with pytest.raises(LimitError, match="people 3"):
        lim.check_request(1, 3)
    for bad in ("many", "-1", "nan", "inf"):
        with pytest.raises(LimitError, match="ARRIVAL_MAX_QUEUE"):
            Limits.from_env({"ARRIVAL_MAX_QUEUE": bad})
    for var in ("ARRIVAL_MAX_EPISODES", "ARRIVAL_MAX_PEOPLE", "ARRIVAL_MAX_QUEUE"):
        with pytest.raises(LimitError, match="whole number"):
            Limits.from_env({var: "0.5"})
    for var in ("ARRIVAL_MAX_UPLOAD_MB", "ARRIVAL_MAX_JOB_MB"):
        with pytest.raises(LimitError, match="too small"):
            Limits.from_env({var: "0.0000001"})
    assert lim.replace(max_episodes=None, max_people=1).max_people == 1  # None keeps the current value
    assert lim.replace(max_episodes=None).max_episodes == 3


def test_create_job_refuses_a_full_queue(tmp_path):
    jobs = tmp_path / "jobs"
    a = create_job(jobs, "a.json", data=b"{}", max_queue=2)
    create_job(jobs, "b.json", data=b"{}", max_queue=2)
    with pytest.raises(QueueFull):
        create_job(jobs, "c.json", data=b"{}", max_queue=2)
    assert count_active(jobs) == 2 and len(list(jobs.iterdir())) == 2  # nothing was written for the refused one
    write_status(jobs / a["id"], {**a, "status": "done"})  # a finished job frees a slot
    create_job(jobs, "c.json", data=b"{}", max_queue=2)
    create_job(jobs, "d.json", data=b"{}", max_queue=0)  # 0 = unlimited


def _post(port, body: bytes, ctype="multipart/form-data; boundary=BB", length=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest("POST", "/jobs")
    c.putheader("Content-Type", ctype)
    c.putheader("Content-Length", str(len(body) if length is None else length))
    c.endheaders(body)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data)


def _multipart(payload: bytes) -> bytes:
    return (b'--BB\r\nContent-Disposition: form-data; name="scan"; filename="k.json"\r\n'
            b"Content-Type: application/json\r\n\r\n" + payload + b"\r\n--BB--\r\n")


def test_web_enforces_queue_and_configured_upload_limits(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("ARRIVAL_MAX_QUEUE", "1")
    monkeypatch.setenv("ARRIVAL_MAX_UPLOAD_MB", "0.001")  # 1048 bytes
    srv = make_server(port=0, out_dir=out, scans_dir=REPO / "scans")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        code, body = _post(port, _multipart(b"x" * 5000))
        assert code == 413 and body["max_bytes"] == 1048
        code, body = _post(port, _multipart(b"{}"))
        assert code == 202 and body["status"] == "queued"
        code, body = _post(port, _multipart(b"{}"))
        assert code == 429 and body["max_queue"] == 1 and "full" in body["error"]
        assert count_active(jobs_of(out)) == 1  # the refused upload left no job behind
        monkeypatch.setenv("ARRIVAL_MAX_QUEUE", "lots")
        code, body = _post(port, _multipart(b"{}"))
        assert code == 500 and "ARRIVAL_MAX_QUEUE" in body["error"]
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_job_over_episode_or_people_ceiling_fails_and_worker_continues(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    st = run_once(out, limits=Limits(max_episodes=5), episodes=50, log=QUIET)
    assert st["status"] == "failed" and st["error"].startswith("LimitError: episodes 50 exceeds")
    assert st["stage"] is None  # rejected before any stage ran
    assert status_of(out, job)["status"] == "failed" and not (jobs_of(out) / job["id"] / LOCK).exists()
    create_job(jobs_of(out), "k.json", data=b"{}")
    st = run_once(out, limits=Limits(max_people=1), people=3, log=QUIET)
    assert st["status"] == "failed" and "people 3" in st["error"]


@pytest.mark.parametrize("res", ["0", "100"])
def test_work_cli_refuses_a_res_every_job_would_fail_on(tmp_path, res):
    from arrival.cli import main

    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"not json")
    assert main(["work", "--once", "--res", res, "--out", str(out)]) == 2
    assert status_of(out, job)["status"] == "queued"


def test_work_refuses_options_above_the_limits_before_touching_jobs(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    with pytest.raises(LimitError):
        work(out, once=True, limits=Limits(max_episodes=2), episodes=3, log=QUIET)
    assert status_of(out, job)["status"] == "queued"


def test_oversized_upload_fails_at_normalize_with_a_clear_error(tmp_path):
    out = tmp_path / "out"
    create_job(jobs_of(out), "big.json", data=b"x" * 100)
    st = run_once(out, limits=Limits(max_upload_bytes=50), log=QUIET)
    assert st["status"] == "failed" and st["stage"] == "normalize"
    assert "upload is 100 bytes, over the limit of 50" in st["error"]


# ---- wall-clock limit: the job runs in a child process that is killed -----------------------------------------

STAGE_THEN_HANG = ("import sys, json, time; from pathlib import Path; p = Path(sys.argv[1]) / 'status.json'; "
                   "s = json.loads(p.read_text()); s['stage'] = 'sim'; p.write_text(json.dumps(s)); time.sleep(120)")


def test_job_over_wall_clock_limit_is_killed_and_recorded_with_its_stage(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "slow.json", data=b"{}")
    t0 = time.monotonic()
    st = run_once(out, limits=Limits(max_job_seconds=1.0), isolate=True, log=QUIET,
                  child_cmd=[sys.executable, "-c", STAGE_THEN_HANG])
    assert time.monotonic() - t0 < 30  # killed at ~1 s, not after the child's 120 s sleep
    assert st["status"] == "failed" and st["stage"] == "sim"
    assert st["job_attempts"] == 1 and "retry_errors" not in st  # a timeout is not retried: the limit is per job
    assert st["error"] == "TimeoutError: job exceeded the 1 s wall-clock limit during stage 'sim'"
    disk = status_of(out, job)
    assert disk["status"] == "failed" and disk["finished_at"] and disk["usage"]["wall_seconds"] >= 1.0
    assert "TimeoutError" in (jobs_of(out) / job["id"] / "error.txt").read_text()
    assert not (jobs_of(out) / job["id"] / LOCK).exists()
    ledger = [json.loads(x) for x in (out / "usage" / "jobs.jsonl").read_text().splitlines()]
    assert [x["id"] for x in ledger] == [job["id"]] and ledger[0]["status"] == "failed"
    nxt = create_job(jobs_of(out), "next.json", data=b"not json")  # the worker goes on to the next job
    assert run_once(out, limits=Limits(max_job_seconds=0), log=QUIET)["id"] == nxt["id"]


def test_child_that_dies_is_retried_then_recorded_not_left_running(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    kw = dict(limits=Limits(max_job_seconds=30, job_retries=1, job_retry_backoff=0), isolate=True, log=QUIET,
              child_cmd=[sys.executable, "-c", "import sys; sys.exit(7)"])
    st = run_once(out, **kw)
    assert st["status"] == "queued" and st["error"] is None and "code 7" in st["retry_errors"][0]
    st = run_once(out, **kw)
    assert st["status"] == "failed" and "WorkerProcessError" in st["error"] and "code 7" in st["error"]
    assert st["job_attempts"] == 2 and status_of(out, job)["status"] == "failed" and run_once(out, **kw) is None


def test_real_child_process_runs_a_job_to_its_own_failure_once(tmp_path):
    """The default child (``python -m arrival.jobs.child``) records a bad upload exactly as the in-process path does."""
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "broken.json", data=b"not json")
    st = run_once(out, limits=Limits(max_job_seconds=120), isolate=True, log=QUIET)
    assert st["id"] == job["id"] and st["status"] == "failed" and "JSONDecodeError" in st["error"]
    assert st["stage"] == "normalize" and (jobs_of(out) / job["id"] / "error.txt").is_file()
    assert len((out / "usage" / "jobs.jsonl").read_text().splitlines()) == 1  # the parent did not log it again


# ---- concurrent claiming ---------------------------------------------------------------------------------------

def test_threads_racing_on_one_queue_claim_each_job_once(tmp_path):
    out = tmp_path / "out"
    ids = {create_job(jobs_of(out), f"k{i}.json", data=b"{}")["id"] for i in range(12)}
    got, lock = [], threading.Lock()
    start = threading.Barrier(4)

    def racer(name):
        start.wait()
        while (c := claim_next(jobs_of(out), name)) is not None:
            with lock:
                got.append((name, c[1]["id"]))

    ts = [threading.Thread(target=racer, args=(f"w{i}",)) for i in range(4)]
    [t.start() for t in ts]
    [t.join(timeout=30) for t in ts]
    assert sorted(i for _, i in got) == sorted(ids)  # every job exactly once, none twice
    assert {status_of(out, {"id": i})["worker"] for i in ids} <= {"w0", "w1", "w2", "w3"}
    assert claim_next(jobs_of(out)) is None


def test_processes_racing_on_one_queue_claim_each_job_once(tmp_path):
    """O_EXCL across real processes, not just threads."""
    out = tmp_path / "out"
    ids = {create_job(jobs_of(out), f"k{i}.json", data=b"{}")["id"] for i in range(10)}
    code = ("import sys; from arrival.jobs.worker import claim_next; from pathlib import Path\n"
            "while (c := claim_next(Path(sys.argv[1]), sys.argv[2])) is not None: print(c[1]['id'], flush=True)")
    procs = [subprocess.Popen([sys.executable, "-c", code, str(jobs_of(out)), f"p{i}"], stdout=subprocess.PIPE, text=True)
             for i in range(3)]
    claimed = [line for p in procs for line in p.communicate(timeout=60)[0].split()]
    assert sorted(claimed) == sorted(ids)


def test_a_locked_job_is_skipped_and_a_finished_one_is_not_reclaimed(tmp_path):
    out = tmp_path / "out"
    a = create_job(jobs_of(out), "a.json", data=b"{}")
    b = create_job(jobs_of(out), "b.json", data=b"{}")
    (jobs_of(out) / a["id"] / LOCK).write_text("{}")  # another worker is mid-claim on the oldest job
    d, st = claim_next(jobs_of(out))
    assert st["id"] == b["id"]
    release(d)
    write_status(d, {**st, "status": "done"})
    assert claim_next(jobs_of(out)) is None  # a's lock is still held, b is done


# ---- heartbeat, stale workers, crash then requeue ------------------------------------------------------------------

def test_heartbeat_keeps_the_lock_fresh_while_a_job_runs(tmp_path):
    out = tmp_path / "out"
    create_job(jobs_of(out), "k.json", data=b"{}")
    d, _ = claim_next(jobs_of(out))
    old = time.time() - 1000
    os.utime(d / LOCK, (old, old))
    with Heartbeat(d, 0.05):
        time.sleep(0.4)
    assert time.time() - (d / LOCK).stat().st_mtime < 5


@posix_only
def test_crash_then_requeue_when_the_workers_process_is_gone(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "broken.json", data=b"not json")
    d = fake_running(out, job, pid=dead_pid(), age=0)  # heartbeat is fresh, but the process is dead
    assert reap_stale(jobs_of(out), 120, log=QUIET) == 1
    st = status_of(out, job)
    assert st["status"] == "queued" and st["requeues"] == 1 and st["worker"] is None and not (d / LOCK).exists()
    assert run_once(out, log=QUIET)["status"] == "failed"  # and another worker can now take it


def test_a_live_worker_with_a_fresh_heartbeat_is_left_alone(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    d = fake_running(out, job, pid=os.getpid(), age=5)
    assert reap_stale(jobs_of(out), 120, log=QUIET) == 0
    assert status_of(out, job)["status"] == "running" and (d / LOCK).exists()
    assert claim_next(jobs_of(out)) is None


def test_stale_heartbeat_requeues_even_with_a_live_looking_pid(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    fake_running(out, job, pid=os.getpid(), age=600)  # hung or on another host: no heartbeat for 10 minutes
    assert reap_stale(jobs_of(out), 120, log=QUIET) == 1
    assert status_of(out, job)["status"] == "queued"


def test_job_that_keeps_killing_its_worker_fails_instead_of_looping(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    for n in range(worker.MAX_REQUEUES):
        fake_running(out, {**job, "requeues": n}, age=600)
        assert reap_stale(jobs_of(out), 120, log=QUIET) == 1
        assert status_of(out, job)["status"] == "queued"
    fake_running(out, {**job, "requeues": worker.MAX_REQUEUES}, age=600)
    reap_stale(jobs_of(out), 120, log=QUIET)
    st = status_of(out, job)
    assert st["status"] == "failed" and st["error"].startswith("WorkerLost") and st["finished_at"]


def test_two_reapers_requeue_a_dead_job_once(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    fake_running(out, job, age=600)
    counts, start = [], threading.Barrier(4)

    def reaper():
        start.wait()
        counts.append(reap_stale(jobs_of(out), 120, log=QUIET))

    ts = [threading.Thread(target=reaper) for _ in range(4)]
    [t.start() for t in ts]
    [t.join(timeout=30) for t in ts]
    assert sum(counts) == 1 and status_of(out, job)["requeues"] == 1


def test_stale_lock_on_a_queued_job_is_cleared_so_it_can_be_claimed(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    lock = jobs_of(out) / job["id"] / LOCK  # a worker died between creating the lock and marking the job running
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    old = time.time() - 600
    os.utime(lock, (old, old))
    assert claim_next(jobs_of(out)) is None
    assert reap_stale(jobs_of(out), 120, log=QUIET) == 1
    assert claim_next(jobs_of(out))[1]["id"] == job["id"]


def test_work_reaps_a_dead_workers_job_at_startup(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "broken.json", data=b"not json")
    fake_running(out, job, age=600)
    assert work(out, once=True, limits=Limits(max_job_seconds=0), log=QUIET) == 1
    assert status_of(out, job)["status"] == "failed" and status_of(out, job)["requeues"] == 1


# ---- cancel ----------------------------------------------------------------------------------------------------

def test_cancel_a_queued_job_before_any_worker_takes_it(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    assert cancel_job(jobs_of(out), job["id"]) == "cancelled"
    st = status_of(out, job)
    assert st["status"] == "cancelled" and st["finished_at"] and not (jobs_of(out) / job["id"] / LOCK).exists()
    assert claim_next(jobs_of(out)) is None
    assert cancel_job(jobs_of(out), job["id"]) == "cancelled"  # already finished: reports its state
    assert cancel_job(jobs_of(out), "job_nope") is None


def test_cancel_a_running_job_stops_it_at_the_next_stage(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    d, st = claim_next(jobs_of(out))
    assert cancel_job(jobs_of(out), job["id"]) == "cancelling" and (d / CANCEL).exists()
    final = process(d, st, out, log=QUIET)
    assert final["status"] == "cancelled" and final["error"] is None and final["finished_at"]
    assert not (d / CANCEL).exists() and not (d / "error.txt").exists()


def test_cancel_kills_a_job_running_in_a_child_process(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    threading.Timer(0.5, lambda: cancel_job(jobs_of(out), job["id"])).start()
    t0 = time.monotonic()
    st = run_once(out, limits=Limits(max_job_seconds=60), isolate=True, log=QUIET,
                  child_cmd=[sys.executable, "-c", STAGE_THEN_HANG])
    assert time.monotonic() - t0 < 30
    assert st["status"] == "cancelled" and st["error"] is None
    assert not (jobs_of(out) / job["id"] / CANCEL).exists()


# ---- automatic retry -------------------------------------------------------------------------------------------

HERO_SCAN = (REPO / "scans" / "hero_kitchen" / "scan.json").read_bytes()


def fake_sim(monkeypatch, plan, invalid=()):
    """Stand in for the sim and the dataset validator, so a job runs every stage without MuJoCo. ``plan(n)`` is what the
    n-th sim run of a room does: an exception to raise, or report fields over a full success ({} = every episode
    succeeds). The validator returns ``invalid``. Returns {room_id: sim runs so far}."""
    runs, lock = {}, threading.Lock()

    def collect(twin, families, episodes, out_dir, log=print, **_):
        with lock:
            n = runs[twin["room_id"]] = runs.get(twin["room_id"], 0) + 1
        outcome = plan(n)
        if isinstance(outcome, BaseException):
            raise outcome
        ds = Path(out_dir) / "datasets" / f"{twin['room_id']}_sim"
        (ds / "meta").mkdir(parents=True, exist_ok=True)
        (ds / "meta" / "info.json").write_text(json.dumps({"total_frames": 10}))
        report = {"attempts": episodes, "successes": episodes, "skipped": 0, "skipped_unreachable": 0,
                  "skipped_error": 0, "skip_reasons": [], **outcome}
        return {"dataset": ds, "report_data": report}

    monkeypatch.setitem(sys.modules, "arrival.sim.collect", types.SimpleNamespace(collect=collect))
    monkeypatch.setattr("arrival.export.validate.validate", lambda ds: list(invalid))
    return runs


def test_retryable_tells_the_uploads_faults_from_the_machines():
    from arrival.scan.bundle import ScanBundleError

    for e in (OSError(28, "No space left on device"), MemoryError(), RuntimeError("renderer lost its context"),
              ImportError("no mujoco on this worker")):
        assert retryable(e), e
    for e in (ValueError("x"), json.JSONDecodeError("x", "", 0), ScanBundleError("x"), LimitError("x"),
              UnsupportedLayout("x"), UnusableDataset("x"), KeyboardInterrupt(), SystemExit(1)):
        assert not retryable(e), e


def test_a_transient_failure_is_retried_and_the_job_then_succeeds(tmp_path, monkeypatch):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=HERO_SCAN)
    fake_sim(monkeypatch, lambda n: OSError(28, "No space left on device") if n == 1 else {})
    written = []  # every status this worker writes, as a client polling the job could see it
    write = worker.write_status
    monkeypatch.setattr(worker, "write_status", lambda d, s: written.append(s["status"]) or write(d, s))
    first = run_once(out, limits=Limits(job_retry_backoff=0), log=QUIET)
    assert first["status"] == "queued" and first["job_attempts"] == 1 and first["error"] is None
    assert first["retry_errors"] == ["OSError: [Errno 28] No space left on device"]
    assert first["stage"] is None and first["finished_at"] is None and first["worker"] is None
    assert "failed" not in written  # the capture app stops polling at "failed"; a job that will run again never shows it
    assert status_of(out, job) == first and not (jobs_of(out) / job["id"] / LOCK).exists()
    second = run_once(out, limits=Limits(job_retry_backoff=0), log=QUIET)
    assert second["status"] == "done" and second["job_attempts"] == 2 and second["error"] is None
    assert second["retry_errors"] == first["retry_errors"] and Path(second["zip"]).is_file()
    assert run_once(out, log=QUIET) is None
    ledger = [json.loads(x) for x in (out / "usage" / "jobs.jsonl").read_text().splitlines()]
    assert [(x["job_attempts"], x["status"]) for x in ledger] == [(1, "failed"), (2, "done")]  # one line per attempt


CRASH_ONCE = ("import sys, json; from pathlib import Path; p = Path(sys.argv[1]) / 'status.json'; "
              "s = json.loads(p.read_text())\nif s['job_attempts'] == 1: sys.exit(9)\n"
              "s.update(status='done', stage='zip'); p.write_text(json.dumps(s))")


def test_a_child_that_dies_once_is_retried_and_the_next_attempt_finishes(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    kw = dict(limits=Limits(max_job_seconds=30, job_retry_backoff=0), isolate=True, log=QUIET,
              child_cmd=[sys.executable, "-c", CRASH_ONCE])
    first = run_once(out, **kw)  # killed, as by the OOM killer: the parent records it and queues the job again
    assert first["status"] == "queued" and first["job_attempts"] == 1
    assert first["retry_errors"][0].startswith("WorkerProcessError") and "code 9" in first["retry_errors"][0]
    second = run_once(out, **kw)
    assert second["status"] == "done" and second["job_attempts"] == 2 and status_of(out, job)["status"] == "done"


@pytest.mark.parametrize("retries", [0, 1, 2])
def test_retries_run_out_and_the_job_fails_with_its_last_error_and_attempts(tmp_path, monkeypatch, retries):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=HERO_SCAN)
    fake_sim(monkeypatch, lambda n: RuntimeError(f"renderer lost its context ({n})"))
    limits = Limits(job_retries=retries, job_retry_backoff=0)
    states = [run_once(out, limits=limits, log=QUIET)["status"] for _ in range(retries + 1)]
    assert states == ["queued"] * retries + ["failed"] and run_once(out, limits=limits, log=QUIET) is None
    st = status_of(out, job)
    assert st["job_attempts"] == retries + 1 and st["stage"] == "sim" and st["finished_at"]
    assert st["error"] == f"RuntimeError: renderer lost its context ({retries + 1})"
    assert st.get("retry_errors", []) == [f"RuntimeError: renderer lost its context ({n})" for n in range(1, retries + 1)]
    assert f"({retries + 1})" in (jobs_of(out) / job["id"] / "error.txt").read_text()  # the last attempt's traceback


@pytest.mark.parametrize("case, error", [
    ("not_json", "JSONDecodeError: "), ("invalid_scan", "ScanBundleError: units must be 'meters'"),
    ("over_limit", "LimitError: upload is"), ("unsupported_layout", "unsupported layout: "),
    ("no_successes", "UnusableDataset: no successful episodes"), ("invalid_dataset", "UnusableDataset: dataset invalid")])
def test_bad_input_and_failures_a_rerun_would_repeat_are_not_retried(tmp_path, monkeypatch, case, error):
    out = tmp_path / "out"
    data = {"not_json": b"not json", "invalid_scan": b'{"schema_version": "1", "units": "feet"}'}.get(case, HERO_SCAN)
    job = create_job(jobs_of(out), "k.json", data=data)
    report = {"unsupported_layout": {"attempts": 0, "successes": 0}, "no_successes": {"successes": 0}}.get(case, {})
    runs = fake_sim(monkeypatch, lambda n: report, invalid=["episode 0 has no frames"] if case == "invalid_dataset" else ())
    limits = Limits(max_upload_bytes=100) if case == "over_limit" else Limits()
    st = run_once(out, limits=limits, log=QUIET)
    assert st["id"] == job["id"] and st["status"] == "failed" and st["error"].startswith(error)
    assert st["job_attempts"] == 1 and "retry_errors" not in st and run_once(out, limits=limits, log=QUIET) is None
    assert sum(runs.values()) == (0 if case in ("not_json", "invalid_scan", "over_limit") else 1)


def test_a_job_queued_for_a_retry_stays_locked_until_its_worker_lets_go(tmp_path, monkeypatch):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=HERO_SCAN)
    fake_sim(monkeypatch, lambda n: OSError("flaky disk") if n == 1 else {})
    d, st = claim_next(jobs_of(out), "w1")
    limits = Limits(job_retry_backoff=0)
    assert process(d, st, out, limits=limits, log=QUIET)["status"] == "queued" and (d / LOCK).exists()  # w1 has not released it yet
    assert claim_next(jobs_of(out), "w2") is None and reap_stale(jobs_of(out), 120, log=QUIET) == 0
    release(d)
    d2, st2 = claim_next(jobs_of(out), "w2")
    assert (st2["id"], st2["worker"], st2["job_attempts"]) == (job["id"], "w2", 2)
    assert process(d2, st2, out, limits=limits, log=QUIET)["status"] == "done"


def test_racing_workers_run_each_attempt_of_a_retried_job_once(tmp_path, monkeypatch):
    out = tmp_path / "out"
    ids = {create_job(jobs_of(out), f"k{i}.json", data=HERO_SCAN)["id"] for i in range(4)}
    runs = fake_sim(monkeypatch, lambda n: OSError("flaky disk") if n == 1 else {})
    start = threading.Barrier(3)

    def racer(name):
        start.wait()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and any(status_of(out, {"id": i})["status"] in ("queued", "running")
                                                  for i in ids):
            if run_once(out, worker=name, limits=Limits(job_retry_backoff=0), log=QUIET) is None:
                time.sleep(0.01)

    ts = [threading.Thread(target=racer, args=(f"w{i}",)) for i in range(3)]
    [t.start() for t in ts]
    [t.join(timeout=90) for t in ts]
    for s in (status_of(out, {"id": i}) for i in ids):
        assert (s["status"], s["job_attempts"], s["retry_errors"]) == ("done", 2, ["OSError: flaky disk"]), s
    assert sorted(runs.values()) == [2] * len(ids)  # one failed and one good sim run per job: no attempt ran twice


def test_a_cancel_request_stops_a_retry(tmp_path, monkeypatch):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=HERO_SCAN)

    def cancel_then_fail(n):
        cancel_job(jobs_of(out), job["id"])  # the uploader cancels while the sim runs, then the sim fails
        return OSError("flaky disk")

    fake_sim(monkeypatch, cancel_then_fail)
    st = run_once(out, log=QUIET)
    assert st["status"] == "failed" and st["error"] == "OSError: flaky disk" and "retry_errors" not in st
    assert run_once(out, log=QUIET) is None


def test_work_cli_takes_retries_from_the_env_then_the_flag(tmp_path, monkeypatch):
    from arrival.cli import main

    seen = {}
    monkeypatch.setattr("arrival.jobs.worker.work", lambda out, **kw: seen.update(kw) or 0)
    monkeypatch.setenv("ARRIVAL_JOB_RETRIES", "5")
    assert main(["work", "--once", "--out", str(tmp_path)]) == 0 and seen["limits"].job_retries == 5
    assert main(["work", "--once", "--retries", "0", "--out", str(tmp_path)]) == 0 and seen["limits"].job_retries == 0
    assert main(["work", "--once", "--retries", "-1", "--out", str(tmp_path)]) == 2


def test_requeue_sets_not_before_with_exponential_backoff(tmp_path):
    d = tmp_path / "job_x"
    d.mkdir()
    st = {"id": "job_x", "status": "failed", "error": "OSError: disk"}
    limits = Limits(job_retries=5, job_retry_backoff=5)
    t0 = time.time()
    assert requeue_failed(d, st, limits, log=QUIET)
    assert st["status"] == "queued" and t0 + 4 <= st["not_before"] <= t0 + 6
    st["error"] = "OSError: disk again"
    t1 = time.time()
    assert requeue_failed(d, st, limits, log=QUIET)
    assert t1 + 9 <= st["not_before"] <= t1 + 11
    st["error"] = "OSError: disk still"
    t2 = time.time()
    assert requeue_failed(d, st, Limits(job_retries=5, job_retry_backoff=200), log=QUIET)
    assert t2 + 299 <= st["not_before"] <= t2 + 302  # 200 * 4 would be 800, capped at 300


def test_claim_skips_a_queued_job_until_not_before(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    d = jobs_of(out) / job["id"]
    write_status(d, {**status_of(out, job), "not_before": int(time.time()) + 60})
    assert claim_next(jobs_of(out)) is None
    write_status(d, {**status_of(out, job), "not_before": int(time.time()) - 1})
    assert claim_next(jobs_of(out))[1]["id"] == job["id"]


def test_claim_skips_a_waiting_retry_and_takes_the_next_ready_job(tmp_path):
    out = tmp_path / "out"
    waiting = create_job(jobs_of(out), "a.json", data=b"{}")
    ready = create_job(jobs_of(out), "b.json", data=b"{}")
    write_status(jobs_of(out) / waiting["id"],
                 {**status_of(out, waiting), "created_unix": 1, "not_before": int(time.time()) + 60})
    write_status(jobs_of(out) / ready["id"], {**status_of(out, ready), "created_unix": 2})
    d, st = claim_next(jobs_of(out))
    assert st["id"] == ready["id"] and d.name == ready["id"]


def test_a_requeued_job_is_not_claimed_before_not_before(tmp_path, monkeypatch):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=HERO_SCAN)
    fake_sim(monkeypatch, lambda n: OSError("flaky disk") if n == 1 else {})
    limits = Limits(job_retry_backoff=60)
    first = run_once(out, limits=limits, log=QUIET)
    assert first["status"] == "queued" and first["not_before"] > time.time()
    assert run_once(out, limits=limits, log=QUIET) is None
    write_status(jobs_of(out) / job["id"], {**status_of(out, job), "not_before": int(time.time()) - 1})
    second = run_once(out, limits=limits, log=QUIET)
    assert second["status"] == "done" and second["job_attempts"] == 2


# ---- retention -------------------------------------------------------------------------------------------------

def _finished_job(out, name, hours_old, status="done", with_outputs=True):
    job = create_job(jobs_of(out), f"{name}.json", data=b"{}")
    room = worker.room_id_for(f"{name}.json", job["id"])
    d = jobs_of(out) / job["id"]
    st = {**job, "status": status, "room_id": room, "finished_at": int(time.time() - hours_old * 3600)}
    write_status(d, st)
    if with_outputs:
        for p in (out / "twins" / room, out / "datasets" / f"{room}_sim"):
            p.mkdir(parents=True)
            (p / "f.bin").write_bytes(b"x" * 100)
        (d / "dataset.zip").write_bytes(b"z" * 50)
        (out / "reports").mkdir(exist_ok=True)
        (out / "reports" / f"{room}_sim.json").write_text("{}")
        (out / "progress").mkdir(exist_ok=True)
        (out / "progress" / f"{room}_sim.jsonl").write_text("{}\n")
    return job, room


def test_gc_deletes_old_finished_jobs_and_everything_they_produced(tmp_path):
    out = tmp_path / "out"
    old, old_room = _finished_job(out, "old", 48)
    failed, _ = _finished_job(out, "failed", 30, status="failed", with_outputs=False)
    young, young_room = _finished_job(out, "young", 1)
    running = create_job(jobs_of(out), "run.json", data=b"{}")
    write_status(jobs_of(out) / running["id"], {**running, "status": "running", "created_unix": 1})
    queued = create_job(jobs_of(out), "q.json", data=b"{}")
    hand_made = out / "datasets" / "hero_kitchen_sim"  # not a job's: must survive even though a job names it
    hand_made.mkdir(parents=True)
    odd, _ = _finished_job(out, "odd", 50, with_outputs=False)
    write_status(jobs_of(out) / odd["id"], {**status_of(out, odd), "room_id": "hero_kitchen"})
    gone = sorted([old["id"], failed["id"], odd["id"]])

    dry = gc(out, 24, dry_run=True)
    assert sorted(dry["removed"]) == gone and (jobs_of(out) / old["id"]).exists()
    res = gc(out, 24, log=QUIET)
    assert sorted(res["removed"]) == gone and res["bytes"] > 0
    assert not (jobs_of(out) / old["id"]).exists() and not (jobs_of(out) / failed["id"]).exists()
    assert not (out / "twins" / old_room).exists() and not (out / "datasets" / f"{old_room}_sim").exists()
    assert not (out / "reports" / f"{old_room}_sim.json").exists()
    assert not (out / "progress" / f"{old_room}_sim.jsonl").exists()
    assert (out / "progress" / f"{young_room}_sim.jsonl").exists()
    assert hand_made.is_dir()  # odd's room_id does not end with its job id, so its room is not its to delete
    for p in (jobs_of(out) / young["id"], jobs_of(out) / running["id"], jobs_of(out) / queued["id"],
              out / "twins" / young_room, out / "datasets" / f"{young_room}_sim", out / "reports" / f"{young_room}_sim.json"):
        assert p.exists(), p
    assert gc(out, 24)["removed"] == []
    with pytest.raises(ValueError):
        gc(out, 0)


def test_delete_job_removes_twin_dataset_report_and_zip_but_not_a_busy_job(tmp_path):
    out = tmp_path / "out"
    job, room = _finished_job(out, "mine", 0.1)
    res = delete_job(out, job["id"])
    assert res["bytes"] >= 250 and not (jobs_of(out) / job["id"]).exists()
    for p in (out / "twins" / room, out / "datasets" / f"{room}_sim", out / "reports" / f"{room}_sim.json",
              out / "progress" / f"{room}_sim.jsonl"):
        assert not p.exists(), p
    assert delete_job(out, job["id"]) is None and delete_job(out, "../out") is None
    q = create_job(jobs_of(out), "q.json", data=b"{}")
    with pytest.raises(JobBusy):
        delete_job(out, q["id"])
    assert (jobs_of(out) / q["id"]).exists()


def test_retention_does_not_erase_metering(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"not json")
    st = run_once(out, log=QUIET)
    assert st["status"] == "failed"
    write_status(jobs_of(out) / job["id"], {**status_of(out, job), "finished_at": 1})
    assert gc(out, 1, log=QUIET)["removed"] == [job["id"]]
    assert cost.job_usage(out)[job["id"]]["wall_seconds"] is not None  # the ledger line is still there


def test_gc_cli_dry_run_and_delete(tmp_path, capsys):
    from arrival.cli import main

    out = tmp_path / "out"
    job, _ = _finished_job(out, "old", 100)
    assert main(["gc", "--max-age-hours", "24", "--dry-run", "--out", str(out)]) == 0
    assert "would remove 1 job(s)" in capsys.readouterr().out and (jobs_of(out) / job["id"]).exists()
    assert main(["gc", "--max-age-hours", "24", "--out", str(out)]) == 0
    assert "removed 1 job(s)" in capsys.readouterr().out and not (jobs_of(out) / job["id"]).exists()
    assert main(["gc", "--max-age-hours", "0", "--out", str(out)]) == 2


def test_work_sweeps_old_jobs_when_retention_is_set(tmp_path):
    out = tmp_path / "out"
    old, _ = _finished_job(out, "old", 100)
    assert work(out, once=True, limits=Limits(max_job_seconds=0, retention_hours=24), log=QUIET) == 0
    assert not (jobs_of(out) / old["id"]).exists()
    keep, _ = _finished_job(out, "keep", 100)
    assert work(out, once=True, limits=Limits(max_job_seconds=0), log=QUIET) == 0  # no retention configured
    assert (jobs_of(out) / keep["id"]).exists()


# ---- metering --------------------------------------------------------------------------------------------------

def test_failed_job_is_metered_with_unmeasured_fields_null(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "broken.json", data=b"not json")
    st = run_once(out, log=QUIET)
    u = status_of(out, job)["usage"]
    assert u["wall_seconds"] >= 0 and "normalize" in u["stage_seconds"] and u["measured_on"] == "local_cpu"
    assert u["upload_bytes"] == 8 and u["episodes_attempted"] is None and u["frames"] is None
    assert u["zip_bytes"] is None and u["dataset_bytes"] is None  # never 0 for what was not produced
    line = json.loads((out / "usage" / "jobs.jsonl").read_text())
    assert line["id"] == job["id"] and line["status"] == "failed" and line["wall_seconds"] == u["wall_seconds"]
    assert st["usage"] == u


def test_build_usage_reads_frames_and_sizes_from_the_written_dataset(tmp_path):
    ds = tmp_path / "room_sim"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps({"total_frames": 321}))
    (ds / "data.bin").write_bytes(b"x" * 1000)
    zp = tmp_path / "dataset.zip"
    zp.write_bytes(b"z" * 40)
    u = build_usage({"attempts": 4, "successes": 3, "bytes": 77, "dataset_dir": str(ds), "zip": str(zp)}, 12.3456,
                    {"sim": 9.0})
    assert u["frames"] == 321 and u["zip_bytes"] == 40 and u["upload_bytes"] == 77
    assert u["dataset_bytes"] == 1000 + len(json.dumps({"total_frames": 321}))
    assert (u["episodes_attempted"], u["episodes_succeeded"], u["wall_seconds"]) == (4, 3, 12.346)
    assert build_usage({"dataset_dir": str(tmp_path / "gone")}, 1)["frames"] is None


def test_cost_report_reads_job_metering_from_status_and_ledger(tmp_path):
    out = tmp_path / "out"
    live = {"id": "job_live", "room_id": "k_job_live", "status": "done", "started_at": 100, "finished_at": 130,
            "attempts": 4, "successes": 3,
            "usage": {"wall_seconds": 25.5, "episodes_attempted": 4, "episodes_succeeded": 3, "frames": 900,
                      "upload_bytes": 10, "zip_bytes": 20, "dataset_bytes": None}}
    (out / "jobs" / "job_live").mkdir(parents=True)
    (out / "jobs" / "job_live" / "status.json").write_text(json.dumps(live))
    (out / "usage").mkdir()
    ledger = [{"id": "job_gone", "room_id": "g_job_gone", "status": "done", "attempts": 2, "successes": 2,
               "wall_seconds": 10.0, "frames": 100, "upload_bytes": 5, "zip_bytes": None, "dataset_bytes": 300},
              {"id": "job_live", "wall_seconds": 999.0, "frames": 1, "attempts": 1, "successes": 1}]
    (out / "usage" / "jobs.jsonl").write_text("\n".join(json.dumps(x) for x in ledger) + "\n{cut off by a cra")
    rep = cost.build(out)
    rows = {r["room"]: r for r in rep["rows"] if r["source"] == "job"}
    assert rows["k_job_live"]["seconds"] == 25.5 and rows["k_job_live"]["frames"] == 900  # status.json beats the ledger
    assert rows["g_job_gone"]["seconds"] == 10.0 and rows["g_job_gone"]["frames"] == 100  # deleted job keeps its row
    m = rep["metering"]
    assert m["jobs"] == 2 and m["wall_seconds"] == 35.5 and m["frames"] == 1000 and m["episodes_succeeded"] == 3
    assert m["upload_bytes"] == 15 and m["zip_bytes"] == 20 and m["dataset_bytes"] == 300  # only measured values summed
    assert rep["dollars"]["serverless_jobs"]["usd"] is None  # metering is unpriced
    assert "Worker metering over 2 job(s)" in cost.markdown(rep)
    empty = cost.build(tmp_path / "nothing")
    assert empty["metering"]["jobs"] == 0 and empty["metering"]["wall_seconds"] is None
    assert "Worker metering" not in cost.markdown(empty)


# ---- per-episode progress and the optional memory cap ----------------------------------------------------------------

def test_progress_log_counts_finished_sim_tasks(tmp_path):
    from arrival.jobs.worker import progress_log

    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    d, st = claim_next(jobs_of(out))
    seen = []
    hook = progress_log(d, st, 3, seen.append)
    assert status_of(out, job)["progress"] == {"done": 0, "total": 3}
    for line in ("[sim] t0: planned by rule -> {}", "[sim] t0: OK door ik_err=0.1", "[sim] t1: MISS x",
                 "[sim] t2: SKIP at env: why", "[sim] t3: DROP human_contact y"):
        hook(line)
    assert len(seen) == 5 and status_of(out, job)["progress"] == {"done": 3, "total": 3}  # capped; planner line ignored


def test_claim_clears_old_progress(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "k.json", data=b"{}")
    path = jobs_of(out) / job["id"] / "status.json"
    st = json.loads(path.read_text())
    st["progress"] = {"done": 2, "total": 2}
    path.write_text(json.dumps(st))
    claim_next(jobs_of(out))
    assert status_of(out, job)["progress"] is None


def test_memory_limit_is_off_by_default_and_read_in_megabytes():
    assert Limits().max_job_bytes == 0
    assert Limits.from_env({"ARRIVAL_MAX_JOB_MB": "512"}).max_job_bytes == 512 * 1024 * 1024


# allocates `MB` megabytes in a grandchild (the way the job's video encoder is a process of its own) and waits
GRANDCHILD_ALLOCATES = ("import subprocess, sys, time; "
                        "subprocess.Popen([sys.executable, '-c', 'import time; b = b\"x\" * ({MB} << 20); time.sleep(60)']); "
                        "time.sleep(60)")


@pytest.mark.skipif(sys.platform == "win32", reason="tree_rss reads /proc or ps; not measured on Windows")
def test_tree_rss_counts_the_processes_a_child_starts():
    p = subprocess.Popen([sys.executable, "-c", GRANDCHILD_ALLOCATES.format(MB=200)])
    try:
        rss, t0 = 0, time.monotonic()
        while rss < 200 << 20 and time.monotonic() - t0 < 20:
            time.sleep(0.2)
            rss = worker.tree_rss(p.pid) or 0
        assert rss >= 200 << 20  # the grandchild's 200 MB count toward the child
    finally:
        p.kill()
        p.wait()
    assert worker.tree_rss(os.getpid()) > 0 and worker.tree_rss(2 ** 22 + 12345) is None


STAGE_THEN_ALLOCATE = ("import sys, json, time; from pathlib import Path; p = Path(sys.argv[1]) / 'status.json'; "
                       "s = json.loads(p.read_text()); s['stage'] = 'sim'; p.write_text(json.dumps(s)); "
                       "b = b'x' * (300 << 20); time.sleep(120)")


@pytest.mark.skipif(sys.platform == "win32", reason="tree_rss reads /proc or ps; not measured on Windows")
def test_job_over_memory_limit_is_killed_and_recorded_with_its_stage(tmp_path):
    out = tmp_path / "out"
    job = create_job(jobs_of(out), "big.json", data=b"{}")
    t0 = time.monotonic()
    st = run_once(out, limits=Limits(max_job_seconds=60, max_job_bytes=100 << 20, job_retries=2), isolate=True,
                  log=QUIET, child_cmd=[sys.executable, "-c", STAGE_THEN_ALLOCATE])
    assert time.monotonic() - t0 < 30  # killed once it passes 100 MB, not after the child's 120 s sleep
    assert st["status"] == "failed" and st["stage"] == "sim" and st["job_attempts"] == 1  # not retried
    assert st["error"].startswith("MemoryLimitError: the job used ") and st["error"].endswith(
        " MB resident, over the 100 MB limit (ARRIVAL_MAX_JOB_MB), during stage 'sim'")
    assert float(st["error"].split("used ")[1].split(" MB")[0]) > 100
    assert "MemoryLimitError" in (jobs_of(out) / job["id"] / "error.txt").read_text()
    assert status_of(out, job)["status"] == "failed" and not (jobs_of(out) / job["id"] / LOCK).exists()


def test_job_under_memory_limit_is_not_killed(tmp_path):
    out = tmp_path / "out"
    create_job(jobs_of(out), "small.json", data=b"{}")
    small = "import time, sys; b = b'x' * (20 << 20); time.sleep(1.5); sys.exit(7)"
    st = run_once(out, limits=Limits(max_job_seconds=60, max_job_bytes=1 << 30, job_retries=0), isolate=True,
                  log=QUIET, child_cmd=[sys.executable, "-c", small])
    assert st["status"] == "failed" and st["error"].startswith("WorkerProcessError") and "code 7" in st["error"]
