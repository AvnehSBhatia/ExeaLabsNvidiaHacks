"""Drain out/jobs: upload -> twin -> MuJoCo sim dataset -> dataset.zip.

``python -m arrival work`` runs this loop. The web server never imports it. Every
stage boundary rewrites ``status.json``; a failure is recorded there and the loop
moves on, so one bad upload never stops the queue.

Several workers may share one ``out/jobs``. A worker claims a job by creating ``claim.lock`` with
``O_CREAT|O_EXCL`` (exactly one creator wins, across processes), keeps the lock's mtime fresh while the job runs
(the heartbeat), and removes it when the job ends. A ``running`` job whose lock stopped being touched, or whose
worker process is gone on the same host, is requeued by ``reap_stale``. The claim is local-filesystem only: on a
network share without working ``O_EXCL`` it is not safe.

Limits (``arrival.jobs.limits``): a wall-clock timeout per job (the job runs in a child process that is killed),
episodes and people ceilings, and the upload size. Each finished attempt gets a ``usage`` block in ``status.json``
and one line in ``out/usage/jobs.jsonl``; ``arrival.eval.cost`` reads both.

A failed attempt goes back to the queue up to ``limits.job_retries`` times (``requeue_failed``), unless the failure
would only repeat (``retryable``): a bad upload or a request over a limit, a layout the sim cannot use, a dataset with
nothing usable in it, or the wall-clock timeout. ``job_attempts`` in ``status.json`` counts the starts and
``retry_errors`` keeps the error of each attempt that was retried. The requeue writes ``not_before`` (``limits.job_retry_backoff``
seconds, doubling per attempt, capped at 300) and is written while the worker still holds ``claim.lock``, so no other
worker can take the job before this one lets go of it and the backoff has elapsed.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

from arrival.jobs.limits import LimitError, Limits
from arrival.jobs.queue import write_status

DEFAULT_FAMILIES = ("counter_to_sink", "open_drawer", "move_on_table")  # move_on_table runs only in rooms with a table
STAGES = ("normalize", "twin", "realism", "objects", "sim", "validate", "zip")
ROOM_RE = re.compile(r"[^a-z0-9_]+")
LOCK = "claim.lock"  # present while a worker owns the job; its mtime is the heartbeat
CANCEL = "cancel.requested"  # marker written by cancel_job for a running job
REAP_LOCK = "reap.lock"  # arbiter so two workers never requeue the same job
MAX_REQUEUES = 2  # a job whose worker died this many times fails instead of looping forever
USAGE_LEDGER = ("usage", "jobs.jsonl")  # under out/: one JSON line per finished job, survives gc
MEM_TICK = 0.5  # seconds between resident-memory checks of a job's child process tree (limits.max_job_bytes)
MB = 1024 * 1024


SIM_TASK_RE = re.compile(r"^\[sim\] \S+: (?:OK|MISS|DROP|SKIP)\b")  # the line collect logs when one task finishes


class JobCancelled(Exception):
    """Raised at a stage boundary when cancel_job asked for this running job to stop."""


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class UnsupportedLayout(RuntimeError):
    """Not one task of the requested families can run in the uploaded room (every candidate failed the pre-flight
    in `arrival.sim.preflight`). The job fails with a typed status and the reasons, not a traceback."""

    def __init__(self, message: str, reasons: list[str] | None = None):
        super().__init__(message)
        self.reasons = reasons or []


class UnusableDataset(RuntimeError):
    """The sim ran but left nothing to ship: no successful episode, or a dataset the validator rejects. The sim is
    seeded, so running the same upload again gives the same result."""


NOT_RETRIED = (ValueError, UnsupportedLayout, UnusableDataset)  # the same upload would fail the same way again


def retryable(e: BaseException) -> bool:
    """True when another attempt may not hit ``e`` again: an OS or I/O error, running out of memory, an unexpected
    error in the twin compiler, renderer or sim. False for the upload's own faults, which are ``ValueError``s
    (unreadable JSON, ``ScanBundleError`` from the scan validators, ``LimitError`` for a request over a limit), for
    ``UnsupportedLayout`` and ``UnusableDataset``, and for Ctrl-C or SystemExit."""
    return isinstance(e, Exception) and not isinstance(e, NOT_RETRIED)


def room_id_for(filename: str, job_id: str) -> str:
    """``My Kitchen (2).json`` + ``job_1a2b3c4d`` -> ``my_kitchen_2_job_1a2b3c4d``."""
    stem = ROOM_RE.sub("_", Path(filename or "").stem.lower()).strip("_")
    return f"{stem}_{job_id}" if stem else job_id


def load_upload(path: Path, room_id: str) -> dict:
    """Read a RoomPlan CapturedRoom export or an authored scan.json into a ScanBundle named ``room_id``."""
    from arrival.scan import bundle as scan_bundle
    from arrival.scan import roomplan

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("upload is not a JSON object")
    if "schema_version" in data:
        bundle = scan_bundle.validate(data)
        bundle["source_room_id"] = bundle["room_id"]
        bundle["room_id"] = room_id
        return bundle
    def posed(d: dict) -> bool:
        return any(isinstance(x, dict) and "transform" in x for k in ("walls", "objects")
                   for x in (d[k] if isinstance(d.get(k), list) else []))

    rooms = data.get("rooms")  # a CapturedStructure export nests the room dicts
    if posed(data) or (isinstance(rooms, list) and any(isinstance(r, dict) and posed(r) for r in rooms)):
        return roomplan.to_bundle(data, room_id)
    raise ValueError("upload is neither a RoomPlan export (walls/objects with transform) nor a scan.json (schema_version)")


def _write_status(job_dir: Path, status: dict) -> dict:
    return write_status(job_dir, status)


def _read_status(job_dir: Path) -> dict | None:
    try:
        st = json.loads((Path(job_dir) / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return st if isinstance(st, dict) else None


class Meter:
    """Wall seconds per stage for one job, written into ``status["usage"]`` as the job goes."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.stage: str | None = None
        self.t_stage = self.t0
        self.seconds: dict[str, float] = {}

    def enter(self, stage: str | None) -> None:
        now = time.monotonic()
        if self.stage is not None:
            self.seconds[self.stage] = round(self.seconds.get(self.stage, 0.0) + now - self.t_stage, 3)
        self.stage, self.t_stage = stage, now

    def wall(self) -> float:
        return round(time.monotonic() - self.t0, 3)


def _dir_bytes(path) -> int | None:
    try:
        return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    except OSError:
        return None


def build_usage(status: dict, wall_seconds: float, stage_seconds: dict | None = None) -> dict:
    """Units one job consumed on this machine. Anything not measured is None, never zero: ``frames`` and the byte
    counts exist only when the dataset or zip was written, and nothing here is priced (a local CPU run is $0 and
    ``arrival.eval.cost`` attaches no rate to it)."""
    def size(key):
        try:
            return Path(status[key]).stat().st_size if status.get(key) else None
        except OSError:
            return None

    frames = None
    ds = status.get("dataset_dir")
    if ds:
        try:
            frames = int(json.loads((Path(ds) / "meta" / "info.json").read_text(encoding="utf-8"))["total_frames"])
        except (OSError, ValueError, KeyError, TypeError):
            frames = None
    return {"wall_seconds": round(wall_seconds, 3), "stage_seconds": dict(stage_seconds or {}),
            "episodes_attempted": status.get("attempts"), "episodes_succeeded": status.get("successes"),
            "frames": frames, "upload_bytes": status.get("bytes"), "zip_bytes": size("zip"),
            "dataset_bytes": _dir_bytes(ds) if ds else None, "measured_on": "local_cpu"}


def record_usage(out_dir: Path, status: dict) -> None:
    """Append one line for a finished attempt to ``out/usage/jobs.jsonl``: a retried job has one line per attempt,
    told apart by ``job_attempts``. The ledger outlives ``arrival gc``."""
    usage = status.get("usage")
    if not isinstance(usage, dict):
        return
    line = {"id": status.get("id"), "room_id": status.get("room_id"), "status": status.get("status"),
            "finished_at": status.get("finished_at"), "job_attempts": status.get("job_attempts"),
            "attempts": status.get("attempts"), "successes": status.get("successes"), **usage}
    path = Path(out_dir).joinpath(*USAGE_LEDGER)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:  # one write per line: O_APPEND keeps concurrent workers' lines whole
        f.write(json.dumps(line) + "\n")


def _stage(job_dir: Path, status: dict, stage: str, log, meter: Meter | None = None) -> None:
    if (Path(job_dir) / CANCEL).exists():
        raise JobCancelled("cancelled before stage " + stage)
    status["stage"] = stage
    if meter is not None:
        meter.enter(stage)
        status["usage"] = build_usage(status, meter.wall(), meter.seconds)
    _write_status(job_dir, status)
    log(f"[work] {status['id']}: {stage}")


def objects_status(twin: dict) -> dict:
    """What the job status says about the twin's objects: how many carry each provenance label and how the
    detection run went (`not_run` with a reason when it did not read a frame, or when the scan lists its objects)."""
    from arrival.export.provenance import object_counts

    run = twin.get("object_detection") or {"state": "not_run", "reason": "the scan lists its objects"}
    return {"counts": object_counts(twin), "detection": run["state"], "reason": run.get("reason"),
            "reply_sources": run.get("reply_sources", {}), "rejected": len(run.get("rejected", []))}


def progress_log(job_dir: Path, status: dict, total: int, log):
    """Wrap ``log`` so each finished sim task (an ``[sim] <task>: OK|MISS|DROP|SKIP`` line from ``collect``) bumps
    ``status["progress"] = {"done", "total"}`` and rewrites status.json. ``total`` is the requested episode count, which
    is also the task count, so ``done`` is capped at it."""
    status["progress"] = {"done": 0, "total": int(total)}
    _write_status(job_dir, status)

    def hook(msg=""):
        log(msg)
        if SIM_TASK_RE.match(str(msg)):
            p = status["progress"]
            p["done"] = min(p["done"] + 1, p["total"])
            _write_status(job_dir, status)
    return hook


# ---- claiming, heartbeat, stale reaping -------------------------------------------------------------------------

def _create_lock(path: Path, owner: str) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"worker": owner, "host": socket.gethostname(), "pid": os.getpid(), "claimed_unix": time.time()}, f)
    return True


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _pid_alive(pid) -> bool | None:
    """True/False for a process on this host, None when it cannot be told (Windows: os.kill(pid, 0) would kill)."""
    if sys.platform == "win32" or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_is_stale(job_dir: Path, stale_seconds: float, now: float | None = None) -> bool:
    """True when the job's lock (or, with no lock, its status.json) has not been touched for ``stale_seconds``, or the
    worker that wrote the lock is a dead process on this host."""
    now = time.time() if now is None else now
    lock = Path(job_dir) / LOCK
    try:
        mtime = lock.stat().st_mtime
    except FileNotFoundError:
        try:
            mtime = (Path(job_dir) / "status.json").stat().st_mtime
        except OSError:
            return False
        return now - mtime > stale_seconds
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    if isinstance(info, dict) and info.get("host") == socket.gethostname() and _pid_alive(info.get("pid")) is False:
        return True
    return now - mtime > stale_seconds


class Heartbeat:
    """Touch ``claim.lock`` every ``interval`` seconds from a daemon thread while a job runs."""

    def __init__(self, job_dir: Path, interval: float = 10.0) -> None:
        self.lock = Path(job_dir) / LOCK
        self.interval = interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                os.utime(self.lock)
            except OSError:
                pass

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


def claim_next(jobs_root: Path, worker: str | None = None) -> tuple[Path, dict] | None:
    """Claim the oldest queued job by ``created_unix`` (then id): create its ``claim.lock`` exclusively, then mark it
    running and count the start in ``job_attempts``. A queued job whose ``not_before`` is still in the future is
    skipped. Safe with several workers; None when nothing is claimable. The caller must ``release`` the job when done."""
    jobs_root = Path(jobs_root)
    worker = worker or worker_id()
    if not jobs_root.is_dir():
        return None
    queued = []
    for d in jobs_root.iterdir():
        st = _read_status(d)
        if st is None:
            continue  # not a job, or a status.json this worker cannot read; leave it alone
        try:
            created = int(st.get("created_unix", 0))
        except (TypeError, ValueError):
            continue
        if st.get("status") == "queued" and "id" in st:
            ready = st.get("not_before")
            if isinstance(ready, (int, float)) and ready > time.time():
                continue
            queued.append((created, d.name, d))
    for _, _, job_dir in sorted(queued):
        if not _create_lock(job_dir / LOCK, worker):
            continue  # another worker owns it
        st = _read_status(job_dir)  # re-read under the lock: it may have finished or been cancelled since the scan
        if st is None or st.get("status") != "queued":
            _unlink(job_dir / LOCK)
            continue
        ready = st.get("not_before")
        if isinstance(ready, (int, float)) and ready > time.time():
            _unlink(job_dir / LOCK)
            continue
        n = st.get("job_attempts")
        st.update({"job_attempts": (n if isinstance(n, int) and n > 0 else 0) + 1,
                   "status": "running", "stage": None, "started_at": int(time.time()), "finished_at": None,
                   "error": None, "room_id": None, "twin_dir": None, "dataset_dir": None, "zip": None,
                   "attempts": None, "successes": None, "skipped": None, "error_kind": None,
                   "error_reasons": None, "realism": None, "usage": None, "progress": None,
                   "worker": worker})
        if st.get("frames"):
            st["frames"].update(consumed=False, used_for=None)
        return job_dir, _write_status(job_dir, st)
    return None


def release(job_dir: Path) -> None:
    _unlink(Path(job_dir) / LOCK)


def requeue_failed(job_dir: Path, status: dict, limits: Limits, log=print) -> bool:
    """Turn a failed attempt's ``status`` back into ``queued`` when retries are left and no cancel was asked for: the
    attempt's ``error`` moves to ``retry_errors``, so ``error`` stays null on a job that is not finished. ``not_before``
    is now plus ``limits.job_retry_backoff`` seconds, doubling per attempt, capped at 300. The caller holds
    ``claim.lock`` and writes the status, so the job is claimable only after ``release`` and that time. Returns False,
    and leaves ``status`` alone, when the failure stands."""
    errors = status.get("retry_errors") if isinstance(status.get("retry_errors"), list) else []
    if len(errors) >= limits.job_retries:
        if limits.job_retries:
            log(f"[work] {status['id']}: not retried, all {limits.job_retries} retries used")
        return False
    if (Path(job_dir) / CANCEL).exists():
        log(f"[work] {status['id']}: not retried, a cancel was requested")
        return False
    delay = min(300, limits.job_retry_backoff * (2 ** len(errors)))
    status.update({"status": "queued", "stage": None, "started_at": None, "finished_at": None, "worker": None,
                   "error": None, "retry_errors": [*errors, status.get("error")],
                   "not_before": int(time.time() + delay)})
    log(f"[work] {status['id']}: queued again, retry {len(errors) + 1} of {limits.job_retries}")
    return True


def reap_stale(jobs_root: Path, stale_seconds: float = 120.0, log=print, now: float | None = None) -> int:
    """Requeue ``running`` jobs whose worker died (stale heartbeat or dead local pid), and clear a stale lock left on a
    ``queued`` job. Safe to call from every worker: ``reap.lock`` lets one worker act on a job at a time and the
    staleness is re-checked under it. A job requeued ``MAX_REQUEUES`` times fails; a running job with a cancel marker
    becomes ``cancelled``. Returns the number of jobs changed."""
    jobs_root = Path(jobs_root)
    n = 0
    for d in sorted(jobs_root.iterdir()) if jobs_root.is_dir() else []:
        st = _read_status(d)
        if st is None or "id" not in st or st.get("status") not in ("running", "queued"):
            continue
        if st["status"] == "queued" and not (d / LOCK).exists():
            continue
        if not lock_is_stale(d, stale_seconds, now):
            continue
        if not _create_lock(d / REAP_LOCK, worker_id()):
            try:  # a reaper that died holding the arbiter must not block this job forever
                if (time.time() if now is None else now) - (d / REAP_LOCK).stat().st_mtime > stale_seconds:
                    _unlink(d / REAP_LOCK)
            except OSError:
                pass
            continue
        try:
            st = _read_status(d)  # re-check under the arbiter
            if st is None or st.get("status") not in ("running", "queued") or not lock_is_stale(d, stale_seconds, now):
                continue
            _unlink(d / LOCK)
            if st["status"] == "queued":
                log(f"[work] {st['id']}: cleared a stale claim lock")
                n += 1
                continue
            done = int(time.time())
            if (d / CANCEL).exists():
                st.update({"status": "cancelled", "finished_at": done, "error": None})
                _unlink(d / CANCEL)
                log(f"[work] {st['id']}: cancelled (its worker died)")
            elif int(st.get("requeues") or 0) >= MAX_REQUEUES:
                st.update({"status": "failed", "finished_at": done,
                           "error": f"WorkerLost: the worker died {MAX_REQUEUES + 1} times while running this job"})
                log(f"[work] {st['id']}: failed, worker lost too many times")
            else:
                st.update({"status": "queued", "stage": None, "started_at": None, "worker": None,
                           "requeues": int(st.get("requeues") or 0) + 1})
                log(f"[work] {st['id']}: worker is gone, requeued")
            _write_status(d, st)
            n += 1
        finally:
            _unlink(d / REAP_LOCK)
    return n


def requeue_running(jobs_root: Path, log=print) -> int:
    """Put every ``running`` job back in the queue unconditionally, ignoring heartbeats. Only for an operator who knows
    no worker is running; ``work`` uses ``reap_stale``, which is safe beside other workers."""
    jobs_root = Path(jobs_root)
    n = 0
    for d in sorted(jobs_root.iterdir()) if jobs_root.is_dir() else []:
        st = _read_status(d)
        if st is not None and st.get("status") == "running" and "id" in st:
            _unlink(d / LOCK)
            st.update({"status": "queued", "stage": None, "started_at": None, "worker": None})
            _write_status(d, st)
            log(f"[work] {st['id']}: was running with no worker, requeued")
            n += 1
    return n


def cancel_job(jobs_root: Path, job_id: str) -> str | None:
    """Ask for a job to stop. Returns None for an unknown id, ``"cancelled"`` for a job that was still queued (done
    now), ``"cancelling"`` for a running one (it stops at the next stage boundary, or at once when the worker runs
    jobs in a child process), and the job's status for one that already finished."""
    job_dir = Path(jobs_root) / job_id
    st = _read_status(job_dir)
    if st is None:
        return None
    if st.get("status") == "queued" and _create_lock(job_dir / LOCK, "cancel"):  # claim it so no worker takes it
        try:
            st = _read_status(job_dir) or st
            if st.get("status") == "queued":
                st.update({"status": "cancelled", "finished_at": int(time.time())})
                _write_status(job_dir, st)
                return "cancelled"
        finally:
            _unlink(job_dir / LOCK)
    if st.get("status") in ("queued", "running"):
        (job_dir / CANCEL).write_text(str(int(time.time())), encoding="utf-8")
        return "cancelling"
    return st.get("status")


# ---- one job --------------------------------------------------------------------------------------------------

def process(job_dir: Path, status: dict, out_dir: Path, episodes: int = 8, families=DEFAULT_FAMILIES,
            people: int = 1, res: int = 128, seed: int = 0, log=print, limits: Limits | None = None) -> dict:
    """Run every stage for one claimed job, one attempt. Returns the status the attempt left: ``done``, ``failed``,
    ``cancelled``, or ``queued`` again for a retry (``requeue_failed``); never raises (except Ctrl-C and SystemExit)."""
    out_dir, job_dir = Path(out_dir), Path(job_dir)
    limits = limits or Limits()
    meter = Meter()
    retry = False
    status["realism"] = status["objects"] = None
    if status.get("frames"):
        status["frames"].update(consumed=False, used_for=None)
    try:
        limits.check_request(episodes, people)
        from arrival.export.provenance import sha256_file
        from arrival.export.validate import validate
        from arrival.twin.compile import compile_twin, fill_objects, write_twin

        room = room_id_for(status.get("filename", ""), status["id"])
        status["room_id"] = room
        _stage(job_dir, status, "normalize", log, meter)
        upload = job_dir / "upload.bin"
        if limits.max_upload_bytes and upload.stat().st_size > limits.max_upload_bytes:
            raise LimitError(f"upload is {upload.stat().st_size} bytes, over the limit of {limits.max_upload_bytes}")
        bundle = load_upload(upload, room)
        bundle["input_sha256"] = sha256_file(upload)  # provenance: the upload as received

        _stage(job_dir, status, "twin", log, meter)
        twin = compile_twin(bundle)

        _stage(job_dir, status, "realism", log, meter)
        from arrival.jobs.realism import measure

        status["realism"] = measure(job_dir, bundle, twin, log=log)
        if status.get("frames"):
            consumed = status["realism"]["consumed"]
            status["frames"].update(consumed=consumed, used_for="realism" if consumed else None)

        _stage(job_dir, status, "objects", log, meter)
        if not twin["objects"]:
            from arrival.twin.detect import detect_objects

            frames = job_dir / "frames"
            fill_objects(twin, detect_objects(frames, twin, bundle, log=log) if frames.is_dir() else None, seed=seed)
        status["objects"] = objects_status(twin)
        twin_dir = out_dir / "twins" / room
        write_twin(twin, twin_dir)
        status["twin_dir"] = str(twin_dir)

        _stage(job_dir, status, "sim", log, meter)
        from arrival.sim.collect import collect

        res_ = collect(twin, list(families), episodes, out_dir, people=people, res=res, seed=seed,
                       log=progress_log(job_dir, status, episodes, log))
        r = res_["report_data"]
        status.update({"dataset_dir": str(res_["dataset"]), "attempts": r["attempts"], "successes": r["successes"],
                       "skipped": r["skipped"]})
        if r["skipped"]:
            log(f"[work] {status['id']}: skipped {r['skipped']} ({r['skipped_unreachable']} unreachable, "
                f"{r['skipped_error']} errors); the dataset holds only episodes that passed the physics check")
        if r["attempts"] == 0:  # nothing could even be set up: a layout problem the uploader can act on
            why = [x["reason"] for x in r["skip_reasons"]][:3]
            raise UnsupportedLayout("no task can run in this room" + (": " + "; ".join(why) if why else ""), why)
        if r["successes"] == 0:
            raise UnusableDataset(f"no successful episodes in {r['attempts']} attempts")

        _stage(job_dir, status, "validate", log, meter)
        errors = validate(res_["dataset"])
        if errors:
            raise UnusableDataset("dataset invalid: " + "; ".join(errors[:5]))

        _stage(job_dir, status, "zip", log, meter)
        zip_path = job_dir / "dataset.zip"
        ds = Path(res_["dataset"])
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(ds.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(ds.name) / p.relative_to(ds))
        status.update({"zip": str(zip_path), "status": "done"})
    except JobCancelled:
        status.update({"status": "cancelled", "error": None})
        _unlink(job_dir / CANCEL)
        log(f"[work] {status['id']}: cancelled at {status.get('stage')}")
    except BaseException as e:  # noqa: BLE001 - any stage failure is the job's failure, not the worker's
        status.update({"status": "failed", "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__})
        retry = retryable(e)
        if isinstance(e, UnsupportedLayout):  # a typed status, not a traceback: the room, not the worker, is the problem
            status.update({"error": f"unsupported layout: {e}", "error_kind": "unsupported_layout",
                           "error_reasons": e.reasons})
            log(f"[work] {status['id']}: unsupported layout at {status.get('stage')}: {e}")
        elif not isinstance(e, Exception):  # Ctrl-C or SystemExit: record it so the job is not left running, then stop
            status["error"] = f"interrupted: {status['error']}"
            status["finished_at"] = int(time.time())
            meter.enter(None)
            status["usage"] = build_usage(status, meter.wall(), meter.seconds)
            _write_status(job_dir, status)
            raise
        else:
            (job_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[work] {status['id']}: failed at {status.get('stage')}: {status['error']}")
    status["finished_at"] = int(time.time())
    meter.enter(None)
    status["usage"] = build_usage(status, meter.wall(), meter.seconds)
    ended = dict(status)  # the ledger line is this attempt as it ended, also when the job goes back to the queue
    if retry:  # decided before the write: a client polling the job never sees "failed" for a job that will run again
        requeue_failed(job_dir, status, limits, log)
    _write_status(job_dir, status)
    try:
        record_usage(out_dir, ended)
    except OSError as e:  # metering must never turn a finished job into a failure
        log(f"[work] {status['id']}: usage ledger not written: {e}")
    if status["status"] == "done":
        log(f"[work] {status['id']}: done, {status['successes']}/{status['attempts']} episodes -> {status['zip']}")
    return status


def tree_rss(pid: int) -> int | None:
    """Resident bytes of process ``pid`` and every process under it (the video encoder it starts, for one): from /proc on
    Linux, from ``ps`` elsewhere. None when the process table cannot be read or ``pid`` is gone."""
    rows: dict[int, tuple[int, int]] = {}  # pid -> (parent pid, resident bytes)
    try:
        if sys.platform == "linux":
            page = os.sysconf("SC_PAGE_SIZE")
            for d in os.listdir("/proc"):
                if not d.isdigit():
                    continue
                try:
                    raw = Path("/proc", d, "stat").read_text()
                except OSError:
                    continue  # exited between the listing and the read
                f = raw[raw.rindex(")") + 2:].split()  # fields after "pid (comm) ": state, ppid, ... rss is the 22nd
                rows[int(d)] = (int(f[1]), int(f[21]) * page)
        else:
            ps = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=5)
            for line in ps.stdout.splitlines():
                p, pp, kb = line.split()
                rows[int(p)] = (int(pp), int(kb) * 1024)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if pid not in rows:
        return None
    kids: dict[int, list[int]] = {}
    for p, (pp, _) in rows.items():
        kids.setdefault(pp, []).append(p)
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        total += rows[p][1]
        todo += kids.get(p, [])
    return total


def run_isolated(job_dir: Path, status: dict, out_dir: Path, limits: Limits, log=print, child_cmd=None,
                 tick: float = 0.1, **kw) -> dict:
    """Run ``process`` in a child process and enforce ``limits.max_job_seconds`` and ``limits.max_job_bytes`` (resident
    memory of the child and everything it starts, read every MEM_TICK s) by killing it. The child writes its
    own status, retry included; this function only steps in when the child timed out, was cancelled, or died without
    finishing (killed by the OS, segfault), and then records the failure with the stage it reached. A child that died
    is retried like any other failure that may pass; a timeout is not, since the limit is what one job may cost."""
    job_dir, out_dir = Path(job_dir), Path(out_dir)
    payload = {k: (list(v) if k == "families" else v) for k, v in kw.items()
               if k in ("episodes", "families", "people", "res", "seed")}
    payload["limits"] = asdict(limits)
    cmd = list(child_cmd or [sys.executable, "-m", "arrival.jobs.child"]) + [str(job_dir), str(out_dir), json.dumps(payload)]
    t0 = time.monotonic()
    deadline = t0 + limits.max_job_seconds if limits.max_job_seconds else None
    proc = subprocess.Popen(cmd)
    why, next_mem, rss = None, t0, 0
    try:
        while proc.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                why = "timeout"
            elif (job_dir / CANCEL).exists():
                why = "cancel"
            elif limits.max_job_bytes and time.monotonic() >= next_mem:
                next_mem = time.monotonic() + MEM_TICK
                rss = tree_rss(proc.pid) or 0
                if rss > limits.max_job_bytes:
                    why = "memory"
            if why:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(tick)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    final = _read_status(job_dir) or status
    if final.get("status") != "running" and why is None:
        return final  # the child ended the attempt itself (done, failed, cancelled or queued again), ledger included
    done = (final.get("usage") or {}).get("stage_seconds") or {}
    stage = final.get("stage")
    if why == "timeout":
        final.update({"status": "failed", "error": f"TimeoutError: job exceeded the {limits.max_job_seconds:g} s "
                                                    f"wall-clock limit during stage {stage!r}"})
    elif why == "memory":  # like a timeout, not retried: the limit is what one job may use
        final.update({"status": "failed", "error": f"MemoryLimitError: the job used {rss / MB:.1f} MB resident, over "
                                                    f"the {limits.max_job_bytes / MB:g} MB limit (ARRIVAL_MAX_JOB_MB), "
                                                    f"during stage {stage!r}"})
    elif why == "cancel":
        final.update({"status": "cancelled", "error": None})
        _unlink(job_dir / CANCEL)
    elif final.get("status") == "running":
        final.update({"status": "failed", "error": f"WorkerProcessError: the job's process exited with code "
                                                    f"{proc.returncode} during stage {stage!r}"})
    if final["status"] == "failed":
        (job_dir / "error.txt").write_text(final["error"] + "\n", encoding="utf-8")
    final["finished_at"] = int(time.time())
    final["usage"] = build_usage(final, time.monotonic() - t0, done)  # the stage in flight is not split out
    log(f"[work] {final['id']}: {final['status']}: {final.get('error') or 'stopped'}")
    ended = dict(final)
    if why is None and final["status"] == "failed":  # the child died: out of memory or killed, which may not recur
        requeue_failed(job_dir, final, limits, log)
    _write_status(job_dir, final)
    try:
        record_usage(out_dir, ended)
    except OSError as e:
        log(f"[work] {final['id']}: usage ledger not written: {e}")
    return final


def run_once(out_dir: Path, limits: Limits | None = None, isolate: bool = False, worker: str | None = None,
             child_cmd=None, **kw) -> dict | None:
    """Claim and process one queued job (one attempt; a failure that may pass leaves it ``queued`` for the next claim).
    None when nothing is queued. With ``isolate`` the job runs in a child process under ``limits.max_job_seconds``;
    otherwise in this process (no timeout)."""
    limits = limits or Limits()
    claimed = claim_next(Path(out_dir) / "jobs", worker)
    if claimed is None:
        return None
    job_dir, status = claimed
    try:
        with Heartbeat(job_dir, min(10.0, max(0.05, limits.stale_seconds / 4))):
            if isolate:
                return run_isolated(job_dir, status, out_dir, limits, child_cmd=child_cmd, **kw)
            return process(job_dir, status, out_dir, limits=limits, **kw)
    finally:
        release(job_dir)


def work(out_dir: Path, once: bool = False, poll: float = 2.0, log=print, limits: Limits | None = None,
         isolate: bool | None = None, worker: str | None = None, **kw) -> int:
    """Poll until Ctrl-C (or one job with ``once``). Returns the number of job attempts run (a retry counts again).

    ``limits`` default to ``Limits.from_env()``; sim options above them raise ``LimitError`` before any job is touched.
    ``isolate`` defaults to on when ``limits.max_job_seconds`` is set. Other workers may run on the same ``out``."""
    limits = limits or Limits.from_env()
    limits.check_request(int(kw.get("episodes", 8)), int(kw.get("people", 1)))
    if isolate is None:
        isolate = bool(limits.max_job_seconds)
    out_dir = Path(out_dir)
    jobs = out_dir / "jobs"
    n = 0
    next_reap = next_gc = 0.0
    try:
        while True:
            now = time.monotonic()
            if now >= next_reap:
                reap_stale(jobs, limits.stale_seconds, log=log)
                next_reap = now + max(poll, limits.stale_seconds / 4)
            if limits.retention_hours and now >= next_gc:
                from arrival.jobs.retention import gc

                for jid in gc(out_dir, limits.retention_hours, log=log)["removed"]:
                    log(f"[work] retention: removed {jid}")
                next_gc = now + 300
            st = run_once(out_dir, limits=limits, isolate=isolate, worker=worker, log=log, **kw)
            if st is not None:
                n += 1
            if once:
                if st is None:
                    log("[work] no queued jobs")
                return n
            if st is None:
                time.sleep(poll)
    except KeyboardInterrupt:
        return n
