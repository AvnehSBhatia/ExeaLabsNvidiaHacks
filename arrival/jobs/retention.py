"""Retention: delete finished jobs and what they produced, once they are old enough.

``python -m arrival gc --max-age-hours N`` and the worker's sweep (``ARRIVAL_RETENTION_HOURS``) call ``gc``; a future
token-gated ``DELETE /jobs/<id>`` would call ``delete_job``. A job is removed with its ``out/jobs/<id>/`` folder (the
upload, status, zip), its twin ``out/twins/<room>/``, its dataset ``out/datasets/<room>_sim/``, its sim report and its
sim progress journal ``out/progress/<room>_sim.jsonl``. Queued and running jobs are never touched. The usage ledger
``out/usage/jobs.jsonl`` is kept so metering survives deletion. Only paths derived from the job id are removed: a room
that does not end with the job id (a hand-made room such as ``hero_kitchen``) is never deleted.
"""

from __future__ import annotations

import math
import re
import shutil
import time
from pathlib import Path

from arrival.jobs.worker import _read_status

FINISHED = ("done", "failed", "cancelled")
ID_RE = re.compile(r"^[a-z0-9_]+$")


class JobBusy(RuntimeError):
    """The job is queued or running and cannot be deleted."""


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def job_paths(out_dir: Path, job_id: str, status: dict) -> list[Path]:
    """Everything a job owns under ``out_dir`` that exists on disk, job folder last."""
    out = Path(out_dir)
    paths = []
    room = status.get("room_id")
    if isinstance(room, str) and ID_RE.match(room) and room.endswith(job_id):
        paths += [out / "twins" / room, out / "datasets" / f"{room}_sim", out / "reports" / f"{room}_sim.json",
                  out / "progress" / f"{room}_sim.jsonl"]
    paths.append(out / "jobs" / job_id)
    return [p for p in paths if p.exists()]


def delete_job(out_dir: Path, job_id: str, dry_run: bool = False) -> dict | None:
    """Delete one finished job. None for an unknown id; raises ``JobBusy`` for a queued or running one."""
    if not isinstance(job_id, str) or not ID_RE.match(job_id):
        return None
    out = Path(out_dir)
    status = _read_status(out / "jobs" / job_id)
    if status is None or status.get("id") != job_id:
        return None
    if status.get("status") not in FINISHED:
        raise JobBusy(f"{job_id} is {status.get('status')}; cancel it first")
    paths = job_paths(out, job_id, status)
    freed = sum(_size(p) for p in paths)
    if not dry_run:
        # A file that cannot be removed (held open on Windows, no permission) raises OSError. status.json goes last so
        # the job stays listed and a later gc or DELETE retries it instead of leaving files no job points to.
        *owned, folder = paths
        for p in owned + [c for c in folder.iterdir() if c.name != "status.json"]:
            _remove(p)
        _remove(folder / "status.json")
        folder.rmdir()
    return {"id": job_id, "bytes": freed, "paths": [str(p) for p in paths]}


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def gc(out_dir: Path, max_age_hours: float, now: float | None = None, dry_run: bool = False, log=print) -> dict:
    """Delete finished jobs whose ``finished_at`` (else ``created_unix``) is older than ``max_age_hours``."""
    if (type(max_age_hours) not in (int, float) or not math.isfinite(max_age_hours)
            or max_age_hours <= 0):
        raise ValueError("max_age_hours must be finite and > 0")
    now = time.time() if now is None else now
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError("now must be a finite timestamp")
    root = Path(out_dir) / "jobs"
    removed, freed = [], 0
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        st = _read_status(d)
        if st is None or st.get("status") not in FINISHED or d.name != st.get("id"):
            continue
        stamp = st.get("finished_at") or st.get("created_unix")
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or now - stamp <= max_age_hours * 3600):
            continue
        try:
            res = delete_job(out_dir, d.name, dry_run=dry_run)
        except OSError as e:
            log(f"[gc] {d.name}: {e}")
            continue
        if res:
            removed.append(d.name)
            freed += res["bytes"]
    return {"removed": removed, "bytes": freed, "dry_run": dry_run}
