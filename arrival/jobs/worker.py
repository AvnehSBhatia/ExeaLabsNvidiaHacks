"""Drain out/jobs: upload -> twin -> MuJoCo sim dataset -> dataset.zip, one job at a time.

``python -m arrival work`` runs this loop. The web server never imports it. Every
stage boundary rewrites ``status.json``; a failure is recorded there and the loop
moves on, so one bad upload never stops the queue.
"""

from __future__ import annotations

import json
import re
import time
import traceback
import zipfile
from pathlib import Path

DEFAULT_FAMILIES = ("counter_to_sink", "open_drawer")
STAGES = ("normalize", "twin", "objects", "sim", "validate", "zip")
ROOM_RE = re.compile(r"[^a-z0-9_]+")


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
        bundle["room_id"] = room_id
        return bundle
    if any(isinstance(x, dict) and "transform" in x for k in ("walls", "objects") for x in data.get(k, [])):
        return roomplan.to_bundle(data, room_id)
    raise ValueError("upload is neither a RoomPlan export (walls/objects with transform) nor a scan.json (schema_version)")


def _write_status(job_dir: Path, status: dict) -> dict:
    (job_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def _stage(job_dir: Path, status: dict, stage: str, log) -> None:
    status["stage"] = stage
    _write_status(job_dir, status)
    log(f"[work] {status['id']}: {stage}")


def claim_next(jobs_root: Path) -> tuple[Path, dict] | None:
    """Oldest queued job by ``created_unix`` (then id), marked running. None when the queue is empty."""
    jobs_root = Path(jobs_root)
    if not jobs_root.is_dir():
        return None
    queued = []
    for d in jobs_root.iterdir():
        try:
            st = json.loads((d / "status.json").read_text(encoding="utf-8"))
            created = int(st.get("created_unix", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            continue  # not a job, or a status.json this worker cannot read; leave it alone
        if isinstance(st, dict) and st.get("status") == "queued" and "id" in st:
            queued.append((created, d.name, d, st))
    if not queued:
        return None
    _, _, job_dir, st = min(queued)
    st.update({"status": "running", "stage": None, "started_at": int(time.time()), "finished_at": None,
               "error": None, "room_id": None, "twin_dir": None, "dataset_dir": None, "zip": None,
               "attempts": None, "successes": None})
    return job_dir, _write_status(job_dir, st)


def process(job_dir: Path, status: dict, out_dir: Path, episodes: int = 8, families=DEFAULT_FAMILIES,
            people: int = 1, res: int = 128, seed: int = 0, mobile_base: bool = False, log=print) -> dict:
    """Run every stage for one claimed job. Returns the final status; never raises."""
    out_dir, job_dir = Path(out_dir), Path(job_dir)
    try:
        from arrival.export.validate import validate
        from arrival.twin.compile import compile_twin, place_default_objects, write_twin

        room = room_id_for(status.get("filename", ""), status["id"])
        status["room_id"] = room
        _stage(job_dir, status, "normalize", log)
        bundle = load_upload(job_dir / "upload.bin", room)

        _stage(job_dir, status, "twin", log)
        twin = compile_twin(bundle)

        _stage(job_dir, status, "objects", log)
        if not twin["objects"]:
            place_default_objects(twin, seed=seed)
        twin_dir = out_dir / "twins" / room
        write_twin(twin, twin_dir)
        status["twin_dir"] = str(twin_dir)

        _stage(job_dir, status, "sim", log)
        from arrival.sim.collect import collect

        res_ = collect(twin, list(families), episodes, out_dir, people=people, res=res, seed=seed,
                       mobile_base=mobile_base, log=log)
        r = res_["report_data"]
        status.update({"dataset_dir": str(res_["dataset"]), "attempts": r["attempts"], "successes": r["successes"]})
        if r["successes"] == 0:
            raise RuntimeError(f"no successful episodes in {r['attempts']} attempts")

        _stage(job_dir, status, "validate", log)
        errors = validate(res_["dataset"])
        if errors:
            raise RuntimeError("dataset invalid: " + "; ".join(errors[:5]))

        _stage(job_dir, status, "zip", log)
        zip_path = job_dir / "dataset.zip"
        ds = Path(res_["dataset"])
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(ds.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(ds.name) / p.relative_to(ds))
        status.update({"zip": str(zip_path), "status": "done"})
    except Exception as e:  # noqa: BLE001 - any stage failure is the job's failure, not the worker's
        status.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        (job_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"[work] {status['id']}: failed at {status.get('stage')}: {status['error']}")
    status["finished_at"] = int(time.time())
    _write_status(job_dir, status)
    if status["status"] == "done":
        log(f"[work] {status['id']}: done, {status['successes']}/{status['attempts']} episodes -> {status['zip']}")
    return status


def run_once(out_dir: Path, **kw) -> dict | None:
    """Claim and process one queued job. None when nothing is queued."""
    claimed = claim_next(Path(out_dir) / "jobs")
    if claimed is None:
        return None
    job_dir, status = claimed
    return process(job_dir, status, out_dir, **kw)


def work(out_dir: Path, once: bool = False, poll: float = 2.0, log=print, **kw) -> int:
    """Poll until Ctrl-C (or one job with ``once``). Returns the number of jobs processed."""
    n = 0
    try:
        while True:
            st = run_once(out_dir, log=log, **kw)
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
