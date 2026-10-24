"""Multi-room summary (L28): run the pipeline on a few rooms with tiny settings and tabulate what worked.

One row per room, one cell per stage: normalize -> compile -> tasks -> sim -> validate (structural, then per episode).
A stage that fails records the reason and the later stages are ``not_run``; nothing is retried or papered over.
Two columns are filled only when there is something real to put in them, and say so otherwise:

* tape error: only when a ``measurements.json`` exists for the room, and flagged ``authored`` when the numbers were
  derived from the scan itself rather than taken with a tape (hero_kitchen's are authored)
* realism: always ``null`` here. It needs held-out RGB-D frames of the room, which no room in this repo has.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOMS = ("hero_kitchen", "tests/fixtures/roomplan/amal.json", "tests/fixtures/roomplan/shopyseek.json")
STAGES = ("normalize", "compile", "tasks", "sim", "validate", "episodes")
REALISM_NOTE = "not measured: needs held-out RGB-D frames of the room (none exist for any room here)"


def _stage(status: str, detail: str = "", **extra) -> dict:
    return {"status": status, "detail": detail, **extra}


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


def resolve_room(spec: str, scans_dir: Path) -> tuple[str, Path]:
    """``hero_kitchen`` (a scan under scans_dir) or a path to a RoomPlan / scan.json file -> (room_id, file)."""
    p = Path(spec)
    if p.suffix == ".json":
        cand = p if p.is_file() else REPO / p
        if not cand.is_file():
            raise FileNotFoundError(f"no such scan file: {spec}")
        # a generic file name (scans/<room>/scan.json, roomplan.json) names the room after its folder, so two
        # rooms passed as scan.json files are two rows and two datasets, not both "scan"
        stem = cand.parent.name if cand.stem.lower() in ("scan", "roomplan") else cand.stem
        room = "".join(c if c.isalnum() else "_" for c in stem.lower()).strip("_") or "room"
        return room, cand
    for name in ("scan.json", "roomplan.json"):
        if (scans_dir / spec / name).is_file():
            return spec, scans_dir / spec / name
    raise FileNotFoundError(f"room {spec!r} not found under {scans_dir}")


def _measurements(room: str, scans_dir: Path) -> Path | None:
    for p in (scans_dir / room / "measurements.json", REPO / "tests" / "fixtures" / "measurements" / f"{room}.json"):
        if p.is_file():
            return p
    return None


def run_room(spec: str, out_dir: Path, scans_dir: Path, episodes: int = 2, res: int = 64, people: int = 1,
             families: tuple[str, ...] = ("counter_to_sink", "open_drawer"), seed: int = 0, log=lambda *_: None) -> dict:
    """Run one room through every stage. Never raises: each failure lands in that stage's cell."""
    row: dict = {"room": spec, "stages": {s: _stage("not_run") for s in STAGES}, "tape": None, "realism": None,
                 "realism_note": REALISM_NOTE}
    st = row["stages"]

    def fail(stage: str, why: str) -> dict:
        st[stage] = _stage("failed", why)
        for later in STAGES[STAGES.index(stage) + 1:]:
            st[later] = _stage("not_run", f"{stage} failed")
        return row

    try:
        room, path = resolve_room(spec, scans_dir)
        row["room"] = room
        from arrival.jobs.worker import load_upload

        bundle = load_upload(path, room)
        row["geometry_source"] = bundle.get("geometry_source")
        st["normalize"] = _stage("ok", f"{path.name}: {len(bundle['walls'])} walls, {len(bundle['fixtures'])} fixtures")
    except Exception as e:  # noqa: BLE001 - one room's bad input must not stop the table
        return fail("normalize", _err(e))
    try:
        from arrival.twin.compile import compile_twin, place_default_objects, write_twin

        twin = compile_twin(bundle)
        placed = False
        if not twin["objects"]:
            place_default_objects(twin, seed=seed)  # same step the upload worker takes
            placed = True
        write_twin(twin, Path(out_dir) / "twins" / room)
        kinds: dict[str, int] = {}
        for f in twin["fixtures"]:
            kinds[f["type"]] = kinds.get(f["type"], 0) + 1
        st["compile"] = _stage("ok", f"{len(twin['fixtures'])} fixtures, {len(twin['objects'])} objects"
                                     + (" (default_synthetic: invented, not scanned)" if placed else ""),
                               fixtures=kinds, objects=len(twin["objects"]), objects_default_synthetic=placed)
    except Exception as e:  # noqa: BLE001
        return fail("compile", _err(e))
    mpath = _measurements(room, scans_dir)
    if mpath:
        try:
            from arrival.twin.accuracy import compare, load_measurements

            cmp = compare(twin, load_measurements(mpath))
            errs = [abs(d["error_m"]) for d in cmp["dimensions"].values() if d["error_m"] is not None]
            row["tape"] = {"max_abs_error_m": round(max(errs), 4) if errs else None, "tolerance_m": cmp["tolerance_m"],
                           "ok": cmp["ok"], "tape_measured": cmp["tape_measured"],
                           "dimensions": {k: v["error_m"] for k, v in cmp["dimensions"].items()}}
        except Exception as e:  # noqa: BLE001 - e.g. "no scanned counter to measure"
            row["tape"] = {"error": _err(e), "tape_measured": False}
    try:
        from arrival.sim.tasks import make_tasks

        tasks = make_tasks(twin, list(families), episodes, seed=seed, people=people, log=log)
        if not tasks:
            st["tasks"] = _stage("skipped", f"no reachable task in {', '.join(families)} for a fixed-base arm")
            for later in ("sim", "validate", "episodes"):
                st[later] = _stage("not_run", "no tasks to simulate")
            return row
        st["tasks"] = _stage("ok", f"{len(tasks)} task(s): " + ", ".join(sorted({t['family'] for t in tasks})))
    except Exception as e:  # noqa: BLE001
        return fail("tasks", _err(e))
    try:
        from arrival.sim.collect import collect

        t0, lines = time.perf_counter(), []

        def keep(*parts):
            lines.append(" ".join(str(x) for x in parts))
            log(*parts)

        res_ = collect(twin, list(families), episodes, out_dir, people=people, res=res, seed=seed, log=keep)
        r = res_["report_data"]
        misses = [ln.split("[sim] ", 1)[-1][:160] for ln in lines if ": MISS" in ln or ": DROP" in ln]
        st["sim"] = _stage("ok" if r["successes"] else "failed",
                           f"{r['successes']}/{r['attempts']} episodes passed the physics check in "
                           f"{time.perf_counter() - t0:.1f}s" + (f"; {misses[0]}" if misses else ""),
                           attempts=r["attempts"], successes=r["successes"], human_contacts=r["human_contacts"],
                           pass_rate=round(r["successes"] / r["attempts"], 3) if r["attempts"] else None,
                           misses=misses[:5])
        if not r["successes"]:  # keep the attempt counts in the cell; the later stages have nothing to check
            for later in ("validate", "episodes"):
                st[later] = _stage("not_run", "sim failed")
            return row
        ds = Path(res_["dataset"])
        info = json.loads((ds / "meta" / "info.json").read_text())
        st["sim"]["arrival_physically_valid"] = info.get("arrival_physically_valid")
    except Exception as e:  # noqa: BLE001
        return fail("sim", _err(e))
    try:
        from arrival.export.validate import validate

        errors = validate(ds)
        st["validate"] = _stage("ok" if not errors else "failed", "structural check passed" if not errors
                                else f"{len(errors)} error(s): " + "; ".join(errors[:3]))
    except Exception as e:  # noqa: BLE001
        st["validate"] = _stage("failed", _err(e))
    try:
        from arrival.export.quality import check_episodes

        q = check_episodes(ds)
        st["episodes"] = _stage("ok" if not q["flagged"] else "failed",
                                f"{q['kept']}/{q['episodes']} episodes pass the per-episode checks"
                                + (f"; flagged {q['flagged_episodes']}: {q['flags_by_check']}" if q["flagged"] else ""),
                                episodes=q["episodes"], flagged=q["flagged"])
    except Exception as e:  # noqa: BLE001
        st["episodes"] = _stage("failed", _err(e))
    row["dataset"] = str(ds)
    return row


def summarize(rooms=DEFAULT_ROOMS, out_dir: str | Path = "out/summary", scans_dir: str | Path = "scans",
              episodes: int = 2, res: int = 64, people: int = 1,
              families: tuple[str, ...] = ("counter_to_sink", "open_drawer"), seed: int = 0, log=print) -> dict:
    out_dir, scans_dir = Path(out_dir), Path(scans_dir)
    rows = []
    for spec in rooms:
        log(f"[summary] {spec}")
        rows.append(run_room(spec, out_dir, scans_dir, episodes, res, people, tuple(families), seed))
    return {"settings": {"episodes_per_room": episodes, "res": res, "people": people, "families": list(families),
                         "seed": seed},
            "note": "tiny runs to show which stages work per room, not a benchmark",
            "rooms": rows}


def _cell(stage: dict) -> str:
    return {"ok": "ok", "failed": "FAILED", "skipped": "skipped", "not_run": "-"}.get(stage["status"], stage["status"])


def to_markdown(summary: dict) -> str:
    s = summary["settings"]
    lines = [f"# Room summary ({s['episodes_per_room']} episodes per room, {s['res']} px, {s['people']} person)", "",
             "| room | geometry | normalize | compile | tasks | sim | validate | per-episode | tape error | realism |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["rooms"]:
        st = r["stages"]
        sim = _cell(st["sim"]) + (f" {st['sim']['successes']}/{st['sim']['attempts']}"
                                  if "attempts" in st["sim"] else "")
        t = r["tape"]
        if t is None:
            tape = "not measured"
        elif "error" in t:
            tape = f"n/a ({t['error']})"
        elif t.get("max_abs_error_m") is None:
            tape = "n/a (no comparable dimensions)"
        else:
            tape = (f"{t['max_abs_error_m'] * 100:.1f} cm max" + ("" if t["tape_measured"] else " (authored, not tape)"))
        lines.append(f"| {r['room']} | {r.get('geometry_source') or '-'} | {_cell(st['normalize'])} | "
                     f"{_cell(st['compile'])} | {_cell(st['tasks'])} | {sim} | {_cell(st['validate'])} | "
                     f"{_cell(st['episodes'])} | {tape} | not measured |")
    lines += ["", "## Details", ""]
    for r in summary["rooms"]:
        lines.append(f"### {r['room']}")
        for name in STAGES:
            c = r["stages"][name]
            lines.append(f"- {name}: {_cell(c)}" + (f", {c['detail']}" if c["detail"] else ""))
        lines.append("")
    lines += [f"Realism: {REALISM_NOTE}.", "",
              "Rooms with `default_synthetic` objects hold invented objects (a RoomPlan export has none); only the "
              "fixtures are scanned.", ""]
    return "\n".join(lines)


def write_summary(summary: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "summary.json", out_dir / "summary.md"
    jp.write_text(json.dumps(summary, indent=2) + "\n")
    mp.write_text(to_markdown(summary))
    return jp, mp
