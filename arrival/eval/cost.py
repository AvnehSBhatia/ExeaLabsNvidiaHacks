"""Time and cost report computed from local runs: python -m arrival.eval.cost [--out out].

Reads out/reports/*.json (demo and sim reports), out/eval/*/report.json, legacy out/eval/*.json, out/jobs/*/status.json and the per-job usage ledger
out/usage/jobs.jsonl (the worker's metering: wall seconds, episodes, frames, bytes; it outlives `arrival gc`, so a deleted
job keeps its row). Metered units are measured on the local machine and carry no price. Throughput numbers are
whatever those files hold. A dollar row is measured only when a usage file supplied the units AND the pricing line
says its rate is invoiced; with the placeholder rates of configs/pricing.example.yaml it is an estimate and says so.
The generated-vs-teleop comparison prints only when a valid teleop_baseline.json exists.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

# report-schema key (docs/04-contracts.md) -> pricing product that fills it
PRODUCT_KEYS = {"serverless_jobs": "gpu_hour", "endpoint": "cpu_hour", "object_storage": "object_storage_gb_month",
                "token_factory": "token_factory_per_mtoken"}
BASELINE_FIELDS = ("task", "room", "n_success", "operator_seconds_total", "setup_assumptions", "cost_basis_usd")


class CostError(ValueError):
    """A pricing, usage or baseline file that cannot be used."""


def _load(path: str | Path, what: str) -> dict:
    try:
        text = Path(path).read_text()
        data = yaml.safe_load(text) if str(path).endswith((".yaml", ".yml")) else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise CostError(f"{what} {path}: cannot read ({type(e).__name__}: {e})") from e
    if not isinstance(data, dict):
        raise CostError(f"{what} {path}: expected an object at the top level")
    return data


def _num(v) -> bool:
    return type(v) is int or (type(v) is float and math.isfinite(v))


def load_baseline(path: str | Path) -> dict:
    """Validate teleop_baseline.json: {task, room, n_success, operator_seconds_total, setup_assumptions, cost_basis_usd}."""
    b = _load(path, "teleop baseline")
    missing = [k for k in BASELINE_FIELDS if k not in b]
    if missing:
        raise CostError(f"teleop baseline {path}: missing {', '.join(missing)}")
    for k in ("task", "room"):
        if not isinstance(b[k], str) or not b[k].strip():
            raise CostError(f"teleop baseline {path}: {k} must be a non-empty string")
    if not isinstance(b["n_success"], int) or isinstance(b["n_success"], bool) or b["n_success"] < 1:
        raise CostError(f"teleop baseline {path}: n_success must be an integer >= 1")
    if not _num(b["operator_seconds_total"]) or b["operator_seconds_total"] <= 0:
        raise CostError(f"teleop baseline {path}: operator_seconds_total must be a number > 0")
    sa = b["setup_assumptions"]
    if not (isinstance(sa, str) and sa.strip()) and not (isinstance(sa, list) and sa and all(isinstance(s, str) for s in sa)):
        raise CostError(f"teleop baseline {path}: setup_assumptions must be a non-empty string or list of strings")
    if not _num(b["cost_basis_usd"]) or b["cost_basis_usd"] < 0:
        raise CostError(f"teleop baseline {path}: cost_basis_usd must be a number >= 0 (total operator and setup cost)")
    return b


def load_pricing(path: str | Path) -> dict:
    p = _load(path, "pricing")
    prods = p.get("products")
    if not isinstance(prods, dict):
        raise CostError(f"pricing {path}: missing 'products'")
    for name, row in prods.items():
        if not isinstance(row, dict) or not _num(row.get("usd_per_unit")) or row["usd_per_unit"] < 0:
            raise CostError(f"pricing {path}: {name} needs a numeric usd_per_unit >= 0")
    return prods


def load_usage(path: str | Path) -> dict:
    u = _load(path, "usage")
    for k, v in u.items():
        if k == "scope":
            if (not isinstance(v, dict) or set(v) != {"room", "task"} or
                    any(not isinstance(v[name], str) or not v[name].strip() for name in ("room", "task"))):
                raise CostError(f"usage {path}: scope must name a room and task")
            continue
        if k not in PRODUCT_KEYS.values():
            raise CostError(f"usage {path}: unknown product {k!r} (known: {', '.join(PRODUCT_KEYS.values())})")
        if not _num(v) or v < 0:
            raise CostError(f"usage {path}: {k} must be a number of units >= 0")
    return u


def _row(source: str, room: str, family: str | None, attempts, successes, seconds, human_contacts=None,
         frames=None) -> dict:
    ok = type(attempts) is int and type(successes) is int and 0 <= successes <= attempts
    secs = seconds if _num(seconds) and seconds > 0 else None
    return {"source": source, "room": room, "family": family, "attempts": attempts if ok else None,
            "successes": successes if ok else None, "seconds": secs, "human_contacts": human_contacts,
            "frames": frames if type(frames) is int and frames >= 0 else None}


METER_KEYS = ("wall_seconds", "episodes_attempted", "episodes_succeeded", "frames", "upload_bytes", "zip_bytes",
              "dataset_bytes")


def job_usage(out_dir: str | Path) -> dict[str, dict]:
    """Per-job metering by job id: the ledger first, then each status.json (which wins: it is the live record).
    A value is None when it was not measured; a job without a usage block (an older worker) is not here."""
    out = Path(out_dir)
    jobs: dict[str, dict] = {}
    ledger = out / "usage" / "jobs.jsonl"
    try:
        body = ledger.read_text(encoding="utf-8") if ledger.is_file() else ""
    except OSError as e:
        raise CostError(f"usage ledger {ledger}: cannot read ({type(e).__name__}: {e})") from e
    lines = body.splitlines()
    for index, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except ValueError as e:
            if index == len(lines) - 1 and not body.endswith("\n"):
                continue  # a kill may leave an incomplete final append
            raise CostError(f"usage ledger {ledger}: invalid JSON on line {index + 1}") from e
        if isinstance(rec, dict) and isinstance(rec.get("id"), str):
            jobs[rec["id"]] = {"room_id": rec.get("room_id"), "attempts": rec.get("attempts"),
                               "successes": rec.get("successes"), **{k: rec.get(k) for k in METER_KEYS}}
    for p in sorted((out / "jobs").glob("*/status.json")):
        s = _load(p, "job status")
        u = s.get("usage")
        if isinstance(u, dict) and isinstance(s.get("id"), str) and s.get("finished_at") is not None:
            jobs[s["id"]] = {"room_id": s.get("room_id"), "attempts": s.get("attempts"),
                             "successes": s.get("successes"), **{k: u.get(k) for k in METER_KEYS}}
    return jobs


def collect_rows(out_dir: str | Path) -> list[dict]:
    """One row per (source, room) and per family where the file has one. Sources overlap: never add across them."""
    out = Path(out_dir)
    rows = []
    for p in sorted((out / "reports").glob("*.json")):
        r = _load(p, "report")
        room = r.get("room_id", p.stem)
        rows.append(_row("report", room, None, r.get("attempts"), r.get("successes"), r.get("seconds"),
                         r.get("human_contacts")))
        fams = r.get("families")
        if isinstance(fams, dict):  # sim reports: per-family counts, with optional per-family seconds
            for fam, f in fams.items():
                rows.append(_row("report", room, fam, f.get("attempts"), f.get("successes"), f.get("seconds"), f.get("human_contacts")))
    for p in sorted(list((out / "eval").glob("*.json")) + list((out / "eval").glob("*/report.json"))):
        r = _load(p, "report")
        if not isinstance(r.get("totals"), dict) or not isinstance(r.get("families"), dict):
            continue
        room = r.get("room_id", p.stem)
        start = len(rows)
        for fam, f in (r.get("families") or {}).items():
            n = f.get("episodes")
            wall = f.get("mean_wall_seconds")
            rows.append(_row("eval", room, fam, n, f.get("successes"),
                             n * wall if _num(wall) and isinstance(n, int) else None, f.get("human_contacts")))
        t = r.get("totals") or {}
        rows.append(_row("eval", room, None, t.get("episodes"), t.get("successes"), t.get("wall_seconds"),
                         t.get("human_contacts")))
        for row in rows[start:]:
            row.update(run_id=r.get("run_id"), checkpoint_id=r.get("checkpoint_id"))
    metered = job_usage(out)
    for p in sorted((out / "jobs").glob("*/status.json")):
        s = _load(p, "job status")
        a, f_ = s.get("started_at"), s.get("finished_at")
        u = metered.get(s.get("id"), {})
        secs = u.get("wall_seconds") if _num(u.get("wall_seconds")) else (f_ - a if _num(a) and _num(f_) else None)
        if s.get("attempts") is None or secs is None:
            continue  # a job that has not finished has no time to report
        rows.append(_row("job", s.get("room_id") or p.parent.name, None, s["attempts"], s.get("successes"), secs,
                         frames=u.get("frames")))
    for jid, u in metered.items():  # jobs deleted by `arrival gc` keep their ledger line
        if not (out / "jobs" / jid / "status.json").is_file() and u["attempts"] is not None:
            rows.append(_row("job", u.get("room_id") or jid, None, u["attempts"], u.get("successes"),
                             u.get("wall_seconds"), frames=u.get("frames")))
    for r in rows:
        a, k, sec = r["attempts"], r["successes"], r["seconds"]
        r["dropped"] = a - k if a is not None else None
        r["drop_rate"] = round((a - k) / a, 4) if a else None
        r["demos_per_hour"] = round(k / sec * 3600, 1) if sec and k is not None else None
        r["seconds_per_demo"] = round(sec / k, 3) if sec and k else None
        r["attempts_per_demo"] = round(a / k, 3) if a and k else None
        r["local_cpu_wall_seconds"] = sec  # wall time on this machine, no unit price attached
    return rows


def dollars(usage: dict | None, pricing: dict | None) -> dict:
    d = {"local_cpu": {"usd": 0, "measured": True}}
    for key in ("token_factory", "serverless_jobs", "cosmos", "object_storage", "endpoint"):
        d[key] = {"usd": None, "measured": False}
    for key, prod in PRODUCT_KEYS.items():
        if not usage or prod not in usage or not pricing or prod not in pricing:
            continue
        rate = pricing[prod]
        basis = str(rate.get("basis", "assumed, not invoiced"))
        invoiced = basis.strip().lower() == "invoiced"
        d[key] = {"usd": round(usage[prod] * rate["usd_per_unit"], 4), "measured": invoiced, "units": usage[prod],
                  "unit": rate.get("unit"), "usd_per_unit": rate["usd_per_unit"], "rate_basis": basis,
                  **({} if invoiced else {"estimate": True})}
    return d


def metering_totals(out_dir: str | Path) -> dict:
    """Sum of the worker's per-job metering. Each total is None when no job measured it (a sum over nothing is not 0)."""
    jobs = job_usage(out_dir)
    for job_id, usage in jobs.items():
        for key in METER_KEYS:
            value = usage.get(key)
            if value is None:
                continue
            valid = _num(value) and value >= 0 if key == "wall_seconds" else type(value) is int and value >= 0
            if not valid:
                raise CostError(f"job {job_id}: invalid metering value for {key}")
        attempted, succeeded = usage.get("episodes_attempted"), usage.get("episodes_succeeded")
        if attempted is not None and succeeded is not None and succeeded > attempted:
            raise CostError(f"job {job_id}: episodes_succeeded exceeds episodes_attempted")
    tot: dict = {"jobs": len(jobs)}
    for k in METER_KEYS:
        vals = [u[k] for u in jobs.values() if _num(u.get(k))]
        tot[k] = round(sum(vals), 3) if vals else None
    tot["note"] = "measured on the local machine by arrival work; not priced (local CPU is $0, no Nebius product was used)"
    return tot


def teleop_block(baseline: dict | None) -> dict | None:
    if baseline is None:
        return None
    n = baseline["n_success"]
    return {**baseline, "seconds_per_success": round(baseline["operator_seconds_total"] / n, 3),
            "usd_per_success": round(baseline["cost_basis_usd"] / n, 4)}


def compare(rows: list[dict], dol: dict, tele: dict | None, usage_scope: dict | None = None) -> dict:
    """Generated-vs-teleop. Only with a baseline, only for runs in the baseline's room, and always with both n."""
    if tele is None:
        return {"measured": False, "reason": "no teleop baseline"}
    reps = [r for r in rows if r["source"] == "report" and r["room"] == tele["room"]]
    fam = [r for r in reps if r["family"] == tele["task"]]
    if not fam and tele["task"] != "all_tasks":
        return {"measured": False, "reason": f"no generated reports for task {tele['task']!r} in room {tele['room']!r}"}
    gen = fam or [r for r in reps if r["family"] is None]
    n = sum(r["successes"] or 0 for r in gen)
    if n == 0:
        return {"measured": False, "reason": f"no generated successes for room {tele['room']!r}"}
    secs = [r["seconds"] for r in gen]
    gen_s = round(sum(secs) / n, 3) if all(s is not None for s in secs) else None
    priced = [v for k, v in dol.items() if v["usd"] is not None and k != "local_cpu"]
    scoped = usage_scope == {"room": tele["room"], "task": tele["task"]}
    gen_usd = round(sum(v["usd"] for v in priced) / n, 4) if scoped and priced else None
    return {"measured": True, "task": tele["task"], "room": tele["room"],
            "generated": {"n_success": n, "scope": "task" if fam else "room", "seconds_per_success": gen_s,
                          "usd_per_success": gen_usd, "usd_measured": bool(gen_usd is not None) and all(v["measured"] for v in priced)},
            "teleop": {"n_success": tele["n_success"], "seconds_per_success": tele["seconds_per_success"],
                       "usd_per_success": tele["usd_per_success"]},
            "seconds_ratio_teleop_over_generated": round(tele["seconds_per_success"] / gen_s, 2) if gen_s else None,
            "note": "generated seconds are local wall time of the generate call; teleop seconds are operator time"}


def build(out_dir, usage_path=None, pricing_path="configs/pricing.example.yaml", baseline_path=None) -> dict:
    out = Path(out_dir)
    repo = Path(__file__).resolve().parents[2]
    if baseline_path is None and (out / "teleop_baseline.json").exists():
        baseline_path = out / "teleop_baseline.json"
    if pricing_path and not Path(pricing_path).exists() and (repo / pricing_path).exists():
        pricing_path = repo / pricing_path
    tele = teleop_block(load_baseline(baseline_path)) if baseline_path else None
    usage = load_usage(usage_path) if usage_path else None
    pricing = load_pricing(pricing_path) if usage else None
    rows = collect_rows(out)
    dol = dollars(usage, pricing)
    return {"rows": rows, "dollars": dol, "teleop_baseline": tele,
            "comparison": compare(rows, dol, tele, usage.get("scope") if usage else None),
            "metering": metering_totals(out)}


def _f(v, fmt="{}"):
    return "-" if v is None else fmt.format(v)


def markdown(rep: dict) -> str:
    L = ["| source | room | family | attempts | successes | drop rate | seconds | demos/hour | s/demo | attempts/demo |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        L.append(f"| {r['source']}{':' + r['run_id'] if r.get('run_id') else ''} | {r['room']} | {r['family'] or 'all'} | {_f(r['attempts'])} | {_f(r['successes'])} | "
                 f"{_f(r['drop_rate'])} | {_f(r['seconds'], '{:.1f}')} | {_f(r['demos_per_hour'])} | "
                 f"{_f(r['seconds_per_demo'])} | {_f(r['attempts_per_demo'])} |")
    if not rep["rows"]:
        L.append("| no reports, eval files or job statuses found | | | | | | | | | |")
    L += ["", "Sources overlap (a sim report and its job describe the same run); do not add across them.", "",
          "| dollars | usd | measured | note |", "|---|---|---|---|"]
    for k, v in rep["dollars"].items():
        note = f"{v['units']} {v['unit']} at {v['usd_per_unit']} ({v['rate_basis']})" if "units" in v else (
            "local run" if k == "local_cpu" else "not measured")
        L.append(f"| {k} | {_f(v['usd'])} | {str(v['measured']).lower()} | {note} |")
    m = rep.get("metering") or {"jobs": 0}
    if m["jobs"]:
        L += ["", f"Worker metering over {m['jobs']} job(s): " + ", ".join(f"{k} {_f(m[k])}" for k in METER_KEYS) + "."]
    runs = {r["run_id"]: r.get("checkpoint_id") for r in rep["rows"] if r.get("run_id")}
    if runs:
        L += ["", "Evaluation runs (checkpoint labels supplied by the caller):"]
        L += [f"- {run_id}: {checkpoint}" for run_id, checkpoint in runs.items()]
    c = rep["comparison"]
    L.append("")
    if not c["measured"]:
        L.append(f"Generated vs teleop: comparison not measured ({c['reason']}).")
    else:
        g, t = c["generated"], c["teleop"]
        est = "" if g["usd_per_success"] is None or g["usd_measured"] else " (estimate)"
        L += [f"Generated vs teleop, {c['task']} in {c['room']}:", "",
              "| | n successful demos | seconds/demo | USD/demo |", "|---|---|---|---|",
              f"| generated ({g['scope']}) | {g['n_success']} | {_f(g['seconds_per_success'])} | "
              f"{_f(g['usd_per_success'])}{est} |",
              f"| teleop | {t['n_success']} | {t['seconds_per_success']} | {t['usd_per_success']} |", "",
              f"Seconds, teleop over generated: {_f(c['seconds_ratio_teleop_over_generated'])}x. {c['note']}."]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arrival.eval.cost", description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="out")
    ap.add_argument("--usage", help="usage.json: {product: units consumed}")
    ap.add_argument("--pricing", default="configs/pricing.example.yaml")
    ap.add_argument("--baseline", help="teleop_baseline.json (default: <out>/teleop_baseline.json if present)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON instead of a table")
    a = ap.parse_args(argv)
    baseline_path = a.baseline
    if baseline_path is None and not (Path(a.out) / "teleop_baseline.json").exists():
        repo = Path(__file__).resolve().parents[2]
        p3 = repo / "docs" / "results" / "p3-teleop-2026-10-21" / "teleop_baseline.json"
        if p3.exists():
            baseline_path = p3
    try:
        rep = build(a.out, a.usage, a.pricing, baseline_path)
    except CostError as e:
        print(f"error: {e}")
        return 2
    print(json.dumps(rep, indent=2) if a.json else markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
