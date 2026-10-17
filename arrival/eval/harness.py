"""Closed-loop eval in the twin: N episodes per family for one policy, scored without writing a dataset.

Success is the physics check from collect.py; an episode with robot-human contact counts as a failure.
Time is sim time (frames / 20 Hz) and wall time per episode. `holdout` draws only the held-out wording and
start offset of docs/07-eval.md (make_tasks(holdout=True)); `sim` datasets never contain them.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from arrival.sim.collect import FPS, SimEnv, finish_task, run_episode
from arrival.sim.tasks import make_tasks


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for k successes in n trials; [0, 1] when n is 0."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    zz = z * z / n
    centre = (p + zz / 2) / (1 + zz)
    half = z * math.sqrt(p * (1 - p) / n + zz / (4 * n)) / (1 + zz)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def evaluate(twin: dict, families: list[str], episodes: int, policy_name: str = "scripted", seed: int = 0,
             people: int = 2, holdout: bool = False, res: int = 128, out_dir: str | Path = "out",
             log=print) -> dict:
    from arrival.sim.policy import ScriptedExpert, VLAClient

    policy = ScriptedExpert() if policy_name == "scripted" else VLAClient()
    room = twin["room_id"]
    per: dict[str, dict] = {}
    t_all = time.perf_counter()
    for j, fam in enumerate(families):
        tasks = [finish_task(t) for t in make_tasks(twin, [fam], episodes, seed=seed + j, people=people,
                                                     holdout=holdout, log=log)]
        if not tasks:
            continue
        rows = []
        for task in tasks:
            t0 = time.perf_counter()
            env = SimEnv(twin, task, res=res)
            ep = run_episode(env, policy, task)
            wall = time.perf_counter() - t0
            s = ep["safety"]
            ok = ep["success"] and not s["human_contact"]
            rows.append((ok, len(ep["states"]) / FPS, wall, s["human_contact"], s["min_human_distance_m"],
                         ep["ik_error"]))
            tag = "DROP human_contact" if s["human_contact"] else ("OK " if ok else "MISS")
            log(f"[eval] {task['task_id']}: {tag} {ep['detail']} sim={len(ep['states']) / FPS:.2f}s "
                f"wall={wall:.1f}s min_human_d={s['min_human_distance_m']}")
        n, k = len(rows), sum(r[0] for r in rows)
        dists = [r[4] for r in rows if r[4] is not None]
        iks = [r[5] for r in rows if r[5] is not None]
        per[fam] = {"episodes": n, "successes": k, "success_rate": round(k / n, 4), "wilson_95": wilson(k, n),
                    "mean_seconds": round(sum(r[1] for r in rows) / n, 3),
                    "mean_wall_seconds": round(sum(r[2] for r in rows) / n, 3),
                    "human_contacts": sum(r[3] for r in rows),
                    "min_human_distance_m": round(min(dists), 4) if dists else None,
                    "ik_error_max": round(max(iks), 4) if iks else None}
    n = sum(f["episodes"] for f in per.values())
    k = sum(f["successes"] for f in per.values())
    report = {"room_id": room, "policy": policy_name, "episodes_per_family": episodes, "seed": seed,
              "people": people, "holdout": holdout, "resolution": res, "families": per,
              "totals": {"episodes": n, "successes": k, "success_rate": round(k / n, 4) if n else None,
                         "wilson_95": wilson(k, n), "human_contacts": sum(f["human_contacts"] for f in per.values()),
                         "wall_seconds": round(time.perf_counter() - t_all, 2)}}
    path = Path(out_dir) / "eval" / f"{room}_{policy_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    report["path"] = path
    return report


def table(report: dict) -> str:
    head = f"{'family':<20}{'ep':>4}{'succ':>6}{'rate':>7}  {'wilson95':<16}{'sim_s':>7}{'wall_s':>8}{'contact':>9}{'min_d_m':>9}{'ik_err':>8}"
    lines = [head]
    for fam, f in report["families"].items():
        lo, hi = f["wilson_95"]
        md = "-" if f["min_human_distance_m"] is None else f"{f['min_human_distance_m']:.3f}"
        ik = "-" if f["ik_error_max"] is None else f"{f['ik_error_max']:.3f}"
        lines.append(f"{fam:<20}{f['episodes']:>4}{f['successes']:>6}{f['success_rate']:>7.3f}  [{lo:.3f}, {hi:.3f}]  "
                     f"{f['mean_seconds']:>7.2f}{f['mean_wall_seconds']:>8.1f}{f['human_contacts']:>9}{md:>9}{ik:>8}")
    t = report["totals"]
    lo, hi = t["wilson_95"]
    rate = "-" if t["success_rate"] is None else f"{t['success_rate']:.3f}"
    lines.append(f"{'total':<20}{t['episodes']:>4}{t['successes']:>6}{rate:>7}  [{lo:.3f}, {hi:.3f}]"
                 f"{'':>24}{t['human_contacts']:>9}")
    return "\n".join(lines)
