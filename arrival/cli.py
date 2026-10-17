"""python -m arrival demo | validate | serve | sim | eval | work"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EXIT_OK, EXIT_INVALID, EXIT_USAGE = 0, 1, 2


def resolve_scans(scans_dir: str | Path = "scans") -> Path:
    """A relative --scans that does not exist in the cwd falls back to <repo root>/<scans_dir>."""
    p = Path(scans_dir)
    if p.is_absolute() or p.exists():
        return p
    repo = Path(__file__).resolve().parents[1] / p
    return repo if repo.exists() else p


def run_demo(room: str = "hero_kitchen", episodes: int = 4, videos: str = "all",
             out_dir: str | Path = "out", scans_dir: str | Path = "scans",
             from_roomplan: bool = False, seed: int = 0, log=print) -> dict:
    """scan -> twin -> tasks -> scripted retarget -> oracle -> sim_texture -> LeRobot v2 -> report."""
    from arrival.eval.report import build_report, write_report
    from arrival.export.lerobot import LeRobotWriter
    from arrival.generate.scripted import generate
    from arrival.render.sim_texture import room_center_x
    from arrival.scan import bundle as scan_bundle
    from arrival.scan import roomplan
    from arrival.tasks.propose import propose, write_tasks
    from arrival.twin.compile import compile_twin, write_twin

    out_dir, scans_dir = Path(out_dir), resolve_scans(scans_dir)
    if not (scans_dir / room).is_dir():
        raise SystemExit(f"room {room!r} not found under {scans_dir}")
    if from_roomplan:
        rp_path = scans_dir / room / "roomplan.json"
        if not rp_path.is_file():
            raise SystemExit(f"no RoomPlan export at {rp_path} (room {room!r}; --scans {scans_dir})")
        bundle = roomplan.load(rp_path, room)
        # RoomPlan captures no small objects: place catalog objects on a counter top (seeded), never borrowed.
        from arrival.twin.compile import place_default_objects
        bundle["objects"] = place_default_objects(compile_twin(bundle), seed=seed)
    else:
        bundle = scan_bundle.load_room(room, scans_dir)
    twin = compile_twin(bundle)
    twin_dir = out_dir / "twins" / room
    write_twin(twin, twin_dir)
    tasks = propose(twin)
    write_tasks(tasks, twin_dir)
    if not tasks["tasks"]:
        raise SystemExit(f"no tasks could be proposed for {room}")
    log(f"[arrival] {room}: {len(twin['fixtures'])} fixtures, {len(tasks['tasks'])} task variations")

    t0 = time.perf_counter()
    dataset_dir = out_dir / "datasets" / room
    writer = LeRobotWriter(dataset_dir, room, videos, room_center_x=room_center_x(twin))
    attempts = 0
    for att in generate(tasks["tasks"], twin, episodes, seed=seed):
        attempts += 1
        if att["success"]:  # oracle: drop misses
            ep = writer.add(att)
            if ep and ep % 100 == 0:
                log(f"[arrival]   {ep} episodes ({attempts} attempts)")
    info = writer.finish(attempts)
    seconds = time.perf_counter() - t0
    report = build_report(room, attempts, info["total_episodes"], seconds, sorted(writer.families))
    report_path = write_report(report, out_dir / "reports")
    return {"dataset": dataset_dir, "report": report_path, "info": info, "report_data": report,
            "twin_dir": twin_dir}


def cmd_demo(a) -> int:
    if a.episodes < 1:
        print("--episodes must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    videos = a.videos or "all"
    res = run_demo(a.room, a.episodes, videos, a.out, a.scans, from_roomplan=a.from_roomplan)
    r = res["report_data"]
    print(f"dataset: {res['dataset']}")
    print(f"report:  {res['report']}")
    print(f"{r['successes']} episodes / {r['attempts']} attempts ({r['dropped']} dropped) in "
          f"{r['seconds']:.1f}s, {res['info']['total_videos']} videos, families={r['tasks']}")
    print("NOTE: scripted placeholder motion; arrival_physically_valid=false. Do not fine-tune on it.")
    return EXIT_OK


def cmd_validate(a) -> int:
    from arrival.export.validate import validate

    errors = validate(a.dataset_dir)
    if errors:
        for e in errors:
            print(f"INVALID: {e}", file=sys.stderr)
        return EXIT_INVALID
    info = json.loads((Path(a.dataset_dir) / "meta" / "info.json").read_text())
    print(f"OK: {a.dataset_dir} — {info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"{info['total_tasks']} tasks, {info['total_chunks']} chunk(s), {info['total_videos']} videos, "
          f"video_policy={info['arrival_video_policy']}, "
          f"physically_valid={info['arrival_physically_valid']}")
    return EXIT_OK


def cmd_serve(a) -> int:
    from arrival.web.app import serve

    serve(port=a.port, out_dir=Path(a.out), scans_dir=resolve_scans(a.scans))
    return EXIT_OK


def _sim_args(a):
    """Shared by sim and eval: (families, twin) or an error message."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return "mujoco is not installed: pip install -e '.[sim]'"
    from arrival.sim.tasks import SIM_FAMILIES

    fams = list(SIM_FAMILIES) if a.families == "all" else [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in SIM_FAMILIES]
    if bad or a.episodes < 1:
        return f"unknown families {bad} (choose from {', '.join(SIM_FAMILIES)})" if bad else "--episodes must be >= 1"
    twin_path = Path(a.out) / "twins" / a.room / "twin.json"
    if twin_path.is_file():
        twin = json.loads(twin_path.read_text())
    else:
        from arrival.scan import bundle as scan_bundle
        from arrival.twin.compile import compile_twin

        twin = compile_twin(scan_bundle.load_room(a.room, Path(a.scans)))
    return fams, twin


def cmd_sim(a) -> int:
    """MuJoCo sim of the compiled twin -> physics-checked demos with real joint states."""
    args = _sim_args(a)
    if isinstance(args, str):
        print(args, file=sys.stderr)
        return EXIT_USAGE
    fams, twin = args
    from arrival.sim.collect import collect

    res = collect(twin, fams, a.episodes, a.out, people=a.people, policy_name=a.policy, res=a.res,
                  seed=a.seed, save_preview=a.save_preview,
                  planner_name=a.planner, split=a.split,
                  use_generated_instructions=(a.instructions == "nemotron"),
                  mobile_base=getattr(a, "mobile_base", False))
    r = res["report_data"]
    print(f"dataset: {res['dataset']}")
    print(f"report:  {res['report']}")
    print(f"{r['successes']} episodes / {r['attempts']} attempts ({r['dropped']} dropped, physics-checked, "
          f"{r['human_contacts']} human contacts) in {r['seconds']:.1f}s; per family: "
          + ", ".join(f"{k} {v['successes']}/{v['attempts']}" for k, v in r["families"].items()))
    if a.save_preview:
        print(f"preview: {Path(a.out) / 'previews'}")
    return EXIT_OK


def cmd_eval(a) -> int:
    """Closed-loop scoring of a policy in the twin; writes out/eval/<room>_<policy>.json, no dataset."""
    args = _sim_args(a)
    if isinstance(args, str):
        print(args, file=sys.stderr)
        return EXIT_USAGE
    fams, twin = args
    from arrival.eval.harness import evaluate, table
    from arrival.sim.policy import VLANotConfigured

    try:
        rep = evaluate(twin, fams, a.episodes, policy_name=a.policy, seed=a.seed, people=a.people,
                       holdout=a.holdout, res=a.res, out_dir=a.out,
                       mobile_base=getattr(a, "mobile_base", False))
    except VLANotConfigured as e:
        print(f"policy vla: {e}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as e:  # URLError is an OSError: the policy server did not answer
        if a.policy != "vla":
            raise
        print(f"policy vla: no answer from {os.environ.get('GR00T_POLICY_URL')}: {e}", file=sys.stderr)
        return EXIT_USAGE
    print(table(rep))
    print(f"eval: {rep['path']}")
    return EXIT_OK


def cmd_work(a) -> int:
    """Drain out/jobs: each queued upload -> twin -> sim dataset -> out/jobs/<id>/dataset.zip."""
    from arrival.jobs.worker import work
    from arrival.sim.tasks import SIM_FAMILIES

    fams = list(SIM_FAMILIES) if a.families == "all" else [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in SIM_FAMILIES]
    if bad or a.episodes < 1:
        print(f"unknown families {bad} (choose from {', '.join(SIM_FAMILIES)})" if bad else "--episodes must be >= 1",
              file=sys.stderr)
        return EXIT_USAGE
    n = work(Path(a.out), once=a.once, poll=a.poll, episodes=a.episodes, families=fams, people=a.people,
             res=a.res, seed=a.seed, mobile_base=getattr(a, "mobile_base", False))
    print(f"processed {n} job(s)")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arrival", description="Arrival: room scan -> GR00T LeRobot v2 dataset")
    sub = p.add_subparsers(dest="cmd")
    d = sub.add_parser("demo", help="compile the room and write the dataset")
    d.add_argument("--episodes", type=int, default=4)
    d.add_argument("--videos", choices=["all", "sparse", "none"], default="all",
                   help="all (default): one mp4 per episode, declared to GR00T; sparse/none: no video "
                        "feature, sparse writes human previews to <out>/datasets/<room>_previews/")
    d.add_argument("--room", default="hero_kitchen")
    d.add_argument("--from-roomplan", action="store_true", help="normalize scans/<room>/roomplan.json instead of scan.json")
    d.add_argument("--out", default="out")
    d.add_argument("--scans", default="scans")
    d.set_defaults(fn=cmd_demo)
    v = sub.add_parser("validate", help="structurally check a LeRobot v2 dataset")
    v.add_argument("dataset_dir")
    v.set_defaults(fn=cmd_validate)
    s = sub.add_parser("serve", help="local page on 127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--out", default="out")
    s.add_argument("--scans", default="scans")
    s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("sim", help="MuJoCo sim of the room: arm + drawers/doors + people -> LeRobot v2 demos")
    m.add_argument("--room", default="hero_kitchen")
    m.add_argument("--episodes", type=int, default=8)
    m.add_argument("--families", default="counter_to_sink,open_drawer",
                   help="all | comma list of counter_to_sink,counter_to_counter,open_drawer,open_door")
    m.add_argument("--people", type=int, default=2)
    m.add_argument("--policy", choices=["scripted", "vla"], default="scripted")
    m.add_argument("--planner", choices=["rule", "vlm"], default="rule",
                   help="Task planner: rule (deterministic) or vlm (Nemotron via Nebius Token Factory)")
    m.add_argument("--instructions", choices=["template", "nemotron"], default="template",
                   help="Instruction source: template (fixed wordings) or nemotron (varied wordings with train/eval split)")
    m.add_argument("--split", choices=["train", "eval"], default="train",
                   help="Dataset instruction split: train or held-out eval")
    m.add_argument("--res", type=int, default=128)
    m.add_argument("--out", default="out")
    m.add_argument("--scans", default="scans")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--save-preview", action="store_true")
    m.add_argument("--mobile-base", action="store_true", help="enable actuated mobile base for all sim tasks")
    m.set_defaults(fn=cmd_sim)
    e = sub.add_parser("eval", help="score a policy closed-loop in the sim: success, time, human contacts")
    e.add_argument("--room", default="hero_kitchen")
    e.add_argument("--policy", choices=["scripted", "vla"], default="scripted")
    e.add_argument("--families", default="counter_to_sink,open_drawer",
                   help="all | comma list of counter_to_sink,counter_to_counter,open_drawer,open_door")
    e.add_argument("--episodes", type=int, default=4, help="episodes per family")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--people", type=int, default=2)
    e.add_argument("--holdout", action="store_true", help="held-out wording and start offset (docs/07-eval.md)")
    e.add_argument("--res", type=int, default=128)
    e.add_argument("--out", default="out")
    e.add_argument("--scans", default="scans")
    e.add_argument("--mobile-base", action="store_true", help="enable actuated mobile base for all sim tasks")
    e.set_defaults(fn=cmd_eval)
    w = sub.add_parser("work", help="process queued uploads under <out>/jobs: twin -> sim dataset -> dataset.zip")
    w.add_argument("--once", action="store_true", help="process one job (or none) and exit")
    w.add_argument("--poll", type=float, default=2.0, help="seconds between queue checks")
    w.add_argument("--episodes", type=int, default=8)
    w.add_argument("--families", default="counter_to_sink,open_drawer",
                   help="all | comma list of counter_to_sink,counter_to_counter,open_drawer,open_door")
    w.add_argument("--people", type=int, default=1)
    w.add_argument("--res", type=int, default=128)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--out", default="out")
    w.add_argument("--mobile-base", action="store_true", help="enable actuated mobile base for all sim tasks")
    w.set_defaults(fn=cmd_work)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        a = parser.parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else EXIT_OK
    if not a.cmd:
        parser.print_usage(sys.stderr)
        return EXIT_USAGE
    return a.fn(a)
