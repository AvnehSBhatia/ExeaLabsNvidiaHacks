"""python -m arrival demo | validate | serve | sim | merge-shards | subset | eval | realism | work | gc | doctor | summary | export-site"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from arrival.config import get as get_setting
from arrival.config import redact

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
    from arrival.export.provenance import run_record
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
    if from_roomplan and twin["objects"]:
        from arrival.export.provenance import note_seed

        note_seed(twin, "default_objects", seed)
    twin_dir = out_dir / "twins" / room
    write_twin(twin, twin_dir)
    tasks = propose(twin)
    write_tasks(tasks, twin_dir)
    if not tasks["tasks"]:
        raise SystemExit(f"no tasks could be proposed for {room}")
    log(f"[arrival] {room}: {len(twin['fixtures'])} fixtures, {len(tasks['tasks'])} task variations")

    t0 = time.perf_counter()
    dataset_dir = out_dir / "datasets" / room
    writer = LeRobotWriter(dataset_dir, room, videos, room_center_x=room_center_x(twin), extra_info={
        "arrival_provenance": run_record(twin, seed=seed, generator="scripted_retarget", episodes=episodes,
                                         videos=videos)})
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

    per_episode = a.episodes or a.frames or bool(a.drop)
    errors = validate(a.dataset_dir)
    if errors:  # with a per-episode pass requested, a bad episode is what we are here to find: keep going
        for e in errors:
            print(f"INVALID: {e}", file=sys.stderr)
        if not per_episode:
            return EXIT_INVALID
    else:
        info = json.loads((Path(a.dataset_dir) / "meta" / "info.json").read_text())
        print(f"OK: {a.dataset_dir}: {info['total_episodes']} episodes, {info['total_frames']} frames, "
              f"{info['total_tasks']} tasks, {info['total_chunks']} chunk(s), {info['total_videos']} videos, "
              f"video_policy={info['arrival_video_policy']}, "
              f"physically_valid={info['arrival_physically_valid']}")
        if not per_episode:
            return EXIT_OK
    from arrival.export.quality import QualityError, check_episodes, drop_flagged, format_report

    try:
        report = check_episodes(a.dataset_dir, frames=a.frames)
        if a.drop:
            report = drop_flagged(a.dataset_dir, a.drop, report)
    except QualityError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return EXIT_INVALID
    print(format_report(report))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {a.json}")
    if a.drop:
        c = report["cleaned"]
        left = validate(a.drop)  # the cleaned copy must pass the structural check, or it is not a fix
        for e in left:
            print(f"INVALID (cleaned dataset): {e}", file=sys.stderr)
        print(f"cleaned dataset: {c['output']} ({c['kept']} kept, {len(c['dropped'])} dropped)")
        return EXIT_INVALID if left else EXIT_OK
    return EXIT_INVALID if (errors or report["flagged"]) else EXIT_OK


def cmd_serve(a) -> int:
    from arrival.web.app import serve

    if a.max_upload_mb <= 0 or a.rate_limit < 0 or a.max_queued < 0:
        print("--max-upload-mb must be > 0; --rate-limit and --max-queued must be >= 0 (0 = off)", file=sys.stderr)
        return EXIT_USAGE
    serve(port=a.port, out_dir=Path(a.out), scans_dir=resolve_scans(a.scans), host=a.host,
          max_upload_mb=a.max_upload_mb, rate_limit=a.rate_limit, max_queued=a.max_queued,
          public=a.public, behind_proxy=a.behind_proxy)
    return EXIT_OK


def cmd_doctor(a) -> int:
    """Offline environment check: python, dependencies, MuJoCo, ffmpeg, which keys are set (never their values), disk."""
    from arrival import doctor

    checks = doctor.run_checks(a.out)
    print(doctor.render_json(checks) if a.json else doctor.render(checks))
    return EXIT_INVALID if doctor.exit_code(checks) else EXIT_OK


def _load_twin(room: str, out: str | Path, scans: str | Path) -> dict:
    """The compiled twin under <out>/twins, unless scans/<room>/scan.json was saved after it was written;
    then the scan is compiled again. Raises FileNotFoundError when neither exists."""
    from arrival.scan import bundle as scan_bundle
    from arrival.twin.compile import compile_twin

    twin_path = Path(out) / "twins" / room / "twin.json"
    scan_path = resolve_scans(scans) / room / "scan.json"
    if twin_path.is_file() and not (scan_path.is_file() and scan_path.stat().st_mtime > twin_path.stat().st_mtime):
        return json.loads(twin_path.read_text())
    return compile_twin(scan_bundle.load_room(room, resolve_scans(scans)))


def _sim_args(a):
    """Shared by sim and eval: (families, twin) or an error message."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return "mujoco is not installed: pip install -e '.[sim]'"
    from arrival.sim.tasks import SIM_FAMILIES

    if getattr(a, "arm", "procedural") == "menagerie":
        from arrival.sim import menagerie

        if not menagerie.available():
            return menagerie.FETCH_HINT

    fams = list(SIM_FAMILIES) if a.families == "all" else [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in SIM_FAMILIES]
    if bad or a.episodes < 1:
        return f"unknown families {bad} (choose from {', '.join(SIM_FAMILIES)})" if bad else "--episodes must be >= 1"
    if getattr(a, "base", "fixed") == "mobile" and getattr(a, "policy", "scripted") != "scripted":
        return ("--base mobile needs --policy scripted: the policy server wire format carries no base pose yet "
                "(8-float arm only), so a served policy could not drive the base")
    try:
        twin = _load_twin(a.room, a.out, a.scans)
    except FileNotFoundError:
        return f"no scan for room {a.room!r} under {resolve_scans(a.scans)} and no twin in {a.out}/twins"
    return fams, twin


def cmd_sim(a) -> int:
    """MuJoCo sim of the compiled twin -> physics-checked demos with real joint states."""
    args = _sim_args(a)
    if isinstance(args, str):
        print(args, file=sys.stderr)
        return EXIT_USAGE
    fams, twin = args
    from arrival.jobs.shards import ShardError, parse_shard

    try:
        shard = parse_shard(a.shard) if getattr(a, "shard", None) else None
    except ShardError as e:
        print(e, file=sys.stderr)
        return EXIT_USAGE
    if getattr(a, "tavily", False):
        from arrival.tasks.tavily_research import TavilyResearchAgent
        agent = TavilyResearchAgent()
        twin = agent.enrich_twin(twin)
    from arrival.sim.collect import collect
    from arrival.sim.policy import VLANotConfigured

    try:
        res = collect(twin, fams, a.episodes, a.out, people=a.people, policy_name=a.policy, res=a.res,
                      seed=a.seed, save_preview=a.save_preview,
                      planner_name=a.planner, split=a.split,
                      use_generated_instructions=(a.instructions == "nemotron"), base=getattr(a, "base", "fixed"),
                      shard=shard, resume=getattr(a, "resume", False), arm=getattr(a, "arm", "procedural"),
                      exec_noise=getattr(a, "dart", 0.0),
                      exocentric=getattr(a, "exocentric", False))
    except ShardError as e:
        print(e, file=sys.stderr)
        return EXIT_USAGE
    except VLANotConfigured as e:
        print(f"policy vla: {e}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as e:  # URLError is an OSError: the policy server did not answer
        if a.policy != "vla":
            raise
        print(redact(f"policy vla: no answer from {get_setting('GR00T_POLICY_URL')}: {e}"), file=sys.stderr)
        return EXIT_USAGE
    r = res["report_data"]
    print(f"dataset: {res['dataset']}")
    print(f"report:  {res['report']}")
    if shard:
        s = r["shard"]
        print(f"shard {s['index']}/{s['count']}: tasks {s['task_range'][0]}..{s['task_range'][1] - 1} of {s['total_tasks']}; "
              f"join the shards with: python -m arrival merge-shards {Path(a.out) / 'shards'} --out {a.out}")
    print(f"{r['successes']} episodes / {r['attempts']} attempts ({r['dropped']} dropped, physics-checked, "
          f"{r['human_contacts']} human contacts) in {r['seconds']:.1f}s; per family: "
          + ", ".join(f"{k} {v['successes']}/{v['attempts']}" for k, v in r["families"].items()))
    if r["skipped"]:
        print(f"skipped {r['skipped']} ({r['skipped_unreachable']} unreachable in the pre-flight, "
              f"{r['skipped_error']} errored; none written):", file=sys.stderr)
        for x in r["skip_reasons"][:5]:
            print(f"  {x.get('task_id') or x['family']}: {x['reason']}", file=sys.stderr)
    if a.save_preview:
        print(f"preview: {Path(a.out) / 'previews'}")
    return EXIT_OK


def cmd_fanout(a) -> int:
    """Run the sim shards resumably on the local fake backend (a directory as object storage) or, with
    `--backend nebius`, as Nebius Serverless AI Jobs with an Object Storage bucket, and merge them."""
    from arrival.export.validate import validate
    from arrival.jobs.fanout import FanoutConfig, LocalBackend, run_fanout
    from arrival.jobs.fanout_status import StatusRecorder
    from arrival.jobs.shards import ShardError

    cfg = FanoutConfig(a.run_id, a.shards, a.room, a.families, a.episodes, a.people, a.res, a.seed, a.scans)
    if a.backend == "nebius":
        from arrival.jobs.nebius import NebiusBackend, NebiusNotConfigured

        try:
            be = NebiusBackend()
        except NebiusNotConfigured as e:
            print(f"fanout: {e}", file=sys.stderr)
            return EXIT_USAGE
    else:
        be = LocalBackend(Path(a.out) / "fanout_store", Path(a.out) / "fanout_scratch")
    rec = StatusRecorder(be)
    try:
        res = run_fanout(rec, cfg, a.out)
    except ShardError as e:
        print(f"fanout: {e}", file=sys.stderr)
        return EXIT_USAGE
    finally:
        if a.backend == "nebius":  # remote jobs outlive this process unless cancelled
            left = rec.cancel_open()
            if left:
                print(f"fanout: could not cancel Nebius jobs {left}; cancel them by hand", file=sys.stderr)
    problems = validate(res["dataset"])
    print(f"dataset: {res['dataset']} (ran shards {res['ran']}, skipped {res['skipped']})")
    return EXIT_OK if not problems else EXIT_USAGE


def cmd_subset(a) -> int:
    """Keep the episodes of a sim dataset that pass a rule (family, drawer opening) as a new validated dataset."""
    from arrival.export.subset import SubsetError, subset_dataset
    from arrival.export.validate import validate

    try:
        res = subset_dataset(a.src, a.out, families=[f for f in a.families.split(",") if f] if a.families else None,
                             min_drawer_opening=a.min_drawer_opening)
    except SubsetError as e:
        print(f"subset: {e}", file=sys.stderr)
        return EXIT_USAGE
    errors = validate(res["dataset"])
    for e in errors:
        print(f"INVALID: {e}", file=sys.stderr)
    print(f"dataset: {res['dataset']}")
    return EXIT_INVALID if errors else EXIT_OK


def cmd_merge_shards(a) -> int:
    """Join the datasets `sim --shard i/n` wrote into one LeRobot v2 dataset and validate it."""
    from arrival.export.validate import validate
    from arrival.jobs.shards import ShardError, merge_shards

    try:
        res = merge_shards(a.shards, a.out)
    except ShardError as e:
        print(f"merge-shards: {e}", file=sys.stderr)
        return EXIT_USAGE
    errors = validate(res["dataset"])
    for e in errors:
        print(f"INVALID: {e}", file=sys.stderr)
    info = res["info"]
    print(f"dataset: {res['dataset']}")
    if res["report"]:
        print(f"report:  {res['report']}")
    if errors:
        return EXIT_INVALID
    print(f"OK: merged {info['arrival_merged_shards']} shards: {info['total_episodes']} episodes, "
          f"{info['total_frames']} frames, {info['total_tasks']} tasks, physically_valid={info['arrival_physically_valid']}")
    return EXIT_OK


def cmd_research(a) -> int:
    """Tavily web research agent for appliance specifications and safety guidelines."""
    from arrival.tasks.tavily_research import TavilyResearchAgent

    agent = TavilyResearchAgent()
    out_dir = Path(a.out) / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.query:
        print(f"Researching query via Tavily: {a.query!r}")
        res = agent.search(a.query)
        print(f"Source: {agent.last_source}")
        if res.get("answer"):
            print(f"Summary: {res['answer']}")
        for r in res.get("results", [])[:3]:
            print(f" - [{r.get('title')}] {r.get('url')}")
        target = out_dir / "query_research.json"
        target.write_text(json.dumps(res, indent=2))
        print(f"Saved: {target}")
        return EXIT_OK

    try:
        twin = _load_twin(a.room, a.out, a.scans)
    except FileNotFoundError:
        print(f"no scan for room {a.room!r} under {resolve_scans(a.scans)} and no twin in {a.out}/twins",
              file=sys.stderr)
        return EXIT_USAGE

    print(f"Researching appliances for room '{a.room}' via Tavily...")
    enriched = agent.enrich_twin(twin)
    research = enriched.get("tavily_research", {})
    print(f"Discovered and researched {len(research)} fixture types (source={agent.last_source}):")
    for ftype, info in research.items():
        print(f"\n[{ftype.upper()}]")
        print(f"  Summary: {info['summary']}")
        print(f"  Safety: {'; '.join(info['safety_guidelines'][:2])}")
        if info.get("sources"):
            print(f"  Source: {info['sources'][0]}")

    target = out_dir / f"{a.room}_research.json"
    target.write_text(json.dumps(research, indent=2))
    print(f"\nSaved research report: {target}")
    return EXIT_OK


def cmd_eval(a) -> int:
    """Score fresh or frozen trials and preserve each run's inputs and outcomes."""
    from arrival.eval.harness import evaluate, table
    from arrival.eval.manifest import load_manifest
    from arrival.sim.policy import VLANotConfigured

    defaults = dict(room="hero_kitchen", families="counter_to_sink,open_drawer,move_on_table", episodes=4, seed=0,
                    people=2, holdout=False, res=128, scans="scans")
    try:
        if getattr(a, "manifest", None):
            conflicts = ["--" + key for key in defaults if getattr(a, key, None) is not None]
            if conflicts:
                raise ValueError("--manifest cannot be combined with " + ", ".join(conflicts))
            frozen = load_manifest(a.manifest)
            rep = evaluate(manifest=frozen, policy_name=a.policy, checkpoint_id=getattr(a, "checkpoint_id", None), out_dir=a.out)
        else:
            for key, value in defaults.items():
                if getattr(a, key, None) is None:
                    setattr(a, key, value)
            args = _sim_args(a)
            if isinstance(args, str):
                raise ValueError(args)
            fams, twin = args
            rep = evaluate(twin, fams, a.episodes, policy_name=a.policy, seed=a.seed, people=a.people,
                           holdout=a.holdout, res=a.res, out_dir=a.out, base=getattr(a, "base", "fixed"),
                           arm=getattr(a, "arm", "procedural"),
                           checkpoint_id=getattr(a, "checkpoint_id", None))
    except VLANotConfigured as e:
        print(f"policy vla: {e}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as e:
        print(f"eval: {e}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as e:
        prefix = (f"policy vla: no answer from {get_setting('GR00T_POLICY_URL')}"
                  if a.policy == "vla" and not isinstance(e, FileNotFoundError) else "eval")
        print(redact(f"{prefix}: {e}"), file=sys.stderr)
        return EXIT_USAGE
    print(table(rep))
    print(f"eval: {rep['path']}\nmanifest: {rep['manifest_path']}")
    return EXIT_OK


def cmd_realism(a) -> int:
    """Score the twin's render against a scan's held-out RGB-D frames; writes out/eval/<room>_realism.json and .png."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("mujoco is not installed: pip install -e '.[sim]'", file=sys.stderr)
        return EXIT_USAGE
    from arrival.eval import realism_cli

    try:
        rep = realism_cli.run_realism(a.scan, a.out, resolve_scans(a.scans), twin_path=a.twin, max_side=a.max_side,
                                      max_frames=a.max_frames, png_frames=a.png_frames,
                                      min_confidence=a.min_confidence)
    except realism_cli.RealismError as e:
        print(f"realism: {e}", file=sys.stderr)
        return EXIT_INVALID
    print(realism_cli.summary(rep))
    print(f"json: {rep['path']}")
    if rep["png"]:
        print(f"png:  {rep['png']}")
    return EXIT_OK if rep["valid"] else EXIT_INVALID


def cmd_work(a) -> int:
    """Drain out/jobs: each queued upload -> twin -> sim dataset -> out/jobs/<id>/dataset.zip."""
    from arrival.jobs.limits import LimitError, Limits
    from arrival.jobs.worker import work
    from arrival.sim.tasks import SIM_FAMILIES

    fams = list(SIM_FAMILIES) if a.families == "all" else [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in SIM_FAMILIES]
    if bad or a.episodes < 1:
        print(f"unknown families {bad} (choose from {', '.join(SIM_FAMILIES)})" if bad else "--episodes must be >= 1",
              file=sys.stderr)
        return EXIT_USAGE
    if a.retries is not None and a.retries < 0:
        print("--retries must be >= 0", file=sys.stderr)
        return EXIT_USAGE
    try:  # ARRIVAL_* variables first, then the flags
        limits = Limits.from_env().replace(max_episodes=a.max_episodes, max_people=a.max_people,
                                           max_job_seconds=a.max_job_seconds, retention_hours=a.retention_hours,
                                           job_retries=a.retries)
        n = work(Path(a.out), once=a.once, poll=a.poll, limits=limits, episodes=a.episodes, families=fams,
                 people=a.people, res=a.res, seed=a.seed)
    except LimitError as e:
        print(f"limit: {e}", file=sys.stderr)
        return EXIT_USAGE
    print(f"processed {n} job(s)")
    return EXIT_OK


def cmd_gc(a) -> int:
    """Delete finished jobs (upload, twin, dataset, zip) older than --max-age-hours."""
    from arrival.jobs.retention import gc

    if a.max_age_hours <= 0:
        print("--max-age-hours must be > 0", file=sys.stderr)
        return EXIT_USAGE
    res = gc(Path(a.out), a.max_age_hours, dry_run=a.dry_run)
    verb = "would remove" if a.dry_run else "removed"
    print(f"{verb} {len(res['removed'])} job(s), {res['bytes']} bytes" + (": " + ", ".join(res["removed"]) if res["removed"] else ""))
    return EXIT_OK


def cmd_summary(a) -> int:
    """Tiny pipeline run on several rooms -> one table of compile / sim / validate results."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("mujoco is not installed: pip install -e '.[sim]'", file=sys.stderr)
        return EXIT_USAGE
    from arrival.eval.summary import summarize, to_markdown, write_summary
    from arrival.sim.tasks import SIM_FAMILIES

    fams = list(SIM_FAMILIES) if a.families == "all" else [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in SIM_FAMILIES]
    if bad or a.episodes < 1:
        print(f"unknown families {bad} (choose from {', '.join(SIM_FAMILIES)})" if bad else "--episodes must be >= 1",
              file=sys.stderr)
        return EXIT_USAGE
    rooms = [r.strip() for r in a.rooms.split(",") if r.strip()] if a.rooms else None
    kw = {"rooms": rooms} if rooms else {}
    summary = summarize(out_dir=a.out, scans_dir=resolve_scans(a.scans), episodes=a.episodes, res=a.res,
                        people=a.people, families=tuple(fams), seed=a.seed, **kw)
    jp, mp = write_summary(summary, a.out)
    print(to_markdown(summary))
    print(f"summary: {jp}\nsummary: {mp}")
    return EXIT_OK


def cmd_export_site(a) -> int:
    """Pre-render the local page to static HTML + JSON (no server needed, GitHub Pages friendly)."""
    from arrival.web.static_export import ExportError, export_site

    try:
        res = export_site(Path(a.out), resolve_scans(a.scans), Path(a.dest), zips=a.zips)
    except (ExportError, OSError) as e:
        print(f"export-site: {e}", file=sys.stderr)
        return EXIT_INVALID
    print(f"static site: {res['dest']} ({res['pages']} pages, {res['previews']} previews, "
          f"{len(res['zips'])} zips, rooms: {', '.join(res['rooms']) or 'none'})")
    print(f"open {Path(res['dest']) / 'index.html'} in a browser, or serve the folder with any static host")
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
    v.add_argument("--episodes", action="store_true",
                   help="also check each episode (NaN, stuck arm, zero motion, joint at limit, length outliers); "
                        "exit 1 if any is flagged")
    v.add_argument("--frames", action="store_true", help="with --episodes: also decode videos and flag blank cameras")
    v.add_argument("--drop", metavar="CLEAN_DIR", default=None,
                   help="write a cleaned copy without the flagged episodes to CLEAN_DIR (implies --episodes)")
    v.add_argument("--json", metavar="FILE", default=None, help="write the per-episode report as JSON")
    v.set_defaults(fn=cmd_validate)
    s = sub.add_parser("serve", help="local page on 127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to accept uploads from the iOS capture app on the same Wi-Fi")
    s.add_argument("--out", default="out")
    s.add_argument("--scans", default="scans")
    s.add_argument("--max-upload-mb", type=float, default=200,
                   help="reject a larger upload body with 413 (default 200)")
    s.add_argument("--rate-limit", type=int, default=10,
                   help="uploads per minute per client before 429; 0 turns it off (default 10)")
    s.add_argument("--max-queued", type=int, default=50,
                   help="refuse uploads with 503 while this many jobs are queued or running; 0 = no cap (default 50)")
    s.add_argument("--public", action="store_true",
                   help="public hosting: the page lists only the visitor's own jobs and the API hides server paths")
    s.add_argument("--behind-proxy", action="store_true",
                   help="trust X-Forwarded-Proto (HSTS, Secure cookie) and X-Forwarded-For (rate limit) from a reverse proxy")
    s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("sim", help="MuJoCo sim of the room: arm + drawers/doors + people -> LeRobot v2 demos")
    m.add_argument("--room", default="hero_kitchen")
    m.add_argument("--episodes", type=int, default=8)
    m.add_argument("--families", default="counter_to_sink,open_drawer,move_on_table",
                   help="all | comma list of counter_to_sink,counter_to_counter,open_drawer,open_door,move_on_table (a family with no variation in this room is skipped)")
    m.add_argument("--people", type=int, default=2)
    m.add_argument("--policy", choices=["scripted", "vla"], default="scripted")
    m.add_argument("--planner", choices=["rule", "vlm"], default="rule",
                   help="Task planner: rule (deterministic) or vlm (Nemotron via Nebius Token Factory)")
    m.add_argument("--instructions", choices=["template", "nemotron"], default="template",
                   help="Instruction source: template (fixed wordings) or nemotron (varied wordings with train/eval split)")
    m.add_argument("--split", choices=["train", "eval"], default="train",
                   help="Dataset instruction split: train or held-out eval")
    m.add_argument("--base", choices=["fixed", "mobile"], default="fixed",
                   help="mobile: the pedestal drives (MAN-3); counter_to_counter becomes reachable")
    m.add_argument("--arm", choices=["procedural", "menagerie"], default="procedural",
                   help="menagerie = the real Franka asset from MuJoCo Menagerie (python scripts/fetch_menagerie.py; meshes are not committed)")
    m.add_argument("--res", type=int, default=128)
    m.add_argument("--out", default="out")
    m.add_argument("--scans", default="scans")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--save-preview", action="store_true")
    m.add_argument("--exocentric", action="store_true",
                   help="also export the synchronized external overview camera; --save-preview adds a paired ego/exo video")
    m.add_argument("--shard", default=None, metavar="I/N",
                   help="run block I of N of the --episodes tasks (I counts from 0), into <out>/shards/<I>of<N>/; "
                        "join with merge-shards")
    m.add_argument("--resume", action="store_true",
                   help="continue a killed run from its progress journal instead of starting over (same settings)")
    m.add_argument("--dart", type=float, default=0.0, metavar="RAD",
                   help="DART: drive the arm with smooth noise, typical std RAD (each episode uses 0.4x to 1.6x), on the joint targets while the dataset "
                        "keeps the clean targets, so a policy sees off-plan states labelled with the way back; "
                        "0 (default) is off, 0.02 to 0.05 is a useful range. Recorded in the dataset provenance")
    m.add_argument("--tavily", action="store_true", help="Enrich digital twin and VLM planning with Tavily appliance research")
    m.set_defaults(fn=cmd_sim)
    f = sub.add_parser("fanout", help="run sim shards on a job backend, resumable, then merge (local by default)")
    f.add_argument("--backend", choices=["local", "nebius"], default="local",
                   help="local subprocesses and a directory, or Nebius Serverless AI Jobs and Object Storage "
                        "(NEBIUS_* settings; tested against fakes only)")
    f.add_argument("--run-id", default="run")
    f.add_argument("--shards", type=int, default=2)
    f.add_argument("--room", default="hero_kitchen")
    f.add_argument("--families", default="counter_to_sink,open_drawer")
    f.add_argument("--episodes", type=int, default=8)
    f.add_argument("--people", type=int, default=0)
    f.add_argument("--res", type=int, default=128)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--scans", default="scans")
    f.add_argument("--out", default="out")
    f.set_defaults(fn=cmd_fanout)
    g = sub.add_parser("merge-shards", help="join sim --shard datasets into one validated LeRobot v2 dataset")
    g.add_argument("shards", nargs="+", help="shard folders (<out>/shards/<i>of<n>), their dataset folders, "
                                             "or the one folder that holds them")
    g.add_argument("--out", default="out", help="the merged dataset goes to <out>/datasets/<room>_sim")
    g.set_defaults(fn=cmd_merge_shards)
    u = sub.add_parser("subset", help="keep the episodes of a sim dataset that pass a rule (family, drawer opening)")
    u.add_argument("src", help="a sim dataset folder (or the folder that holds datasets/<room>_sim)")
    u.add_argument("--out", default="out", help="the subset goes to <out>/datasets/<room>_sim")
    u.add_argument("--families", default="", help="comma list of families to keep (default: all)")
    u.add_argument("--min-drawer-opening", type=float, default=None, metavar="M",
                   help="drop open_drawer episodes whose final drawer opening is under M metres (the clean expert ends at 0.300)")
    u.set_defaults(fn=cmd_subset)
    r = sub.add_parser("research", help="Tavily AI research on kitchen appliances and safety guidelines")
    r.add_argument("--room", default="hero_kitchen")
    r.add_argument("--query", default=None, help="direct query string to search via Tavily")
    r.add_argument("--out", default="out")
    r.add_argument("--scans", default="scans")
    r.set_defaults(fn=cmd_research)
    e = sub.add_parser("eval", help="score a policy on fresh or frozen trials; preserve every run")
    e.add_argument("--room")
    e.add_argument("--policy", choices=["scripted", "vla"], default="scripted")
    e.add_argument("--families", help="all | comma-separated task families (default: counter_to_sink,open_drawer,move_on_table; a family with no variation in this room is skipped)")
    e.add_argument("--episodes", type=int, help="episodes per family (default: 4)")
    e.add_argument("--seed", type=int)
    e.add_argument("--people", type=int)
    e.add_argument("--holdout", action="store_true", default=None, help="held-out wording and start offset")
    e.add_argument("--base", choices=["fixed", "mobile"], default="fixed", help="mobile: the pedestal drives (MAN-3)")
    e.add_argument("--arm", choices=["procedural", "menagerie"], default="procedural",
                   help="menagerie = the real Franka asset from MuJoCo Menagerie (python scripts/fetch_menagerie.py; meshes are not committed)")
    e.add_argument("--res", type=int)
    e.add_argument("--out", default="out")
    e.add_argument("--scans")
    e.add_argument("--manifest", help="replay a previous run's manifest.json without resampling")
    e.add_argument("--checkpoint-id", help="caller-declared checkpoint label; required for VLA")
    e.set_defaults(fn=cmd_eval)
    rl = sub.add_parser("realism", help="score the twin's render against a scan's held-out RGB-D frames "
                                        "(JSON + side-by-side PNG, misaligned-pose control)")
    rl.add_argument("scan", help="scan folder (or a room id under --scans) holding frames_manifest.json and "
                                 "roomplan.json or scan.json")
    rl.add_argument("--twin", default=None, help="score against this twin.json instead of compiling the scan's geometry")
    rl.add_argument("--max-side", type=int, default=192, help="downscale frames so the longer side is at most this")
    rl.add_argument("--max-frames", type=int, default=8, help="score at most this many held-out frames, evenly spaced")
    rl.add_argument("--png-frames", type=int, default=3, help="frames shown in the side-by-side PNG")
    rl.add_argument("--min-confidence", type=int, default=2, choices=[0, 1, 2],
                    help="ARKit depth confidence a pixel needs to count (2 = high; 0 keeps every non-zero depth)")
    rl.add_argument("--out", default="out")
    rl.add_argument("--scans", default="scans")
    rl.set_defaults(fn=cmd_realism)
    w = sub.add_parser("work", help="process queued uploads under <out>/jobs: twin -> sim dataset -> dataset.zip")
    w.add_argument("--once", action="store_true", help="process one job (or none) and exit")
    w.add_argument("--poll", type=float, default=2.0, help="seconds between queue checks")
    w.add_argument("--episodes", type=int, default=8)
    w.add_argument("--families", default="counter_to_sink,open_drawer,move_on_table",
                   help="all | comma list of counter_to_sink,counter_to_counter,open_drawer,open_door,move_on_table (a family with no variation in this room is skipped)")
    w.add_argument("--people", type=int, default=1)
    w.add_argument("--res", type=int, default=128)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--out", default="out")
    w.add_argument("--max-episodes", type=int, help="refuse --episodes above this (env ARRIVAL_MAX_EPISODES, default 100)")
    w.add_argument("--max-people", type=int, help="refuse --people above this (env ARRIVAL_MAX_PEOPLE, default 4)")
    w.add_argument("--max-job-seconds", type=float,
                   help="kill a job after this many wall-clock seconds (env ARRIVAL_MAX_JOB_SECONDS, default 1800; "
                        "0 = run in-process with no timeout)")
    w.add_argument("--retention-hours", type=float,
                   help="also delete finished jobs older than this while working (env ARRIVAL_RETENTION_HOURS; default keep)")
    w.add_argument("--retries", type=int,
                   help="queue a failed job again up to this many times (env ARRIVAL_JOB_RETRIES, default 2; 0 = never); "
                        "a bad upload, an unusable room or a timeout is not retried")
    w.set_defaults(fn=cmd_work)
    g = sub.add_parser("gc", help="delete finished jobs older than --max-age-hours (upload, twin, dataset, zip)")
    g.add_argument("--max-age-hours", type=float, required=True)
    g.add_argument("--dry-run", action="store_true", help="list what would be deleted, delete nothing")
    g.add_argument("--out", default="out")
    g.set_defaults(fn=cmd_gc)
    dr = sub.add_parser("doctor", help="check python, mujoco, ffmpeg, optional keys (set or not, never shown) and disk")
    dr.add_argument("--out", default="out", help="output folder whose disk is checked")
    dr.add_argument("--json", action="store_true", help="print the checks as JSON")
    dr.set_defaults(fn=cmd_doctor)
    u = sub.add_parser("summary", help="tiny run on several rooms: compile / sim / validate table (markdown + json)")
    u.add_argument("--rooms", default=None,
                   help="comma list of scan names under --scans or paths to RoomPlan/scan .json files "
                        "(default: hero_kitchen plus the two real RoomPlan fixtures)")
    u.add_argument("--episodes", type=int, default=2, help="episode attempts per room")
    u.add_argument("--families", default="counter_to_sink,open_drawer",
                   help="all | comma list of counter_to_sink,counter_to_counter,open_drawer,open_door")
    u.add_argument("--people", type=int, default=1)
    u.add_argument("--res", type=int, default=64)
    u.add_argument("--seed", type=int, default=0)
    u.add_argument("--out", default="out/summary", help="work and output directory (not out/: datasets are rewritten)")
    u.add_argument("--scans", default="scans")
    u.set_defaults(fn=cmd_summary)
    x = sub.add_parser("export-site", help="pre-render the local page to static HTML + data (no server; GitHub Pages)")
    x.add_argument("--out", default="out", help="where datasets, twins, reports and previews are read from")
    x.add_argument("--scans", default="scans")
    x.add_argument("--dest", default="out/static_site", help="folder to write (replaced if it exists)")
    x.add_argument("--zips", action="store_true", help="also include each dataset as a downloadable zip (can be large)")
    x.set_defaults(fn=cmd_export_site)
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
