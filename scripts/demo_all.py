"""One-command demo (DEM-4): scan -> twin -> MuJoCo sim -> GR00T dataset -> validate -> loader check -> eval -> contact sheet.

    python scripts/demo_all.py --quick            # 2 episodes, 1 person, 64 px (about a minute, under 1.5 GB)
    python scripts/demo_all.py --full --serve     # 8 episodes, 2 people, 128 px, then the local page
    python scripts/demo_all.py --quick --room hero_kitchen --out out/demo

Runs the existing functions in-process, prints PASS/FAIL and wall time per stage, and writes
<out>/demo_summary.md, <out>/demo_summary.json and a self-contained <out>/index.html. Exits 1 if a stage
fails or the whole run exceeds --timeout, and 2 if the environment cannot run it (mujoco missing, no
working OpenGL backend).
It is fully offline: API keys are ignored and every non-loopback network connection is refused.
The dataset is a scripted expert in simulation; no policy was trained and no cost was measured.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import html
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

HONEST_LABEL = "Scripted expert in simulation, no policy trained."
FAMILIES = ["counter_to_sink", "open_drawer"]  # the README's sim defaults
SIZES = {"quick": {"episodes": 2, "people": 1, "res": 64}, "full": {"episodes": 8, "people": 2, "res": 128}}
EVAL_EPISODES = 2  # per family
UNMEASURED = ["nebius_token_factory_usd", "nebius_gpu_job_usd", "cosmos_restyle_usd", "finetune_usd",
              "teleop_comparison_usd_per_demo"]


# Keys that would make some other script reach for the network. The demo ignores them.
NETWORK_ENV = ["TOKEN_FACTORY_API_KEY", "TOKEN_FACTORY_BASE_URL", "TAVILY_API_KEY", "TAVILY_BASE_URL",
               "GR00T_POLICY_URL", "COSMOS_URL"]
EXIT_ENV = 2  # the machine cannot run the demo (not a stage failure)
DEFAULT_TIMEOUT_S = 900
GL_PROBE = ("import mujoco;"
            "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>');"
            "r=mujoco.Renderer(m,16,16);d=mujoco.MjData(m);mujoco.mj_forward(m,d);r.update_scene(d);r.render()")


class StageFailed(Exception):
    pass


def gl_candidates(platform: str = sys.platform, forced: str | None = None) -> list[str | None]:
    """MuJoCo GL backends to try in order; None means leave MUJOCO_GL unset (the platform default)."""
    if forced:
        return [forced]
    if platform.startswith("linux"):
        return [None, "egl", "osmesa", "glfw"]
    if platform == "darwin":
        return [None, "glfw"]
    return [None, "glfw", "egl"]


def pick_gl_backend(candidates: list[str | None], probe=None) -> tuple[bool, str | None, list[str]]:
    """First backend whose 16 px offscreen render works: (found, backend or None for the default, tried).

    Each probe is a fresh process because MuJoCo reads MUJOCO_GL once, at import time, and a bad
    backend can abort the interpreter instead of raising.
    """
    def run(backend):
        env = dict(os.environ)
        env.pop("MUJOCO_GL", None)
        env.pop("PYOPENGL_PLATFORM", None)
        if backend:
            env["MUJOCO_GL"] = backend
            if backend == "osmesa":
                env["PYOPENGL_PLATFORM"] = "osmesa"
        try:
            return subprocess.run([sys.executable, "-c", GL_PROBE], env=env, capture_output=True,
                                  timeout=60).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    probe, tried = probe or run, []
    for b in candidates:
        tried.append(b or "default")
        if probe(b):
            return True, b, tried
    return False, None, tried


def preflight(log=print, probe=None) -> str | None:
    """None when the demo can run (MUJOCO_GL is then set for this process), else a one-line reason."""
    if importlib.util.find_spec("mujoco") is None:
        return "mujoco is not installed. The demo needs the sim extra: pip install -e '.[sim]'"
    forced = os.environ.get("MUJOCO_GL") or None
    found, backend, tried = pick_gl_backend(gl_candidates(forced=forced), probe)
    if not found:
        return ("no working OpenGL backend for MuJoCo (tried " + ", ".join(tried) + "). On Linux install "
                "libegl1 and libosmesa6 (or set MUJOCO_GL=osmesa); on Windows and macOS a desktop session "
                "with a GPU driver is needed for glfw")
    if backend:
        os.environ["MUJOCO_GL"] = backend
        if backend == "osmesa":
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    log(f"[demo_all] MuJoCo GL backend: {backend or 'platform default'}")
    return None


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) and address else address
    return isinstance(host, str) and (host in ("localhost", "::1", "") or host.startswith("127."))


@contextlib.contextmanager
def offline_guard():
    """Refuse every non-loopback socket connection and hide API keys for the length of the block."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    hidden = {k: os.environ.pop(k) for k in NETWORK_ENV if k in os.environ}

    def guarded(real):
        def inner(self, address, *args):
            if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
                raise OSError(f"demo_all is offline: refused connection to {address!r}")
            return real(self, address, *args)
        return inner

    socket.socket.connect, socket.socket.connect_ex = guarded(real_connect), guarded(real_connect_ex)
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = real_connect, real_connect_ex
        os.environ.update(hidden)


def start_watchdog(seconds: float, out: Path, exit_fn=os._exit) -> threading.Timer | None:
    """After `seconds`, write a FAIL summary, print one line and exit 1. Cancel the returned timer when done."""
    if not seconds or seconds <= 0:
        return None

    def fire():
        msg = f"timed out after {seconds:g} s"
        with contextlib.suppress(OSError):
            out.mkdir(parents=True, exist_ok=True)
            (out / "demo_summary.json").write_text(json.dumps({"overall": "FAIL", "error": msg}, indent=2))
        print(f"demo_all: FAIL {msg}", file=sys.stderr, flush=True)
        exit_fn(1)

    t = threading.Timer(seconds, fire)
    t.daemon = True
    t.start()
    return t


def _stage(stages: list[dict], name: str, fn, log=print):
    """Run one stage, record wall time and PASS/FAIL, and raise StageFailed on failure."""
    t0 = time.perf_counter()
    err = None
    try:
        detail = fn()
    except (Exception, SystemExit) as e:  # run_demo reports a missing room with SystemExit(message)
        detail = None
        err = " ".join(str(e.code if isinstance(e, SystemExit) else e).split()) or type(e).__name__
    row = {"stage": name, "status": "FAIL" if err else "PASS", "seconds": round(time.perf_counter() - t0, 2),
           "detail": err or (detail if isinstance(detail, str) else None)}
    stages.append(row)
    log(f"[demo_all] {row['status']} {name} ({row['seconds']:.1f}s)" + (f": {err}" if err else ""))
    if err:
        raise StageFailed(f"{name}: {err}")
    return detail


def _render_layout(twin: dict, room: str, out: Path) -> str:
    import render_rooms  # scripts/render_rooms.py

    path = out / "renders" / f"{room}_top.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    render_rooms.render(room, twin, 100.0).save(path)
    return str(path)


def _markdown(s: dict) -> str:
    ds = s["dataset"]
    lines = [f"# Arrival demo summary: {s['room']}", "", f"**{HONEST_LABEL}**", "",
             f"Mode: {s['mode']}. Overall: **{s['overall']}**. Wall time {s['seconds']} s.", "",
             "| Stage | Status | Seconds |", "|---|---|---|"]
    lines += [f"| {r['stage']} | {r['status']} | {r['seconds']} |" for r in s["stages"]]
    lines += ["", "## Dataset", ""]
    if ds:
        lines += [f"- Path: `{ds['path']}`", f"- Episodes: {ds['episodes']}, frames: {ds['frames']}",
                  f"- arrival_physically_valid: {str(ds['physically_valid']).lower()}",
                  f"- joints_are_placeholder: {str(ds['joints_are_placeholder']).lower()}",
                  f"- Demos per hour (this laptop): {ds['demos_per_hour']}"]
    else:
        lines.append("- not produced")
    lines += ["", "## Closed-loop eval (scripted expert)", ""]
    if s["per_family"]:
        lines += ["| Family | Successes | Episodes |", "|---|---|---|"]
        lines += [f"| {k} | {v['successes']} | {v['episodes']} |" for k, v in s["per_family"].items()]
    else:
        lines.append("- not run")
    lines += ["", "## Cost (not measured)", ""] + [f"- {k}: null" for k in s["cost"]]
    if s["error"]:
        lines += ["", "## Error", "", s["error"]]
    return "\n".join(lines) + "\n"


def _index_html(s: dict) -> str:
    """One self-contained page: summary, stage table and the contact sheet inlined as a data URI."""
    e = html.escape
    ds = s["dataset"]
    rows = "".join(f"<tr><td>{e(r['stage'])}</td><td>{r['status']}</td><td>{r['seconds']}</td></tr>"
                   for r in s["stages"])
    img = ""
    sheet = s["artifacts"].get("contact_sheet")
    if sheet and Path(sheet).is_file():
        b64 = base64.b64encode(Path(sheet).read_bytes()).decode()
        img = f'<h2>Contact sheet</h2><img alt="contact sheet" src="data:image/png;base64,{b64}">'
    facts = ""
    if ds:
        facts = (f"<ul><li>Dataset: <code>{e(str(ds['path']))}</code></li>"
                 f"<li>Episodes: {ds['episodes']}, frames: {ds['frames']}</li>"
                 f"<li>arrival_physically_valid: {str(ds['physically_valid']).lower()}</li>"
                 f"<li>joints_are_placeholder: {str(ds['joints_are_placeholder']).lower()}</li></ul>")
    err = f"<p><b>Error:</b> {e(s['error'])}</p>" if s["error"] else ""
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Arrival demo: {e(s['room'])}</title>"
            "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem}"
            "table{border-collapse:collapse}td,th{border:1px solid #8885;padding:.25rem .6rem;text-align:left}"
            "img{max-width:100%}</style></head><body>"
            f"<h1>Arrival demo: {e(s['room'])}</h1><p><b>{e(HONEST_LABEL)}</b></p>"
            f"<p>Mode {e(s['mode'])}. Overall <b>{s['overall']}</b>. Wall time {s['seconds']} s. "
            f"Run offline: no network, no API keys.</p>{err}"
            f"<table><tr><th>Stage</th><th>Status</th><th>Seconds</th></tr>{rows}</table>{facts}{img}"
            "</body></html>\n")


def run(room: str, mode: str, out: Path, scans: str = "scans", log=print) -> tuple[dict, int]:
    from arrival.cli import run_demo

    size = SIZES[mode]
    t_all = time.perf_counter()
    stages: list[dict] = []
    s = {"room": room, "mode": mode, "size": size, "honest_label": HONEST_LABEL, "stages": stages,
         "dataset": None, "per_family": {}, "successes": 0, "cost": {k: None for k in UNMEASURED},
         "artifacts": {}, "error": None}
    quiet = lambda *_: None  # noqa: E731
    try:
        # scan normalize + compile; the 1-episode placeholder dataset it also writes is not the demo dataset
        res = _stage(stages, "scan normalize + compile",
                     lambda: run_demo(room, 1, "none", out, scans, log=quiet), log)
        twin = json.loads((res["twin_dir"] / "twin.json").read_text())
        s["artifacts"]["layout"] = _stage(stages, "twin layout png", lambda: _render_layout(twin, room, out), log)

        from arrival.sim.collect import collect

        sim = _stage(stages, "sim dataset", lambda: collect(
            twin, FAMILIES, size["episodes"], out, people=size["people"], res=size["res"], save_preview=True,
            log=quiet), log)
        ds_dir = Path(sim["dataset"])
        info = json.loads((ds_dir / "meta" / "info.json").read_text())
        emb = json.loads((ds_dir / "meta" / "embodiment.json").read_text())  # the placeholder flag lives here
        rep = sim["report_data"]
        s["dataset"] = {"path": str(ds_dir), "episodes": info["total_episodes"], "frames": info["total_frames"],
                        "physically_valid": info.get("arrival_physically_valid"),
                        "joints_are_placeholder": emb.get("joints_are_placeholder"),
                        "demos_per_hour": rep["demos_per_hour"], "attempts": rep["attempts"]}
        s["successes"] = rep["successes"]

        def check():
            from arrival.export.validate import validate

            errs = validate(ds_dir)
            if errs:
                raise StageFailed("; ".join(errs[:3]))
            return f"{info['total_episodes']} episodes valid"

        _stage(stages, "validate", check, log)

        def loader():
            from arrival.export.loader_check import check_loader

            res = check_loader(ds_dir)
            if not res["ok"]:
                raise StageFailed("; ".join(f"{e['code']}: {e['msg']}" for e in res["errors"][:3]))
            return f"{res['info']['videos_decoded']} mp4s decoded, {len(res['warnings'])} warnings"

        _stage(stages, "loader check", loader, log)

        def ev():
            from arrival.eval.harness import evaluate

            r = evaluate(twin, FAMILIES, EVAL_EPISODES, "scripted", people=size["people"], res=size["res"],
                         out_dir=out, log=quiet)
            s["artifacts"].update(eval_report=str(r["path"]), eval_manifest=r["manifest_path"],
                                  eval_episodes=r["episodes_path"])
            s["per_family"] = {k: {"successes": v["successes"], "episodes": v["episodes"]}
                               for k, v in r["families"].items()}
            if not s["per_family"]:
                raise StageFailed("no eval episodes ran")
            return f"{r['totals']['successes']}/{r['totals']['episodes']}"

        _stage(stages, "closed-loop eval", ev, log)

        def sheet():
            p = out / "previews" / f"{room}_sim_contact.png"
            if not p.is_file():
                raise StageFailed(f"missing {p}")
            return str(p)

        s["artifacts"]["contact_sheet"] = _stage(stages, "contact sheet", sheet, log)
    except StageFailed as e:
        s["error"] = str(e)
    s["overall"] = "FAIL" if s["error"] else "PASS"
    s["seconds"] = round(time.perf_counter() - t_all, 2)
    s["physically_valid"] = bool(s["dataset"] and s["dataset"]["physically_valid"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_summary.json").write_text(json.dumps(s, indent=2))
    (out / "demo_summary.md").write_text(_markdown(s))
    (out / "index.html").write_text(_index_html(s), encoding="utf-8")
    return s, 1 if s["error"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="2 episodes, 1 person, 64 px (default)")
    g.add_argument("--full", action="store_true", help="8 episodes, 2 people, 128 px")
    ap.add_argument("--room", default="hero_kitchen")
    ap.add_argument("--out", default="out")
    ap.add_argument("--scans", default="scans")
    ap.add_argument("--serve", action="store_true", help="start the local page when every stage passed")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help=f"abort with exit 1 after this many seconds, 0 for none (default {DEFAULT_TIMEOUT_S})")
    a = ap.parse_args(argv)
    out = Path(a.out)
    why = preflight()
    if why:
        print(f"demo_all: {why}", file=sys.stderr)
        return EXIT_ENV
    watchdog = start_watchdog(a.timeout, out)
    try:
        with offline_guard():
            s, code = run(a.room, "full" if a.full else "quick", out, a.scans)
    finally:
        if watchdog:
            watchdog.cancel()
    if code:
        print(f"demo_all: FAIL {s['error']}", file=sys.stderr)
        return code
    print((out / "demo_summary.md").read_text())
    print(f"summary: {out / 'demo_summary.md'}\npage: {out / 'index.html'}")
    if a.serve:
        from arrival.cli import resolve_scans
        from arrival.web.app import serve

        serve(port=a.port, out_dir=out, scans_dir=resolve_scans(a.scans))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
