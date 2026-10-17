"""Local pre-scanned room page and queued upload.

Standard library only. Single process, no authentication, binds 127.0.0.1.
``POST /jobs`` stores the upload and returns 202. It never compiles or
generates inside the request, and this module does not import the pipeline.
"""

from __future__ import annotations

import html
import io
import json
import mmap
import re
import tempfile
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from arrival.jobs.queue import check_upload_size, create_job, read_job

ID_RE = re.compile(r"^[a-z0-9_]+$")
BOUNDARY_RE = re.compile(r'boundary="?([^";]+)"?', re.I)
NAME_RE = re.compile(rb'name="([^"]*)"', re.I)
FILENAME_RE = re.compile(rb'filename="([^"]*)"', re.I)
READ_CHUNK = 1 << 20


class UploadError(ValueError):
    pass


def find_part(buf, boundary: bytes, field: str) -> tuple[int, int, str]:
    """Scan a multipart body for form field ``field``; return (data offset, data length, filename).
    Raises UploadError on a missing closing boundary or a missing field."""
    delim = b"--" + boundary
    if buf.find(delim + b"--") < 0:
        raise UploadError("multipart body has no closing boundary (truncated?)")
    pos = buf.find(delim)
    while pos >= 0:
        start = pos + len(delim)
        if buf[start:start + 2] == b"--":
            break
        head_end = buf.find(b"\r\n\r\n", start)
        if head_end < 0:
            raise UploadError("malformed multipart part headers")
        headers = bytes(buf[start:head_end])
        nxt = buf.find(b"\r\n" + delim, head_end + 4)
        if nxt < 0:
            raise UploadError("multipart part has no terminating boundary")
        name = NAME_RE.search(headers)
        if name and name.group(1).decode("latin-1") == field:
            fn = FILENAME_RE.search(headers)
            return head_end + 4, nxt - (head_end + 4), fn.group(1).decode("utf-8", "replace") if fn else ""
        pos = nxt + 2
    raise UploadError(f"missing file field {field!r}")

CSS = """
:root { --fg:#1d232a; --muted:#5b6672; --line:#d9dee3; --bg:#f7f8fa; --card:#fff;
        --warn-bg:#fff4e0; --warn-fg:#7a4a00; --accent:#1f5fa8; }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.5 -apple-system, system-ui, "Segoe UI", sans-serif;
       color:var(--fg); background:var(--bg); }
main { max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: .02em; }
h2 { font-size: 17px; margin: 28px 0 8px; }
a { color: var(--accent); }
.muted { color: var(--muted); }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:14px 16px; margin:12px 0; }
.warn { background:var(--warn-bg); color:var(--warn-fg); border:1px solid #f0d199;
        border-radius:8px; padding:10px 14px; margin:12px 0; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
         vertical-align: top; }
th { font-weight:600; color:var(--muted); font-size:13px; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.scroll { overflow-x: auto; }
form { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
button { padding:6px 12px; border-radius:6px; border:1px solid var(--line);
         background:var(--fg); color:#fff; cursor:pointer; }
"""

PLACEHOLDER_NOTE = (
    "Motion in this dataset is a scripted placeholder in simulation "
    "(arrival_generator = scripted_retarget). It is not physically valid, not "
    "MimicGen, and not a real-robot result."
)


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _page(title: str, body: str) -> bytes:
    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )
    return doc.encode("utf-8")


def _fmt_list(values) -> str:
    if isinstance(values, (list, tuple)):
        return "[" + ", ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in values) + "]"
    return str(values)


class ArrivalServer(ThreadingHTTPServer):
    daemon_threads = True
    out_dir: Path
    scans_dir: Path


class Handler(BaseHTTPRequestHandler):
    server: ArrivalServer
    server_version = "Arrival/0.1"
    timeout = 30  # seconds per socket read; a stalled upload is dropped instead of pinning a thread

    # ---- helpers -------------------------------------------------------
    @property
    def out_dir(self) -> Path:
        return Path(self.server.out_dir)

    @property
    def scans_dir(self) -> Path:
        return Path(self.server.scans_dir)

    def log_message(self, format, *args):  # quieter default logging
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _html(self, code: int, title: str, body: str):
        self._send(code, _page(title, body), "text/html; charset=utf-8")

    def _json(self, code: int, obj):
        self._send(code, (json.dumps(obj, indent=2) + "\n").encode(), "application/json")

    def _not_found(self, msg: str = "Not found"):
        self._html(404, "Not found", f"<h1>404</h1><p>{esc(msg)}</p><p><a href='/'>Back</a></p>")

    def _rooms(self) -> list[str]:
        if not self.scans_dir.is_dir():
            return []
        return sorted(
            p.name for p in self.scans_dir.iterdir()
            if p.is_dir() and (p / "scan.json").is_file() and ID_RE.match(p.name)
        )

    # ---- routing -------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/":
            return self._index()
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "download":
            if not ID_RE.match(parts[1]):
                return self._not_found("Invalid id")
            return self._job_download(parts[1])
        if len(parts) == 2 and parts[0] in ("rooms", "download", "jobs"):
            ident = parts[1]
            if not ID_RE.match(ident):
                return self._not_found("Invalid id")
            if parts[0] == "rooms":
                return self._room(ident)
            if parts[0] == "download":
                return self._download(ident)
            return self._job(ident)
        return self._not_found()

    do_HEAD = do_GET

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/jobs":
            return self._not_found()
        return self._post_job()

    # ---- pages ---------------------------------------------------------
    def _task_counts(self, room: str):
        tasks = _read_json(self.out_dir / "twins" / room / "tasks.json")
        if not tasks:
            return None
        return Counter(t.get("family", "?") for t in tasks.get("tasks", []))

    def _index(self):
        rows = []
        for room in self._rooms():
            scan = _read_json(self.scans_dir / room / "scan.json") or {}
            counts = self._task_counts(room)
            if counts is None:
                counts_html = ("<span class='muted'>not generated, run "
                               "<code>python -m arrival demo</code></span>")
            else:
                counts_html = "<br>".join(
                    f"<code>{esc(f)}</code>: {n}" for f, n in sorted(counts.items())
                ) or "0"
            ds = self.out_dir / "datasets" / room
            exists = ds.is_dir()
            ds_html = f"<code>{esc(ds)}</code> " + (
                "<span>(exists)</span>" if exists else "<span class='muted'>(missing)</span>"
            )
            dl = (f"<a href='/download/{esc(room)}'>{esc(room)}.zip</a>" if exists
                  else "<span class='muted'>none</span>")
            rows.append(
                f"<tr><td><a href='/rooms/{esc(room)}'>{esc(room)}</a></td>"
                f"<td><code>{esc(scan.get('geometry_source', 'unknown'))}</code></td>"
                f"<td>{counts_html}</td><td>{ds_html}</td><td>{dl}</td></tr>"
            )
        table = (
            "<div class='scroll'><table><tr><th>Room</th><th>geometry_source</th>"
            "<th>Tasks per family</th><th>Dataset</th><th>Download</th></tr>"
            + ("".join(rows) or "<tr><td colspan='5' class='muted'>No rooms under scans/</td></tr>")
            + "</table></div>"
        )
        upload = (
            "<h2>Queue a scan</h2><div class='card'>"
            "<form method='post' action='/jobs' enctype='multipart/form-data'>"
            "<input type='file' name='scan' required> <button type='submit'>Upload</button>"
            "</form><p class='muted'>Stores the file under <code>out/jobs/&lt;id&gt;/</code> "
            "and returns a queued status (max 200 MB). Nothing is compiled or generated "
            "in the request. Run <code>python -m arrival work</code> in another terminal to turn "
            "queued scans into sim datasets.</p></div>"
        )
        body = (
            "<h1>Arrival</h1><p class='muted'>Pre-scanned rooms → GR00T LeRobot v2 "
            "datasets, generated locally.</p>"
            f"<div class='warn'>{esc(PLACEHOLDER_NOTE)}</div>"
            "<h2>Rooms</h2>" + table + upload + "<h2>Jobs</h2>" + self._jobs_table()
        )
        self._html(200, "Arrival", body)

    def _jobs(self) -> list[dict]:
        root = self.out_dir / "jobs"
        if not root.is_dir():
            return []
        jobs = [s for d in root.iterdir() if ID_RE.match(d.name) and (s := _read_json(d / "status.json"))]
        return sorted(jobs, key=lambda s: (-s.get("created_unix", 0), s.get("id", "")))

    def _jobs_table(self) -> str:
        rows = []
        for s in self._jobs():
            jid, st = s.get("id", "?"), s.get("status", "?")
            if st == "done" and s.get("zip"):
                result = f"<a href='/jobs/{esc(jid)}/download'>dataset.zip</a>"
            elif st == "failed":
                result = f"<span class='muted'>{esc(s.get('error'))}</span>"
            else:
                result = f"<span class='muted'>{esc(s.get('stage') or '')}</span>"
            eps = ("" if s.get("attempts") is None
                   else f"{esc(s.get('successes'))}/{esc(s.get('attempts'))}")
            rows.append(
                f"<tr><td><a href='/jobs/{esc(jid)}'><code>{esc(jid)}</code></a></td>"
                f"<td>{esc(s.get('filename'))}</td><td><code>{esc(st)}</code></td>"
                f"<td><code>{esc(s.get('room_id') or '')}</code></td><td>{eps}</td><td>{result}</td></tr>"
            )
        return (
            "<div class='scroll'><table><tr><th>Job</th><th>Upload</th><th>Status</th><th>Room</th>"
            "<th>Episodes</th><th>Result</th></tr>"
            + ("".join(rows) or "<tr><td colspan='6' class='muted'>No uploads yet</td></tr>")
            + "</table></div>"
        )

    def _room(self, room: str):
        scan_path = self.scans_dir / room / "scan.json"
        if not scan_path.is_file():
            return self._not_found(f"No room {room}")
        scan = _read_json(scan_path) or {}
        twin = _read_json(self.out_dir / "twins" / room / "twin.json")
        src = twin if twin else scan
        fixtures = src.get("fixtures", []) or []
        fx_rows = "".join(
            "<tr>"
            f"<td><code>{esc(f.get('id'))}</code></td>"
            f"<td>{esc(f.get('type', f.get('category', '')))}</td>"
            f"<td class='mono'>{esc(_fmt_list(f.get('size', '')))}</td>"
            f"<td class='mono'>{esc(_fmt_list(f.get('position', '')))}</td>"
            f"<td class='mono'>{esc(f.get('yaw', ''))}</td>"
            "</tr>"
            for f in fixtures if isinstance(f, dict)
        )
        fx_table = (
            f"<p class='muted'>Source: {'compiled twin.json' if twin else 'scan.json (twin not compiled)'}</p>"
            "<div class='scroll'><table><tr><th>id</th><th>type</th><th>size (m)</th>"
            "<th>position (m)</th><th>yaw</th></tr>"
            + (fx_rows or "<tr><td colspan='5' class='muted'>No fixtures</td></tr>")
            + "</table></div>"
        )

        tasks = _read_json(self.out_dir / "twins" / room / "tasks.json")
        if tasks and tasks.get("tasks"):
            grouped = Counter(
                (t.get("family", "?"), t.get("object_id", "?"), t.get("instruction", ""))
                for t in tasks["tasks"]
            )
            fam_counts = Counter(t.get("family", "?") for t in tasks["tasks"])
            summary = " · ".join(f"<code>{esc(f)}</code>: {n}" for f, n in sorted(fam_counts.items()))
            t_rows = "".join(
                f"<tr><td><code>{esc(fam)}</code></td><td><code>{esc(obj)}</code></td>"
                f"<td>{esc(instr)}</td><td>{n}</td></tr>"
                for (fam, obj, instr), n in sorted(grouped.items())
            )
            tasks_html = (
                f"<p>{summary}</p><div class='scroll'><table><tr><th>family</th><th>object</th>"
                "<th>instruction</th><th>count</th></tr>" + t_rows + "</table></div>"
            )
        else:
            tasks_html = (
                "<p class='muted'>Families <code>counter_to_sink</code> and "
                "<code>counter_to_counter</code>. Not generated, run "
                "<code>python -m arrival demo</code>.</p>"
            )

        info = _read_json(self.out_dir / "datasets" / room / "meta" / "info.json") or {}
        valid = info.get("arrival_physically_valid", False)
        valid_text = "true" if valid is True else "false"
        info_bits = ""
        if info:
            info_bits = "".join(
                f"<tr><th>{esc(k)}</th><td><code>{esc(info.get(k))}</code></td></tr>"
                for k in ("arrival_generator", "arrival_appearance", "arrival_video_policy",
                          "total_episodes", "total_frames", "arrival_successes",
                          "arrival_attempts", "robot_type", "fps")
                if k in info
            )
            info_bits = f"<div class='scroll'><table>{info_bits}</table></div>"

        report = _read_json(self.out_dir / "reports" / f"{room}.json")
        if report:
            rep_rows = "".join(
                f"<tr><th>{esc(k)}</th><td><code>{esc(report.get(k))}</code></td></tr>"
                for k in ("attempts", "successes", "dropped", "seconds", "demos_per_hour")
            )
            dollars = report.get("dollars") or {}
            d_rows = "".join(
                f"<tr><th>{esc(k)}</th><td><code>{esc(v.get('usd') if isinstance(v, dict) else v)}</code></td>"
                f"<td>{'measured' if isinstance(v, dict) and v.get('measured') else 'not measured'}</td></tr>"
                for k, v in dollars.items()
            )
            report_html = (
                f"<div class='scroll'><table>{rep_rows}</table></div>"
                "<h2>Cost (USD)</h2><div class='scroll'><table><tr><th>item</th><th>usd</th>"
                f"<th>status</th></tr>{d_rows}</table></div>"
            )
        else:
            report_html = (f"<p class='muted'>No report at <code>out/reports/{esc(room)}.json</code> "
                           "yet.</p>")

        ds_exists = (self.out_dir / "datasets" / room).is_dir()
        dl = (f"<p><a href='/download/{esc(room)}'>Download {esc(room)}.zip</a></p>"
              if ds_exists else "<p class='muted'>Dataset not generated yet.</p>")

        sim_dir = self.out_dir / "datasets" / f"{room}_sim"
        sim_info = _read_json(sim_dir / "meta" / "info.json") if sim_dir.is_dir() else None
        if sim_dir.is_dir():
            si = sim_info if isinstance(sim_info, dict) else {}
            fams = si.get("arrival_families") or si.get("families") or []
            sim_html = (
                f"<p><a href='/download/{esc(room)}_sim'>Download {esc(room)}_sim.zip</a></p>"
                "<div class='scroll'><table>"
                f"<tr><td>arrival_generator</td><td class='mono'>{esc(si.get('arrival_generator', 'unknown'))}</td></tr>"
                f"<tr><td>arrival_physically_valid</td><td class='mono'>{esc(si.get('arrival_physically_valid', 'unknown'))}</td></tr>"
                f"<tr><td>episodes</td><td class='mono'>{esc(si.get('total_episodes', 'unknown'))}</td></tr>"
                f"<tr><td>families</td><td class='mono'>{esc(', '.join(map(str, fams)) if isinstance(fams, list) else fams)}</td></tr>"
                "</table></div>"
            )
        else:
            sim_html = (f"<p class='muted'>No simulation dataset at <code>out/datasets/{esc(room)}_sim</code>. "
                        "Run <code>python -m arrival sim</code>.</p>")

        body = (
            f"<p><a href='/'>← All rooms</a></p><h1>{esc(room)}</h1>"
            f"<p class='muted'>geometry_source: <code>{esc(scan.get('geometry_source', 'unknown'))}</code></p>"
            f"<div class='warn'><strong class='mono'>arrival_physically_valid: {valid_text}</strong>"
            f"<br>{esc(PLACEHOLDER_NOTE)} Do not fine-tune on it while this flag is false.</div>"
            "<h2>Fixtures</h2>" + fx_table
            + "<h2>Tasks</h2>" + tasks_html
            + "<h2>Dataset</h2>" + dl + info_bits
            + "<h2>Simulation dataset (MuJoCo)</h2>" + sim_html
            + "<h2>Report</h2>" + report_html
        )
        self._html(200, f"{room} · Arrival", body)

    def _download(self, room: str):
        ds = self.out_dir / "datasets" / room
        if not ds.is_dir():
            return self._not_found(f"No dataset for {room}")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(ds.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(room) / p.relative_to(ds))
        self._send(
            200, buf.getvalue(), "application/zip",
            {"Content-Disposition": f"attachment; filename={room}.zip"},
        )

    def _job(self, job_id: str):
        status = read_job(self.out_dir / "jobs", job_id)
        if status is None:
            return self._json(404, {"error": "job not found", "id": job_id})
        self._json(200, status)

    def _job_download(self, job_id: str):
        status = read_job(self.out_dir / "jobs", job_id)
        if status is None:
            return self._json(404, {"error": "job not found", "id": job_id})
        zip_path = self.out_dir / "jobs" / job_id / "dataset.zip"
        if status.get("status") != "done" or not zip_path.is_file():
            return self._json(404, {"error": "dataset not ready", "id": job_id, "status": status.get("status")})
        room = status.get("room_id") or job_id
        self._send(200, zip_path.read_bytes(), "application/zip",
                   {"Content-Disposition": f"attachment; filename={room}_sim.zip"})

    def _post_job(self):
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json(411, {"error": "Content-Length required"})
        if not check_upload_size(length):
            self.close_connection = True
            return self._json(413, {"error": "upload too large", "max_bytes": 200 * 1024 * 1024})
        ctype = self.headers.get("Content-Type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            return self._json(400, {"error": "expected multipart/form-data"})
        m = BOUNDARY_RE.search(ctype)
        if not m:
            return self._json(400, {"error": "multipart boundary missing"})
        with tempfile.TemporaryFile() as tmp:
            left = length
            while left > 0:  # stream to disk; never hold the whole body in memory
                chunk = self.rfile.read(min(left, READ_CHUNK))
                if not chunk:
                    break
                tmp.write(chunk)
                left -= len(chunk)
            if left:
                self.close_connection = True
                return self._json(400, {"error": "truncated body", "expected": length, "got": length - left})
            if length == 0:
                return self._json(400, {"error": "empty body"})
            tmp.flush()
            with mmap.mmap(tmp.fileno(), 0, access=mmap.ACCESS_READ) as buf:
                try:
                    offset, size, filename = find_part(buf, m.group(1).encode("latin-1"), "scan")
                except UploadError as e:
                    return self._json(400, {"error": str(e)})
            if not filename:
                return self._json(400, {"error": "file field 'scan' has no filename"})
            if size <= 0:
                return self._json(400, {"error": "uploaded file is empty"})
            status = create_job(self.out_dir / "jobs", filename, src=tmp, offset=offset, length=size)
        self._json(202, {"status": "queued", "id": status["id"],
                         "status_url": f"/jobs/{status['id']}"})


def make_server(host: str = "127.0.0.1", port: int = 8000,
                out_dir: Path = Path("out"), scans_dir: Path = Path("scans")) -> ThreadingHTTPServer:
    server = ArrivalServer((host, port), Handler)
    server.out_dir = Path(out_dir)
    server.scans_dir = Path(scans_dir)
    return server


def serve(port: int = 8000, out_dir: Path = Path("out"), scans_dir: Path = Path("scans")) -> None:
    server = make_server("127.0.0.1", port, out_dir, scans_dir)
    host, real_port = server.server_address[:2]
    print(f"Arrival serving on http://{host}:{real_port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
