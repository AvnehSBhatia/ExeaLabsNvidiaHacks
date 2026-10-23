"""Static export of the local web page: pre-rendered HTML + JSON data, no server needed (GitHub Pages friendly).

The page is rendered by the real ``arrival.web.app`` handler: a throwaway server on an ephemeral 127.0.0.1 port is
crawled, so the export can never drift from what ``arrival serve`` shows. Then the HTML is made static:

* every absolute link (``/rooms/x``) becomes a relative file link (``../rooms/x/index.html``), so it works under a
  project-site prefix such as ``https://user.github.io/repo/``
* the upload form, the jobs table and the page script are replaced by a note, because there is no backend to take a
  scan. The note says how to run the live page instead
* dataset zips are linked only with ``zips=True``; otherwise the download buttons are removed (a zip of every
  dataset can be large)
* ``data/<room>.json`` holds the numbers the page shows, for anything that wants them without scraping HTML

The result is link-checked before this returns: every relative link and asset must resolve inside ``dest``.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlsplit

from arrival.web.app import ID_RE, PREVIEW_RE, make_server

ATTR_RE = re.compile(r"""(href|src)=(['"])(.*?)\2""", re.S)
SCRIPT_RE = re.compile(r"<script>.*?</script>", re.S)
UPLOAD_MARK = "<h2>Upload a scan</h2>"
STATIC_BANNER = ("<div class='warn'>Static snapshot of the local Arrival page. There is no server behind it: "
                 "uploads and the jobs list are not available here. Run <code>python -m arrival serve</code> to "
                 "scan your own room.</div>")
STATIC_UPLOAD = ("<h2>Upload a scan</h2><div class='card empty'><b>Live upload is offline in this snapshot</b>"
                 "Run <code>python -m arrival serve</code> and <code>python -m arrival work</code> locally to turn a "
                 "RoomPlan export into a dataset.</div>")
RETRIES = 5


class ExportError(RuntimeError):
    pass


def _get(url: str, dest: Path | None = None) -> bytes:
    """GET with a short backoff on 429/503 (a hardened server may rate limit the crawler)."""
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                if dest is not None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, "wb") as f:
                        shutil.copyfileobj(r, f)
                    return b""
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < RETRIES - 1:
                time.sleep(min(float(e.headers.get("Retry-After") or 0.2 * 2 ** attempt), 5.0))
                continue
            raise ExportError(f"GET {url} -> HTTP {e.code}") from e
        except urllib.error.URLError as e:
            raise ExportError(f"GET {url} failed: {e.reason}") from e
    raise ExportError(f"GET {url}: still throttled after {RETRIES} tries")


def _rel(depth: int, target: str) -> str:
    return "../" * depth + target


def _rewrite(html: str, depth: int, zips: set[str]) -> str:
    """Absolute links -> relative file links. A link with no static equivalent is dropped to '#removed'."""
    html = SCRIPT_RE.sub("", html)
    html = html.replace("<main>", "<main>" + STATIC_BANNER, 1)
    if UPLOAD_MARK in html and "</main>" in html:
        head, rest = html.split(UPLOAD_MARK, 1)
        html = head + STATIC_UPLOAD + rest[rest.index("</main>"):]
    else:  # layout changed: still never ship a dead form; the link check reports what is left
        html = re.sub(r"<form.*?</form>", "", html, flags=re.S)

    def sub(m: re.Match) -> str:
        attr, q, url = m.groups()
        if not url.startswith("/") or url.startswith("//"):
            return m.group(0)
        path = url.split("?", 1)[0].strip("/")
        parts = path.split("/") if path else []
        if not parts:
            new = _rel(depth, "index.html")
        elif parts[0] == "rooms" and len(parts) == 2:
            new = _rel(depth, f"rooms/{parts[1]}/index.html")
        elif parts[0] == "previews" and len(parts) == 2:
            new = _rel(depth, f"previews/{parts[1]}")
        elif parts[0] == "download" and len(parts) == 2 and parts[1] in zips:
            new = _rel(depth, f"downloads/{parts[1]}.zip")
        else:
            new = "#removed"
        return f"{attr}={q}{new}{q}"

    html = ATTR_RE.sub(sub, html)
    if not zips:  # no zip is shipped: drop the buttons instead of leaving dead '#removed' links
        html = re.sub(r"<a class='btn[^']*' href='#removed'>[^<]*</a>", "", html)
        html = re.sub(r"<p></p>", "", html)
    return html


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def room_data(room: str, out_dir: Path, scans_dir: Path) -> dict:
    """The facts the room page shows, as JSON. Anything absent on disk is null, never invented."""
    twin = _read(out_dir / "twins" / room / "twin.json")
    scan = _read(scans_dir / room / "scan.json") or {}
    src = twin or scan
    datasets = {}
    for name in (room, f"{room}_sim"):
        info = _read(out_dir / "datasets" / name / "meta" / "info.json")
        if isinstance(info, dict):
            keep = ("arrival_generator", "arrival_physically_valid", "arrival_appearance", "arrival_video_policy",
                    "arrival_successes", "arrival_attempts", "total_episodes", "total_frames", "robot_type", "fps")
            datasets[name] = {k: info.get(k) for k in keep if k in info}
    return {"room_id": room, "geometry_source": scan.get("geometry_source"),
            "twin_compiled": twin is not None,
            "fixtures": [{k: f.get(k) for k in ("id", "type", "size", "position", "yaw")}
                         for f in (src.get("fixtures") or []) if isinstance(f, dict)],
            "datasets": datasets,
            "report": _read(out_dir / "reports" / f"{room}_sim.json") or _read(out_dir / "reports" / f"{room}.json")}


def link_check(dest: Path) -> list[str]:
    """Relative links and assets in every exported .html that do not resolve to a file inside ``dest``."""
    dest = dest.resolve()
    broken = []
    for page in sorted(dest.rglob("*.html")):
        for _, _, url in ATTR_RE.findall(page.read_text(encoding="utf-8")):
            url = urldefrag(url)[0]
            if not url or urlsplit(url).scheme or url.startswith("//"):
                continue
            target = (page.parent / unquote(urlsplit(url).path)).resolve()
            if not (target.is_file() and dest in target.parents):
                broken.append(f"{page.relative_to(dest)}: {url}")
    return broken


def export_site(out_dir: str | Path = "out", scans_dir: str | Path = "scans", dest: str | Path = "out/static_site",
                zips: bool = False, log=print) -> dict:
    """Crawl the local page and write the static copy at ``dest`` (replaced if it exists). Returns a summary dict;
    raises ExportError when ``dest`` is a non-empty folder that is not an earlier export, when the crawl fails, or
    when the link check finds a broken link."""
    out_dir, scans_dir, dest = Path(out_dir), Path(scans_dir), Path(dest)
    dest_res = dest.resolve()
    out_res = out_dir.resolve()
    scans_res = scans_dir.resolve()
    cwd_res = Path(".").resolve()

    if dest_res == cwd_res or dest_res in (out_res, scans_res):
        raise ExportError(f"refusing to export over {dest}")
    if dest_res in cwd_res.parents or dest_res in out_res.parents or dest_res in scans_res.parents:
        raise ExportError(f"refusing to export to ancestor directory {dest}")
    if dest.is_file():
        raise ExportError(f"refusing to export over existing file {dest}")
    if dest.is_dir() and any(dest.iterdir()) and not (dest / "data" / "index.json").is_file():
        raise ExportError(f"refusing to replace {dest}: it is not empty and not an earlier export (no data/index.json)")

    server = make_server("127.0.0.1", 0, out_dir, scans_dir)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.is_dir() and not tmp.is_symlink():
        shutil.rmtree(tmp, ignore_errors=True)
    elif tmp.exists() or tmp.is_symlink():
        tmp.unlink(missing_ok=True)

    try:
        try:
            index = _get(base + "/").decode("utf-8")
            rooms = sorted(set(re.findall(r"href='/rooms/([a-z0-9_]+)'", index)))
            pages = {"index.html": (index, 0)}
            for room in rooms:
                pages[f"rooms/{room}/index.html"] = (_get(f"{base}/rooms/{room}").decode("utf-8"), 2)
            previews = sorted({p for html, _ in pages.values() for p in re.findall(r"/previews/([A-Za-z0-9_.-]+)", html)
                               if PREVIEW_RE.match(p)})
            for name in previews:
                _get(f"{base}/previews/{name}", tmp / "previews" / name)
            zipped = set()
            if zips:
                for name in sorted(set(re.findall(r"/download/([a-z0-9_]+)", "".join(h for h, _ in pages.values())))):
                    _get(f"{base}/download/{name}", tmp / "downloads" / f"{name}.zip")
                    zipped.add(name)
            for rel, (html, depth) in pages.items():
                target = tmp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(_rewrite(html, depth, zipped), encoding="utf-8")
            (tmp / "data").mkdir(parents=True, exist_ok=True)
            rows = [room_data(r, out_dir, scans_dir) for r in rooms if ID_RE.match(r)]
            for row in rows:
                (tmp / "data" / f"{row['room_id']}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            (tmp / "data" / "index.json").write_text(json.dumps(
                {"rooms": [r["room_id"] for r in rows], "datasets_present": {r["room_id"]: sorted(r["datasets"]) for r in rows},
                 "zips_included": sorted(zipped), "note": "static snapshot; no server, no uploads"}, indent=2) + "\n", encoding="utf-8")
            (tmp / ".nojekyll").write_text("", encoding="utf-8")
        finally:
            server.shutdown()
            server.server_close()

        broken = link_check(tmp)
        if broken:
            raise ExportError("broken links in the export:\n  " + "\n  ".join(broken[:20]))

        if dest.exists() or dest.is_symlink():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest, ignore_errors=True)
            else:
                dest.unlink(missing_ok=True)
        tmp.rename(dest)
    except Exception:
        if tmp.is_dir() and not tmp.is_symlink():
            shutil.rmtree(tmp, ignore_errors=True)
        elif tmp.exists() or tmp.is_symlink():
            tmp.unlink(missing_ok=True)
        raise

    empty = [r["room_id"] for r in rows if not r["datasets"]]
    if empty:
        log(f"[export-site] warning: no dataset under {out_dir}/datasets for {', '.join(empty)}; "
            "those pages say 'not generated' (run python -m arrival demo first)")
    return {"dest": str(dest), "rooms": [r["room_id"] for r in rows], "pages": len(pages), "previews": len(previews),
            "zips": sorted(zipped), "rooms_without_dataset": empty}
