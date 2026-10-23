import http.client
import json
import threading
from pathlib import Path

import pytest

from arrival.jobs.queue import check_upload_size
from arrival.web.app import make_server

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def server(tmp_path):
    out = tmp_path / "out"
    srv = make_server(port=0, out_dir=out, scans_dir=REPO / "scans")
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        yield srv, out
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _get(srv, path):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def test_index_lists_hero_kitchen(server):
    srv, _ = server
    status, body = _get(srv, "/")
    assert status == 200
    assert b"hero_kitchen" in body


def test_healthz_is_ok_and_leaks_nothing(server):
    srv, _ = server
    status, body = _get(srv, "/healthz")
    assert status == 200 and json.loads(body) == {"status": "ok"}


def test_room_page_shows_flag(server):
    srv, _ = server
    status, body = _get(srv, "/rooms/hero_kitchen")
    assert status == 200
    assert b"arrival_physically_valid: false" in body


def test_post_job_queues(server):
    srv, out = server
    boundary = "----arrivaltestboundary"
    payload = b'{"room_id": "uploaded"}'
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="scan"; filename="scan.json"\r\n'
        "Content-Type: application/json\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request("POST", "/jobs", body=body,
                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    assert resp.status == 202
    assert data["status"] == "queued"
    job_dir = out / "jobs" / data["id"]
    status = json.loads((job_dir / "status.json").read_text())
    assert status["status"] == "queued"
    assert status["bytes"] == len(payload)
    assert (job_dir / "upload.bin").read_bytes() == payload

    code, got = _get(srv, data["status_url"])
    assert code == 200
    assert json.loads(got)["id"] == data["id"]


def test_post_without_scan_field_is_400(server):
    srv, _ = server
    boundary = "b0undary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"other\"\r\n\r\nx\r\n"
            f"--{boundary}--\r\n").encode()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request("POST", "/jobs", body=body,
                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 400


def test_upload_limit():
    assert check_upload_size(201 * 1024 * 1024) is False
    assert check_upload_size(1024) is True


def test_download_missing_is_404(server):
    srv, _ = server
    status, _ = _get(srv, "/download/nonexistent")
    assert status == 404
    status, _ = _get(srv, "/download/..%2F..")
    assert status == 404


def test_download_zip(server):
    import io
    import zipfile

    srv, out = server
    meta = out / "datasets" / "hero_kitchen" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text('{"arrival_physically_valid": false}')
    status, body = _get(srv, "/download/hero_kitchen")
    assert status == 200
    names = zipfile.ZipFile(io.BytesIO(body)).namelist()
    assert "hero_kitchen/meta/info.json" in names


def _raw_post(srv, body: bytes, length: int, boundary: str = "bnd"):
    import socket

    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
    s.sendall((f"POST /jobs HTTP/1.1\r\nHost: x\r\nContent-Type: multipart/form-data; boundary={boundary}\r\n"
               f"Content-Length: {length}\r\nConnection: close\r\n\r\n").encode() + body)
    s.shutdown(socket.SHUT_WR)  # client stops sending: server sees EOF before Content-Length bytes
    data = b""
    while chunk := s.recv(65536):
        data += chunk
    s.close()
    return int(data.split(b" ", 2)[1])


def _part(filename: str, payload: bytes, boundary: str = "bnd", close: bool = True) -> bytes:
    fn = f'; filename="{filename}"' if filename is not None else ""
    return (f'--{boundary}\r\nContent-Disposition: form-data; name="scan"{fn}\r\n\r\n').encode() + payload + (
        f"\r\n--{boundary}--\r\n".encode() if close else b"")


def test_truncated_upload_is_400(server):
    srv, out = server
    body = _part("scan.json", b'{"a": 1}')
    assert _raw_post(srv, body, len(body) + 100) == 400
    assert not (out / "jobs").exists() or not any((out / "jobs").iterdir())


def test_missing_closing_boundary_is_400(server):
    srv, _ = server
    body = _part("scan.json", b'{"a": 1}', close=False)
    assert _raw_post(srv, body, len(body)) == 400


@pytest.mark.parametrize("filename,payload", [("scan.json", b""), (None, b"{}"), ("", b"{}")])
def test_empty_file_or_no_filename_is_400(server, filename, payload):
    srv, _ = server
    body = _part(filename, payload)
    assert _raw_post(srv, body, len(body)) == 400


def test_room_page_links_sim_dataset(server):
    srv, out = server
    meta = out / "datasets" / "hero_kitchen_sim" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"arrival_generator": "mujoco_sim", "total_episodes": 3,
                                                "arrival_physically_valid": True,
                                                "arrival_families": ["counter_to_sink", "open_drawer"]}))
    status, body = _get(srv, "/rooms/hero_kitchen")
    assert status == 200
    assert b"/download/hero_kitchen_sim" in body and b"mujoco_sim" in body and b"open_drawer" in body


def test_capture_app_upload_reaches_worker(server):
    """CAP-1: the iOS app's request (ios/ArrivalCapture/ArrivalCapture/ArrivalAPI.swift writeMultipart, scan part only)
    with a real CapturedRoom export is queued byte for byte and the worker reads it as RoomPlan."""
    from arrival.jobs.worker import load_upload, room_id_for

    srv, out = server
    boundary = "ArrivalBoundary-3F2504E0-4F89-11D3-9A0C-0305E82C3301"
    payload = (REPO / "tests" / "fixtures" / "roomplan" / "amal.json").read_bytes()
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="scan"; filename="kitchen_a.json"\r\n'
        "Content-Type: application/json\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request("POST", "/jobs", body=body,
                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                          "Accept": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    assert resp.status == 202
    job_dir = out / "jobs" / data["id"]
    assert (job_dir / "upload.bin").read_bytes() == payload
    status = json.loads((job_dir / "status.json").read_text())
    room_id = room_id_for(status["filename"], data["id"])
    assert room_id == f"kitchen_a_{data['id']}"
    bundle = load_upload(job_dir / "upload.bin", room_id)
    assert bundle["geometry_source"] == "roomplan"
    assert bundle["room_id"] == room_id
    assert bundle["walls"]


def test_serve_host_flag():
    from arrival.cli import build_parser

    assert build_parser().parse_args(["serve"]).host == "127.0.0.1"
    assert build_parser().parse_args(["serve", "--host", "0.0.0.0"]).host == "0.0.0.0"


def test_non_object_status_does_not_break_index(server):
    srv, out = server
    d = out / "jobs" / "job_deadbeef"
    d.mkdir(parents=True)
    (d / "status.json").write_text("[1]")
    status, _ = _get(srv, "/")
    assert status == 200


def test_job_download_streams_zip(server):
    import json as _json

    srv, out = server
    d = out / "jobs" / "job_cafe0000000000000000000000000001"
    d.mkdir(parents=True)
    payload = b"PK-not-really-but-bytes" * 100000
    (d / "dataset.zip").write_bytes(payload)
    (d / "status.json").write_text(_json.dumps({"id": "job_cafe0000000000000000000000000001", "status": "done", "room_id": "r", "zip": "x"}))
    status, body = _get(srv, "/jobs/job_cafe0000000000000000000000000001/download")
    assert status == 200 and body == payload


# ---- L4 hardening, L14 frames upload, L22 doctor ---------------------------------------------------------------

import base64  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

from arrival.web import hardening as hd  # noqa: E402

HERO_SCAN = (REPO / "scans" / "hero_kitchen" / "scan.json").read_bytes()
JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
TINY = b'{"a": 1}'


@pytest.fixture
def make(tmp_path):
    started = []

    def _make(**kw):
        out = tmp_path / "out"
        srv = make_server(port=0, out_dir=out, scans_dir=REPO / "scans", **kw)
        t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()
        started.append((srv, t))
        return srv, out

    yield _make
    for srv, t in started:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _multipart(parts, boundary="bnd7"):
    body = b""
    for name, filename, payload in parts:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n\r\n'
                 ).encode() + payload + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def _post(srv, parts, headers=None):
    body, h = _multipart(parts)
    h.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request("POST", "/jobs", body=body, headers=h)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, (json.loads(raw) if raw else {}), resp


def _fetch(srv, path, headers=None, method="GET"):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request(method, path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body, resp


def _cookie(resp) -> str:
    return resp.getheader("Set-Cookie").split(";", 1)[0]


def _job_dirs(out):
    root = out / "jobs"
    return sorted(root.iterdir()) if root.is_dir() else []


def _frames_zip(tmp_path, unit="m", mutate=None) -> bytes:
    from arrival.scan.rgbd import write_synthetic_bundle

    root = write_synthetic_bundle(tmp_path / f"bundle_{unit}", depth_unit=unit)
    if mutate:
        mutate(root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(root).as_posix())
    return buf.getvalue()


# -- job ids, paths, cross-user access

def test_job_ids_are_128_bit_random(make):
    srv, out = make(rate_limit=0)
    ids = []
    for _ in range(2):
        code, data, _ = _post(srv, [("scan", "scan.json", TINY)])
        assert code == 202 and JOB_ID.match(data["id"])
        ids.append(data["id"])
    assert ids[0] != ids[1]


@pytest.mark.parametrize("path", [
    "/jobs/..%2F..%2Fetc%2Fpasswd", "/jobs/job_1", "/jobs/JOB_DEADBEEF", "/jobs/job_deadbeef%00", "/jobs/..",
    "/jobs/job_deadbeef/../../secret", "/jobs/%2e%2e%2fsecret/download", "/previews/..%2f..%2fsecret.png",
    "/previews/%2e%2e", "/rooms/..%2f..", "/download/..%2f..%2fsecret", "/jobs/job_" + "f" * 33,
])
def test_traversal_and_malformed_ids_are_404(server, path):
    srv, out = server
    (out / "secret").mkdir(parents=True)
    (out / "secret" / "status.json").write_text(json.dumps({"id": "SECRETJOB", "status": "done", "room_id": "x", "zip": "y"}))
    status, body = _get(srv, path)
    assert status == 404 and b"SECRETJOB" not in body


def test_public_mode_lists_only_own_jobs_and_hides_paths(make):
    srv, out = make(public=True, rate_limit=0)
    code, a, resp_a = _post(srv, [("scan", "alice_kitchen.json", TINY)])
    assert code == 202
    cookie_a = _cookie(resp_a)
    code, b, resp_b = _post(srv, [("scan", "bob_kitchen.json", TINY)])
    cookie_b = _cookie(resp_b)
    assert cookie_a != cookie_b
    _, page_a, _ = _fetch(srv, "/", {"Cookie": cookie_a})
    _, page_b, _ = _fetch(srv, "/", {"Cookie": cookie_b})
    _, page_anon, _ = _fetch(srv, "/")
    assert a["id"].encode() in page_a and b["id"].encode() not in page_a
    assert b["id"].encode() in page_b and a["id"].encode() not in page_b
    assert a["id"].encode() not in page_anon and b["id"].encode() not in page_anon
    # a cookie value this server never issued sees nothing
    _, page_fake, _ = _fetch(srv, "/", {"Cookie": "arrival_client=" + "0" * 32})
    assert a["id"].encode() not in page_fake
    # the status API drops the owner hash and, in public mode, server paths
    path = out / "jobs" / a["id"] / "status.json"
    st = json.loads(path.read_text())
    assert st["owner"]
    st.update({"zip": "/srv/private/dataset.zip", "twin_dir": "/srv/private/twin", "dataset_dir": "/srv/private/ds"})
    path.write_text(json.dumps(st))
    _, body, _ = _fetch(srv, f"/jobs/{a['id']}", {"Cookie": cookie_a})
    got = json.loads(body)
    assert got["id"] == a["id"] and "owner" not in got and "/srv/private" not in body.decode()


def test_public_job_status_and_download_require_owner(make):
    srv, out = make(public=True, rate_limit=0)
    _, a, resp_a = _post(srv, [("scan", "alice.json", TINY)])
    _, b, resp_b = _post(srv, [("scan", "bob.json", TINY)])
    cookie_a, cookie_b = _cookie(resp_a), _cookie(resp_b)
    job = out / "jobs" / a["id"]
    status = json.loads((job / "status.json").read_text())
    status["status"] = "done"
    (job / "status.json").write_text(json.dumps(status))
    (job / "dataset.zip").write_bytes(b"private dataset")

    for path in (f"/jobs/{a['id']}", f"/jobs/{a['id']}/download"):
        code, _, _ = _fetch(srv, path, {"Cookie": cookie_a})
        assert code == 200
        for headers in ({}, {"Cookie": cookie_b}, {"Cookie": "arrival_client=" + "0" * 32}):
            code, body, _ = _fetch(srv, path, headers)
            assert code == 404 and b"private dataset" not in body


def test_public_uploaded_dataset_and_preview_require_owner(make):
    srv, out = make(public=True, rate_limit=0)
    _, a, resp_a = _post(srv, [("scan", "alice.json", TINY)])
    _, _, resp_b = _post(srv, [("scan", "bob.json", TINY)])
    cookie_a, cookie_b = _cookie(resp_a), _cookie(resp_b)
    room = f"alice_{a['id']}"
    job = out / "jobs" / a["id"]
    status = json.loads((job / "status.json").read_text())
    status["room_id"] = room
    (job / "status.json").write_text(json.dumps(status))
    dataset = out / "datasets" / f"{room}_sim"
    dataset.mkdir(parents=True)
    (dataset / "private.txt").write_bytes(b"private dataset")
    preview = out / "previews" / f"{room}_sim_contact.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"private preview")

    for path, expected in ((f"/download/{room}_sim", b"private dataset"),
                           (f"/previews/{preview.name}", b"private preview")):
        code, body, _ = _fetch(srv, path, {"Cookie": cookie_a})
        assert code == 200
        if path.startswith("/download/"):
            assert zipfile.ZipFile(io.BytesIO(body)).read(f"{room}_sim/private.txt") == expected
        else:
            assert expected in body
        for headers in ({}, {"Cookie": cookie_b}):
            code, body, _ = _fetch(srv, path, headers)
            assert code == 404 and expected not in body


def test_local_mode_lists_all_jobs(make):
    srv, _ = make(rate_limit=0)
    _, a, _ = _post(srv, [("scan", "a.json", TINY)])
    _, page, _ = _fetch(srv, "/")
    assert a["id"].encode() in page


# -- upload type, content and filename checks

@pytest.mark.parametrize("name,payload", [
    ("scan.exe", TINY), ("scan", TINY), ("scan.json", b"MZ\x90\x00binary"),
    ("scan.json", b"<html></html>"), ("scan.json.zip", TINY), ("scan.json", b"[1, 2]"),
])
def test_wrong_type_or_content_is_415(make, name, payload):
    srv, out = make()
    code, data, _ = _post(srv, [("scan", name, payload)])
    assert code == 415 and "error" in data
    assert _job_dirs(out) == []


def test_filename_is_sanitised_everywhere(make):
    srv, out = make(rate_limit=0)
    code, data, _ = _post(srv, [("scan", "../../etc/<script>alert(1)</script>.json", TINY)])
    assert code == 202
    job = out / "jobs" / data["id"]
    assert sorted(p.name for p in job.iterdir()) == ["status.json", "upload.bin"]
    assert not (out / "etc").exists() and not (out.parent / "etc").exists()
    st = json.loads((job / "status.json").read_text())
    assert "/" not in st["filename"] and "<" not in st["filename"] and st["filename"].endswith(".json")
    _, page, _ = _fetch(srv, "/")
    assert b"<script>alert(1)" not in page


@pytest.mark.parametrize("raw,expected", [
    ("kitchen_a.json", "kitchen_a.json"), ("..\\..\\win\\evil.json", "evil.json"), ("/abs/path/x.json", "x.json"),
    ("My Kitchen (2).json", "My Kitchen (2).json"), ("", "upload.bin"), (None, "upload.bin"), ("....", "upload.bin"),
    ("a\x00b\nc.json", "abc.json"), ("<b>x</b>.json", "b_.json"), ("..hidden.json", "hidden.json"),
])
def test_safe_filename(raw, expected):
    assert hd.safe_filename(raw) == expected


def test_safe_filename_caps_length_and_keeps_extension():
    out = hd.safe_filename("a" * 500 + ".json")
    assert len(out) <= 80 and out.endswith(".json")


# -- limits: size, rate, queue, concurrency

def test_max_upload_mb_is_enforced(make):
    srv, out = make(max_upload_mb=0.001)  # 1048 bytes
    assert _raw_post(srv, b"", 5000) == 413
    assert _job_dirs(out) == []
    assert _post(srv, [("scan", "scan.json", TINY)])[0] == 202  # under the limit still works


def test_rate_limit_returns_429_with_retry_after(make):
    srv, out = make(rate_limit=2)
    assert [_post(srv, [("scan", "s.json", TINY)])[0] for _ in range(2)] == [202, 202]
    code, data, resp = _post(srv, [("scan", "s.json", TINY)])
    assert code == 429 and int(resp.getheader("Retry-After")) >= 1 and "error" in data
    assert len(_job_dirs(out)) == 2


def test_rate_limit_ignores_forwarded_for_unless_behind_proxy(make):
    srv, _ = make(rate_limit=1)
    assert _post(srv, [("scan", "s.json", TINY)], {"X-Forwarded-For": "203.0.113.1"})[0] == 202
    assert _post(srv, [("scan", "s.json", TINY)], {"X-Forwarded-For": "203.0.113.2"})[0] == 429
    proxy, _ = make(rate_limit=1, behind_proxy=True)
    assert _post(proxy, [("scan", "s.json", TINY)], {"X-Forwarded-For": "203.0.113.1"})[0] == 202
    assert _post(proxy, [("scan", "s.json", TINY)], {"X-Forwarded-For": "203.0.113.2"})[0] == 202
    assert _post(proxy, [("scan", "s.json", TINY)], {"X-Forwarded-For": "198.51.100.9, 203.0.113.2"})[0] == 429


def test_rate_limiter_window_and_bound():
    now = [0.0]
    rl = hd.RateLimiter(2, window=10, clock=lambda: now[0], max_keys=3)
    assert rl.check("a") == (True, 0) and rl.check("a") == (True, 0)
    ok, wait = rl.check("a")
    assert not ok and 1 <= wait <= 10
    assert rl.check("b")[0]  # keys are independent
    now[0] = 10.5
    assert rl.check("a")[0]  # the window moved on
    for k in "cdefg":
        rl.check(k)
    assert len(rl._hits) <= 3  # the table stays bounded
    assert hd.RateLimiter(0).check("x") == (True, 0)  # 0 turns it off


def test_queue_cap_returns_503_until_a_job_finishes(make):
    srv, out = make(max_queued=1, rate_limit=0)
    code, first, _ = _post(srv, [("scan", "a.json", TINY)])
    assert code == 202
    code, data, resp = _post(srv, [("scan", "b.json", TINY)])
    assert code == 503 and resp.getheader("Retry-After") and len(_job_dirs(out)) == 1
    path = out / "jobs" / first["id"] / "status.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "status": "done"}))
    assert _post(srv, [("scan", "c.json", TINY)])[0] == 202


def test_concurrent_upload_cap_returns_503(make):
    srv, out = make(max_concurrent_uploads=1, rate_limit=0)
    assert srv.heavy_slots.acquire(blocking=False)
    try:
        code, _, resp = _post(srv, [("scan", "a.json", TINY)])
    finally:
        srv.heavy_slots.release()
    assert code == 503 and _job_dirs(out) == []
    assert _post(srv, [("scan", "a.json", TINY)])[0] == 202


def test_low_disk_is_507(make, monkeypatch):
    srv, out = make(rate_limit=0)
    monkeypatch.setattr("arrival.web.app.Handler._free_disk", lambda self: 1024)
    code, data, _ = _post(srv, [("scan", "a.json", TINY)])
    assert code == 507 and _job_dirs(out) == []


# -- headers, error pages, logs

def test_security_headers_on_html_json_and_errors(server):
    srv, out = server
    for path in ("/", "/rooms/hero_kitchen", "/nope", "/jobs/job_" + "0" * 32, "/jobs/xx"):
        status, body, resp = _fetch(srv, path)
        h = {k.lower(): v for k, v in resp.getheaders()}
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY", path
        assert h["referrer-policy"] == "no-referrer" and "frame-ancestors 'none'" in h["content-security-policy"], path
        assert "unsafe-inline" not in h["content-security-policy"] and h["cache-control"] == "no-store", path
        assert "strict-transport-security" not in h and h["server"] == "Arrival", path


def test_csp_hashes_match_the_inline_script_and_style(server):
    srv, _ = server
    _, body, resp = _fetch(srv, "/")
    csp = resp.getheader("Content-Security-Policy")
    page = body.decode()
    for tag in ("script", "style"):
        inline = re.search(rf"<{tag}>(.*?)</{tag}>", page, re.S).group(1)
        digest = base64.b64encode(hashlib.sha256(inline.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp, tag
    assert "style=" not in page and "onclick=" not in page  # nothing the hash policy would block


def test_hsts_and_secure_cookie_only_behind_https_proxy(make):
    srv, _ = make(behind_proxy=True, rate_limit=0)
    code, _, resp = _post(srv, [("scan", "a.json", TINY)], {"X-Forwarded-Proto": "https"})
    assert code == 202
    assert "max-age" in resp.getheader("Strict-Transport-Security")
    assert "Secure" in resp.getheader("Set-Cookie") and "HttpOnly" in resp.getheader("Set-Cookie")
    _, _, resp = _post(srv, [("scan", "a.json", TINY)], {"X-Forwarded-Proto": "http"})
    assert resp.getheader("Strict-Transport-Security") is None and "Secure" not in resp.getheader("Set-Cookie")
    plain, _ = make(rate_limit=0)  # without --behind-proxy the header is not trusted
    _, _, resp = _fetch(plain, "/", {"X-Forwarded-Proto": "https"})
    assert resp.getheader("Strict-Transport-Security") is None


def test_crash_gives_generic_500_and_redacted_log(make, monkeypatch, capsys):
    secret = "sk-test-secret-value-1234"
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", secret)
    srv, _ = make()

    def boom(self):
        raise RuntimeError(f"boom with {secret}")

    monkeypatch.setattr("arrival.web.app.Handler._index", boom)
    status, body, resp = _fetch(srv, "/")
    assert status == 500 and b"Internal server error" in body
    assert b"boom" not in body and b"Traceback" not in body and secret.encode() not in body
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    err = capsys.readouterr().err
    assert secret not in err and "RuntimeError" in err


def test_error_pages_are_html_for_browsers_and_json_for_the_api(server):
    srv, _ = server
    status, body, resp = _fetch(srv, "/nope", {"Accept": "text/html"})
    assert status == 404 and resp.getheader("Content-Type").startswith("text/html") and b"Back to rooms" in body
    status, body, resp = _fetch(srv, "/jobs/job_" + "0" * 32, {"Accept": "application/json"})
    assert status == 404 and json.loads(body) == {"error": "job not found"}


def test_unsupported_methods_are_405_with_allow(server):
    srv, _ = server
    for method in ("PUT", "PATCH", "OPTIONS"):
        status, _, resp = _fetch(srv, "/jobs", method=method)
        assert status == 405 and "POST" in resp.getheader("Allow"), method


@pytest.mark.parametrize("raw,code", [(b"GET / HTTP/9.9\r\n\r\n", b" 505 "), (b"GET /" + b"a" * 70000 + b" HTTP/1.1\r\n\r\n", b" 414 ")],
                         ids=["unsupported_version", "long_request"])
def test_protocol_errors_get_our_error_page_not_a_traceback(server, raw, code):
    srv, _ = server
    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
    s.sendall(raw)
    data = b""
    while chunk := s.recv(65536):
        data += chunk
    s.close()
    head = data.split(b"\r\n\r\n", 1)[0]
    assert data.startswith(b"HTTP/") and code in head.split(b"\r\n")[0] + b" "
    assert b"X-Content-Type-Options: nosniff" in head and b"Server: Arrival\r\n" in head
    assert b"Traceback" not in data and b"Python/" not in head and b"Error code" not in data


def test_redact_masks_env_secrets_bearer_and_key_pairs():
    env = {"TAVILY_API_KEY": "tvly-abcdef123456", "PATH": "/usr/bin", "MY_PASSWORD": "hunter2hunter2", "SHORT_KEY": "ab"}
    text = ("call failed for tvly-abcdef123456 with Authorization: Bearer eyJhbGciOi.payload.sig and "
            "url=https://x/?api_key=abc123xyz&b=1 pw hunter2hunter2 in /usr/bin")
    out = hd.redact(text, env)
    for leaked in ("tvly-abcdef123456", "eyJhbGciOi", "abc123xyz", "hunter2hunter2"):
        assert leaked not in out
    assert "/usr/bin" in out and "b=1" in out


def test_index_form_shows_limit_and_frames_input(make):
    srv, _ = make(max_upload_mb=12)
    _, body, _ = _fetch(srv, "/")
    assert b"up to 12 MB" in body and b"name='frames'" in body and b"frames_manifest.json" in body


# -- L14: RGB-D frames through the upload path

@pytest.mark.parametrize("unit", ["m", "mm"])
def test_frames_upload_is_validated_stored_and_scan_still_reaches_the_worker(make, tmp_path, unit):
    from arrival.jobs.worker import claim_next, load_upload
    from arrival.scan import rgbd

    srv, out = make(rate_limit=0)
    blob = _frames_zip(tmp_path, unit)
    code, data, _ = _post(srv, [("scan", "hero.json", HERO_SCAN), ("frames", "frames.zip", blob)])
    assert code == 202, data
    assert data["frames"]["count"] == 2 and data["frames"]["train"] == 1 and data["frames"]["heldout"] == 1
    assert data["frames"]["consumed"] is False
    job = out / "jobs" / data["id"]
    assert (job / "upload.bin").read_bytes() == HERO_SCAN
    assert [p.name for p in (out / "jobs").iterdir() if p.name.startswith(".")] == []  # scratch folder gone
    m = rgbd.load_manifest(job / "frames")  # the stored copy passes the CAP-2a loader
    assert [f.id for f in m.frames] == ["f000", "f001"] and m.depth_unit == unit
    src = rgbd.load_manifest(tmp_path / f"bundle_{unit}")
    for stored, orig in zip(m.frames, src.frames):
        assert stored.rgb.read_bytes() == orig.rgb.read_bytes() and stored.depth.read_bytes() == orig.depth.read_bytes()
    f0 = m.frames[0]
    pts = rgbd.unproject(rgbd.load_depth(m, f0), f0.K, f0.camera_to_room)
    floor = pts[pts[:, 0] < 2.99]  # the wall is x = 3; everything else this camera sees is the z = 0 floor
    assert len(floor) > 100 and np.abs(floor[:, 2]).max() < 2e-3  # stored depth still lands on the floor plane
    _, body, _ = _fetch(srv, f"/jobs/{data['id']}")
    st = json.loads(body)
    assert st["frames"]["stored"] is True and st["frames"]["consumed"] is False
    job_dir, claimed = claim_next(out / "jobs")  # the worker claims it with the frames fields intact
    assert claimed["frames"]["count"] == 2
    assert load_upload(job_dir / "upload.bin", "r")["geometry_source"] == "authored_sample"
    _, page, _ = _fetch(srv, "/")
    assert b"RGB-D 2 frames, comparison pending" in page


def _edit_manifest(fn):
    def mutate(root):
        p = root / "frames_manifest.json"
        m = json.loads(p.read_text())
        fn(m, root)
        p.write_text(json.dumps(m))
    return mutate


def _same_times(m, root):
    m["frames"][1]["timestamp"] = m["frames"][0]["timestamp"]


def _drop_file(m, root):
    (root / m["frames"][0]["rgb"]).unlink()


def _wrong_rgb_size(m, root):
    from PIL import Image

    Image.new("RGB", (10, 10)).save(root / m["frames"][0]["rgb"])


def _corrupt_rgb(m, root):
    (root / m["frames"][0]["rgb"]).write_bytes(b"not a png at all")


def _grayscale_rgb(m, root):
    from PIL import Image

    path = root / m["frames"][0]["rgb"]
    with Image.open(path) as im:
        im.convert("L").save(path)


def _wrong_depth_shape(m, root):
    np.save(root / m["frames"][0]["depth"], np.zeros((3, 3), np.float32))


def _wrong_confidence_size(m, root):
    from PIL import Image

    Image.new("L", (3, 3), 2).save(root / "confidence.png")
    m["frames"][0]["confidence"] = "confidence.png"

def _wrong_depth_dtype(m, root):
    path = root / m["frames"][0]["depth"]
    np.save(path, np.load(path).astype(np.float64))

def _invalid_confidence_level(m, root):
    from PIL import Image

    Image.new("L", tuple(m["depth_size"]), 255).save(root / "confidence.png")
    m["frames"][0]["confidence"] = "confidence.png"


def _invalid_confidence_mode(m, root):
    from PIL import Image

    Image.new("RGB", tuple(m["depth_size"]), (2, 2, 2)).save(root / "confidence.png")
    m["frames"][0]["confidence"] = "confidence.png"


def _pickled_depth(m, root):
    np.save(root / m["frames"][0]["depth"], np.array([{"x": 1}], dtype=object), allow_pickle=True)


def _bad_axes(m, root):
    m["axes"]["camera"] = "opengl"


def _list_depth_unit(m, root):
    m["depth_unit"] = ["m"]


@pytest.mark.parametrize("mutate,code", [
    (_same_times, "non_monotonic_timestamps"), (_drop_file, "missing_file"), (_wrong_rgb_size, "bad_size"),
    (_corrupt_rgb, "bad_image"), (_wrong_depth_shape, "depth_shape"),
    (_wrong_confidence_size, "bad_size"), (_pickled_depth, "bad_depth"),

    (_wrong_depth_dtype, "depth_shape"), (_pickled_depth, "bad_depth"),

    (_corrupt_rgb, "bad_image"), (_grayscale_rgb, "bad_image"),
    (_wrong_depth_shape, "depth_shape"), (_pickled_depth, "bad_depth"),

    (_corrupt_rgb, "bad_image"), (_wrong_depth_shape, "depth_shape"), (_pickled_depth, "bad_depth"),
    (_invalid_confidence_level, "bad_confidence"), (_invalid_confidence_mode, "bad_confidence"),
    (_bad_axes, "axis_convention"), (_list_depth_unit, "depth_unit"),
])
def test_bad_frames_bundle_is_rejected_and_nothing_is_queued(make, tmp_path, mutate, code):
    srv, out = make(rate_limit=0)
    blob = _frames_zip(tmp_path, "m", _edit_manifest(mutate))
    status, data, _ = _post(srv, [("scan", "hero.json", HERO_SCAN), ("frames", "frames.zip", blob)])
    assert status in (400, 422) and data["code"] == code and data["field"] == "frames", data
    assert _job_dirs(out) == []  # no job, no scratch folder


def _zip_of(entries, symlink=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in entries.items():
            zi = zipfile.ZipInfo(name)
            zi.filename = name  # ZipInfo replaces backslashes on Windows; keep the hostile input.
            zf.writestr(zi, payload)
        if symlink:
            zi = zipfile.ZipInfo(symlink)
            zi.external_attr = 0o120777 << 16
            zf.writestr(zi, "/etc/passwd")
    return buf.getvalue()


@pytest.mark.parametrize("entries,symlink", [
    ({"../evil.png": b"x"}, None), ({"/abs.png": b"x"}, None), ({"a/../../evil.png": b"x"}, None),
    ({"rgb\\evil.png": b"x"}, None), ({"run.sh": b"#!/bin/sh"}, None),
    ({"x.PNG": b"a", "x.png": b"b"}, None), ({"frames_manifest.json": b"{}"}, "link.png"),
])
def test_unsafe_zip_members_are_400(make, tmp_path, entries, symlink):
    srv, out = make(rate_limit=0)
    status, data, _ = _post(srv, [("scan", "hero.json", HERO_SCAN), ("frames", "f.zip", _zip_of(entries, symlink))])
    assert status == 400 and data["field"] == "frames", data
    assert _job_dirs(out) == []
    assert not (out / "evil.png").exists() and not (tmp_path / "evil.png").exists()


def test_frames_not_a_zip_or_wrong_name_is_rejected(make):
    srv, out = make(rate_limit=0)
    assert _post(srv, [("scan", "s.json", HERO_SCAN), ("frames", "f.zip", b"PK\x03\x04garbagegarbage")])[0] == 400
    assert _post(srv, [("scan", "s.json", HERO_SCAN), ("frames", "f.zip", b"hello")])[0] == 415
    assert _post(srv, [("scan", "s.json", HERO_SCAN), ("frames", "f.tar", b"PK\x03\x04")])[0] == 415
    assert _post(srv, [("scan", "s.json", HERO_SCAN), ("frames", "f.zip", b"")])[0] == 400
    assert _job_dirs(out) == []


def test_frames_input_left_empty_by_a_browser_is_ignored(make):
    # an unselected <input type=file> is still sent: filename="" and no bytes. The scan alone must queue.
    srv, out = make(rate_limit=0)
    code, body, _ = _post(srv, [("scan", "s.json", HERO_SCAN), ("frames", "", b"")])
    assert code == 202, body
    assert "frames" not in json.loads((out / "jobs" / body["id"] / "status.json").read_text())


def test_frames_zip_bomb_is_stopped_by_the_streamed_size_cap(make, monkeypatch):
    from arrival.web import frames as frames_mod

    monkeypatch.setattr(frames_mod, "MAX_MEMBER_BYTES", 10_000)
    srv, out = make(rate_limit=0)
    bomb = _zip_of({"rgb/big.png": b"\0" * 5_000_000})  # stored uncompressed here, but the cap reads the stream
    status, data, _ = _post(srv, [("scan", "s.json", HERO_SCAN), ("frames", "f.zip", bomb)])
    assert status == 413 and data["code"] == "too_large"
    assert _job_dirs(out) == []


def test_frames_member_count_cap(make, monkeypatch):
    from arrival.web import frames as frames_mod

    monkeypatch.setattr(frames_mod, "MAX_MEMBERS", 3)
    srv, out = make(rate_limit=0)
    z = _zip_of({f"rgb/{i}.png": b"x" for i in range(5)})
    status, data, _ = _post(srv, [("scan", "s.json", HERO_SCAN), ("frames", "f.zip", z)])
    assert status == 400 and data["code"] == "too_many_files" and _job_dirs(out) == []


# -- CLI flags and the doctor

def test_serve_hardening_flags_parse_and_validate():
    from arrival.cli import EXIT_USAGE, build_parser, main

    a = build_parser().parse_args(["serve", "--max-upload-mb", "5", "--rate-limit", "3", "--max-queued", "2",
                                   "--public", "--behind-proxy"])
    assert (a.max_upload_mb, a.rate_limit, a.max_queued, a.public, a.behind_proxy) == (5, 3, 2, True, True)
    d = build_parser().parse_args(["serve"])
    assert (d.max_upload_mb, d.rate_limit, d.max_queued, d.public, d.behind_proxy) == (200, 10, 50, False, False)
    assert main(["serve", "--max-upload-mb", "0"]) == EXIT_USAGE
    assert main(["serve", "--rate-limit", "-1"]) == EXIT_USAGE


def test_doctor_reports_keys_without_values(tmp_path, monkeypatch, capsys):
    from arrival import doctor
    from arrival.cli import main

    env_secret, file_secret = "sk-env-secret-AAAA1111", "tvly-file-secret-BBBB2222"
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", env_secret)
    monkeypatch.setenv("GR00T_POLICY_URL", "http://10.9.8.7:5555")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("COSMOS_URL", raising=False)
    envfile = tmp_path / ".env"
    envfile.write_text(f"# comment\nTAVILY_API_KEY={file_secret}\nCOSMOS_URL=\nexport OTHER=zzz\n")
    checks = doctor.run_checks(tmp_path / "out", env_files=[envfile])
    by = {c.name: c for c in checks}
    assert by["TOKEN_FACTORY_API_KEY"].detail == "set in the environment"
    assert by["TAVILY_API_KEY"].detail.startswith("set in .env")
    assert by["GR00T_POLICY_URL"].status == "ok" and by["COSMOS_URL"].detail.startswith("not set")
    assert by["COSMOS_URL"].status == "info"  # optional keys never fail the doctor
    for text in (doctor.render(checks), doctor.render_json(checks)):
        for leaked in (env_secret, file_secret, "10.9.8.7", "5555"):
            assert leaked not in text
    monkeypatch.chdir(tmp_path)
    assert main(["doctor", "--out", str(tmp_path / "out")]) in (0, 1)
    out = capsys.readouterr().out
    assert env_secret not in out and "10.9.8.7" not in out


def test_doctor_json_cli_and_exit_codes(tmp_path, monkeypatch, capsys):
    import importlib.util

    from arrival.cli import EXIT_INVALID, main

    assert main(["doctor", "--json", "--out", str(tmp_path / "out")]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True
    names = {c["name"] for c in data["checks"]}
    assert {"python", "numpy", "mujoco", "ffmpeg", "disk", "TOKEN_FACTORY_API_KEY"} <= names
    assert all(c["status"] in ("ok", "warn", "fail", "info") for c in data["checks"])
    real = importlib.util.find_spec  # a missing core dependency is a failure with exit code 1
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "pyarrow" else real(name, *a, **k))
    assert main(["doctor", "--out", str(tmp_path / "out")]) == EXIT_INVALID
    assert "FAIL" in capsys.readouterr().out


def test_doctor_disk_and_mujoco_levels(tmp_path, monkeypatch):
    from collections import namedtuple

    from arrival import doctor

    Usage = namedtuple("Usage", "total used free")
    gib = 1024 ** 3
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(100 * gib, 0, int(0.1 * gib)))
    assert doctor.check_disk(tmp_path).status == "fail"
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(100 * gib, 0, int(1 * gib)))
    assert doctor.check_disk(tmp_path).status == "warn"
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(100 * gib, 0, 50 * gib))
    assert doctor.check_disk(tmp_path / "does" / "not" / "exist").status == "ok"  # nearest existing parent
    monkeypatch.setitem(sys.modules, "mujoco", None)  # makes `import mujoco` raise ImportError
    c = doctor.check_mujoco()
    assert c.status == "warn" and "pip install" in c.detail
    assert doctor.exit_code([c]) == 0 and doctor.exit_code([doctor.Check("x", "fail", "")]) == 1


# ---- DELETE /jobs/<id>: cancel or delete, owner only ---------------------------------------------------------------

def _set_status(out, jid, **kw):
    path = out / "jobs" / jid / "status.json"
    st = json.loads(path.read_text())
    st.update(kw)
    path.write_text(json.dumps(st))


def test_delete_job_owner_only_and_404_for_everyone_else(make):
    srv, out = make(rate_limit=0)
    _, a, resp_a = _post(srv, [("scan", "a.json", TINY)])
    cookie_a = _cookie(resp_a)
    _, b, resp_b = _post(srv, [("scan", "b.json", TINY)])
    cookie_b = _cookie(resp_b)
    jid = a["id"]
    _set_status(out, jid, status="done", finished_at=1)
    (out / "jobs" / jid / "dataset.zip").write_bytes(b"PK")
    unknown = "/jobs/job_" + "0" * 16
    for headers in ({}, {"Cookie": cookie_b}, {"Cookie": "arrival_client=" + "0" * 32}):
        code, body, _ = _fetch(srv, f"/jobs/{jid}", headers, method="DELETE")
        assert code == 404
        assert body == _fetch(srv, unknown, headers, method="DELETE")[1]  # same answer as an id that does not exist
    assert _fetch(srv, unknown, {"Cookie": cookie_a}, method="DELETE")[0] == 404
    assert (out / "jobs" / jid).is_dir()
    code, body, _ = _fetch(srv, f"/jobs/{jid}", {"Cookie": cookie_a}, method="DELETE")
    assert code == 200 and json.loads(body) == {"id": jid, "result": "deleted"}
    assert not (out / "jobs" / jid).exists() and (out / "jobs" / b["id"]).is_dir()


def test_delete_cancels_queued_and_running_jobs(make):
    srv, out = make(rate_limit=0)
    _, a, resp = _post(srv, [("scan", "a.json", TINY)])
    cookie = {"Cookie": _cookie(resp)}
    code, body, _ = _fetch(srv, f"/jobs/{a['id']}", cookie, method="DELETE")
    assert code == 200 and json.loads(body)["result"] == "cancelled"
    assert json.loads((out / "jobs" / a["id"] / "status.json").read_text())["status"] == "cancelled"
    _, b, resp = _post(srv, [("scan", "b.json", TINY)])
    cookie = {"Cookie": _cookie(resp)}
    _set_status(out, b["id"], status="running", stage="sim")
    code, body, _ = _fetch(srv, f"/jobs/{b['id']}", cookie, method="DELETE")
    assert code == 202 and json.loads(body)["result"] == "cancelling"
    assert (out / "jobs" / b["id"] / "cancel.requested").exists() and (out / "jobs" / b["id"]).is_dir()


def test_index_shows_retention_note_buttons_for_owner_only_and_progress(make, monkeypatch):
    monkeypatch.delenv("ARRIVAL_RETENTION_HOURS", raising=False)
    srv, out = make(rate_limit=0)
    _, a, resp = _post(srv, [("scan", "a.json", TINY)])
    _, page, _ = _fetch(srv, "/", {"Cookie": _cookie(resp)})
    assert b"data-job-action='cancel'" in page and b"kept until the person who uploaded" in page
    _, page, _ = _fetch(srv, "/")
    assert b"data-job-action='" not in page
    monkeypatch.setenv("ARRIVAL_RETENTION_HOURS", "24")
    _, page, _ = _fetch(srv, "/")
    assert b"deleted 24 hours after" in page
    _set_status(out, a["id"], status="running", stage="sim", progress={"done": 3, "total": 8})
    _, page, _ = _fetch(srv, "/")
    assert b"sim: 3/8 episodes" in page
