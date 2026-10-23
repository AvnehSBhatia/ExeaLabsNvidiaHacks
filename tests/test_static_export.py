"""arrival export-site: the local page pre-rendered to static files with working relative links."""

import json
import re
import zipfile
from pathlib import Path

import pytest

from arrival import cli
from arrival.web import static_export
from arrival.web.static_export import ExportError, export_site, link_check

REPO = Path(__file__).resolve().parents[1]
PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000001"
                    "e221bc330000000049454e44ae426082")


@pytest.fixture
def out(tmp_path):
    """A minimal out/ with one generated dataset, one report, and previews for hero_kitchen."""
    o = tmp_path / "out"
    meta = o / "datasets" / "hero_kitchen_sim" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 3, "arrival_successes": 3, "arrival_attempts": 4,
                                                "arrival_physically_valid": True, "arrival_generator": "mujoco_scripted_ik",
                                                "fps": 20}))
    (o / "datasets" / "hero_kitchen_sim" / "data.bin").write_bytes(b"x" * 100)
    (o / "reports").mkdir()
    (o / "reports" / "hero_kitchen_sim.json").write_text(json.dumps({"attempts": 4, "successes": 3, "human_contacts": 0}))
    (o / "previews").mkdir()
    (o / "previews" / "hero_kitchen_sim_contact.png").write_bytes(PNG)
    (o / "previews" / "hero_kitchen_sim_overview.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return o


def hrefs(html):
    return [m[2] for m in static_export.ATTR_RE.findall(html)]


def test_export_is_self_contained_and_link_checked(out, tmp_path):
    res = export_site(out, REPO / "scans", tmp_path / "site", log=lambda *_: None)
    dest = tmp_path / "site"
    assert res["rooms"] == ["hero_kitchen"] and res["pages"] == 2 and res["previews"] == 2
    index = (dest / "index.html").read_text(encoding="utf-8")
    room = (dest / "rooms" / "hero_kitchen" / "index.html").read_text(encoding="utf-8")
    for page in (index, room):
        assert "<form" not in page and "<script" not in page  # no backend to post to, no page script
        assert "Static snapshot" in page
        assert not [u for u in hrefs(page) if u.startswith("/")], "absolute links break under a Pages prefix"
    assert "rooms/hero_kitchen/index.html" in hrefs(index)
    assert "previews/hero_kitchen_sim_contact.png" in hrefs(index)
    assert "../../index.html" in hrefs(room) and "../../previews/hero_kitchen_sim_overview.mp4" in hrefs(room)
    assert (dest / "previews" / "hero_kitchen_sim_contact.png").read_bytes() == PNG
    assert "Live upload is offline" in index and "Jobs" not in index
    assert link_check(dest) == []
    assert (dest / ".nojekyll").is_file()


def test_data_json_matches_what_is_on_disk(out, tmp_path):
    export_site(out, REPO / "scans", tmp_path / "site", log=lambda *_: None)
    data = json.loads((tmp_path / "site" / "data" / "hero_kitchen.json").read_text(encoding="utf-8"))
    assert data["datasets"]["hero_kitchen_sim"]["total_episodes"] == 3
    assert data["datasets"]["hero_kitchen_sim"]["arrival_physically_valid"] is True
    assert data["report"]["successes"] == 3 and data["fixtures"] and data["geometry_source"]
    idx = json.loads((tmp_path / "site" / "data" / "index.json").read_text(encoding="utf-8"))
    assert idx["rooms"] == ["hero_kitchen"] and idx["zips_included"] == []


def test_zips_are_opt_in(out, tmp_path):
    export_site(out, REPO / "scans", tmp_path / "plain", log=lambda *_: None)
    plain = (tmp_path / "plain" / "index.html").read_text(encoding="utf-8")
    assert "downloads/" not in plain and "#removed" not in plain and "Sim zip" not in plain
    assert not (tmp_path / "plain" / "downloads").exists()
    export_site(out, REPO / "scans", tmp_path / "withzip", zips=True, log=lambda *_: None)
    z = tmp_path / "withzip" / "downloads" / "hero_kitchen_sim.zip"
    assert "downloads/hero_kitchen_sim.zip" in hrefs((tmp_path / "withzip" / "index.html").read_text(encoding="utf-8"))
    assert any(n.endswith("meta/info.json") for n in zipfile.ZipFile(z).namelist())
    assert link_check(tmp_path / "withzip") == []


def test_room_without_dataset_still_exports_and_says_so(tmp_path):
    logs = []
    res = export_site(tmp_path / "empty_out", REPO / "scans", tmp_path / "site", log=logs.append)
    assert res["rooms_without_dataset"] == ["hero_kitchen"]
    assert "not generated" in (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert logs and "hero_kitchen" in logs[0]


def test_link_check_finds_broken_links(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "ok.png").write_bytes(PNG)
    (tmp_path / "a" / "index.html").write_text(
        "<a href='ok.png'>x</a><a href='../missing.html'>y</a><img src='gone.png'>"
        "<a href='https://example.com/'>ext</a><a href='#top'>t</a><a href='../escape.png'>e</a>",
        encoding="utf-8",
    )
    (tmp_path / "escape.png").write_bytes(PNG)  # a file outside dest must not count as a hit
    dest = tmp_path / "a"
    broken = link_check(dest)
    assert sorted(b.split(": ")[1] for b in broken) == ["../escape.png", "../missing.html", "gone.png"]


def test_export_replaces_an_old_export_and_refuses_dangerous_dest(out, tmp_path):
    dest = tmp_path / "site"
    export_site(out, REPO / "scans", dest, log=lambda *_: None)
    (dest / "stale.txt").write_text("old")
    export_site(out, REPO / "scans", dest, log=lambda *_: None)
    assert not (dest / "stale.txt").exists() and (dest / "index.html").is_file()
    with pytest.raises(ExportError, match="refusing to export over"):
        export_site(out, REPO / "scans", out, log=lambda *_: None)


def test_export_refuses_a_folder_that_is_not_an_earlier_export(out, tmp_path):
    datasets = out / "datasets"
    with pytest.raises(ExportError, match="not an earlier export"):
        export_site(out, REPO / "scans", datasets, log=lambda *_: None)
    assert (datasets / "hero_kitchen_sim" / "meta" / "info.json").is_file()
    assert not (out / "datasets.tmp").exists()
    mine = tmp_path / "docs"
    mine.mkdir()
    (mine / "notes.md").write_text("keep me")
    assert cli.main(["export-site", "--out", str(out), "--scans", str(REPO / "scans"), "--dest", str(mine)]) == 1
    assert (mine / "notes.md").read_text() == "keep me"
    empty = tmp_path / "empty"
    empty.mkdir()
    export_site(out, REPO / "scans", empty, log=lambda *_: None)
    assert (empty / "index.html").is_file()


def test_export_refuses_file_dest(out, tmp_path):
    f = tmp_path / "existing_file.html"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(ExportError, match="existing file"):
        export_site(out, REPO / "scans", f, log=lambda *_: None)
    assert not (tmp_path / "existing_file.html.tmp").exists()


def test_export_refuses_ancestor_dest(out, tmp_path):
    with pytest.raises(ExportError, match="ancestor directory"):
        export_site(out, REPO / "scans", tmp_path, log=lambda *_: None)


def test_failed_link_check_leaves_no_half_export(out, tmp_path, monkeypatch):
    monkeypatch.setattr(static_export, "_rewrite", lambda html, depth, zips: html)  # leave absolute links in
    with pytest.raises(ExportError, match="broken links"):
        export_site(out, REPO / "scans", tmp_path / "site", log=lambda *_: None)
    assert not (tmp_path / "site").exists() and not (tmp_path / "site.tmp").exists()


def test_failed_get_leaves_no_tmp_directory(out, tmp_path, monkeypatch):
    monkeypatch.setattr(static_export, "_get", lambda *_, **__: (_ for _ in ()).throw(ExportError("network failed")))
    with pytest.raises(ExportError, match="network failed"):
        export_site(out, REPO / "scans", tmp_path / "site", log=lambda *_: None)
    assert not (tmp_path / "site").exists() and not (tmp_path / "site.tmp").exists()


def test_cli_export_site(out, tmp_path, capsys):
    rc = cli.main(["export-site", "--out", str(out), "--scans", str(REPO / "scans"), "--dest", str(tmp_path / "s")])
    assert rc == cli.EXIT_OK and (tmp_path / "s" / "index.html").is_file()
    assert re.search(r"static site: .* \(2 pages, 2 previews, 0 zips, rooms: hero_kitchen\)", capsys.readouterr().out)


def test_cli_export_site_error(out, tmp_path, capsys):
    f = tmp_path / "a_file.txt"
    f.write_text("content", encoding="utf-8")
    rc = cli.main(["export-site", "--out", str(out), "--scans", str(REPO / "scans"), "--dest", str(f)])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_INVALID
    assert "refusing to export over existing file" in err
