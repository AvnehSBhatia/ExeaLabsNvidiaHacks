"""L5: shardable, resumable sim collection (arrival/jobs/shards.py) and merge-shards."""

import json
import shutil

import numpy as np
import pytest

from arrival import cli
from arrival.export.validate import validate
from arrival.jobs import shards
from arrival.jobs.shards import Journal, ShardError, merge_shards, parse_shard, shard_range
from arrival.sim.export import SimDatasetWriter

CAMS = ("ego_view", "wrist")


# ---- planning ----


def test_parse_shard():
    assert parse_shard("0/4") == (0, 4) and parse_shard(" 3/4 ") == (3, 4)
    for bad in ("4/4", "1/0", "-1/2", "a/b", "2", "1/2/3", ""):
        with pytest.raises(ShardError):
            parse_shard(bad)


@pytest.mark.parametrize("total,n", [(4, 2), (7, 3), (3, 5), (0, 2), (1000, 7)])
def test_shard_ranges_tile_the_task_list_exactly(total, n):
    ranges = [shard_range(total, i, n) for i in range(n)]
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))  # no gap, no overlap
    sizes = [hi - lo for lo, hi in ranges]
    assert max(sizes) - min(sizes) <= 1


# ---- journal ----

FP = {
    "room": "r",
    "families": ["a"],
    "episodes": 4,
    "seed": 0,
    "people": 0,
    "policy": "scripted",
    "planner": "rule",
    "split": "train",
    "instructions": False,
    "base": "fixed",
    "res": 32,
    "shard": [0, 2],
}


def test_journal_resume_cuts_a_torn_line_and_refuses_other_settings(tmp_path):
    j = Journal(tmp_path / "p" / "r_sim.jsonl")
    assert j.start(FP, resume=False) == []
    j.append({"task_id": "t0", "outcome": "miss"})
    j.append({"task_id": "t1", "outcome": "miss"})
    with open(j.path, "a") as f:
        f.write('{"task_id": "t2", "outc')  # killed mid-write
    rows = Journal(j.path).start(FP, resume=True)
    assert [r["task_id"] for r in rows] == ["t0", "t1"]
    assert len(j.path.read_text().splitlines()) == 3  # fingerprint + 2 tasks; the torn line is gone
    with pytest.raises(ShardError, match="seed"):
        Journal(j.path).start({**FP, "seed": 1}, resume=True)
    with pytest.raises(ShardError, match="shard"):
        Journal(j.path).start({**FP, "shard": [1, 2]}, resume=True)
    assert Journal(j.path).start(FP, resume=False) == []  # a fresh run starts over
    assert Journal(tmp_path / "none.jsonl").start(FP, resume=True) == []  # no journal: nothing to resume


def test_journal_resume_rejects_out_of_order_task_records(tmp_path):
    j = Journal(tmp_path / "progress.jsonl")
    j.start(FP, resume=False)
    j.append({"task_id": "t1", "family": "a", "outcome": "miss"})
    expected = [{"task_id": "t0", "family": "a"}, {"task_id": "t1", "family": "a"}]
    before = j.path.read_bytes()
    with pytest.raises(ShardError, match="does not match the task sequence"):
        j.start(FP, resume=True, expected_tasks=expected)
    assert j.path.read_bytes() == before


def test_journal_does_not_discard_a_corrupt_completed_line(tmp_path):
    j = Journal(tmp_path / "progress.jsonl")
    j.start(FP, resume=False)
    with j.path.open("a") as stream:
        stream.write("{broken}\n")  # a complete line, not an interrupted append
    before = j.path.read_bytes()
    with pytest.raises(ShardError, match="line 2 is not JSON"):
        j.start(FP, resume=True)
    assert j.path.read_bytes() == before


# ---- merge, on synthetic episodes (no MuJoCo) ----


def _episode(n, seed):
    rng = np.random.default_rng(seed)
    return {
        "states": rng.normal(size=(n, 11)).astype(np.float32),
        "actions": rng.normal(size=(n, 11)).astype(np.float32),
        "videos": {c: rng.integers(0, 255, (n, 16, 16, 3), dtype=np.uint8) for c in CAMS},
        "detail": {"joint": "j", "qpos": 0.2},
        "safety": {"min_human_distance_m": 1.0, "human_contact": 0},
        "people_xy": [[] for _ in range(n)],
    }


def _task(k, fam="open_drawer"):
    return {
        "task_id": f"sim_{fam}_{k:04d}",
        "family": fam,
        "lighting": "neutral",
        "people": 0,
        "instruction": f"open drawer {k % 2}",
    }


def _write_shard(out, idx, count, total, ks, keep=None):
    """A finished shard dataset holding tasks `ks` of `total`, as `sim --shard idx/count` leaves it."""
    lo, hi = shard_range(total, idx, count)
    root = out / "shards" / f"{idx}of{count}"
    w = SimDatasetWriter(root / "datasets" / "room_sim", "room", 20, 16, CAMS)
    for k in ks:
        w.add_episode(_episode(5 + k, k), _task(k))
    meta = {
        "index": idx,
        "count": count,
        "task_range": [lo, hi],
        "total_tasks": total,
        "episodes": total,
        "seed": 0,
        "families": ["open_drawer"],
        "people": 0,
        "policy": "scripted",
        "planner": "rule",
        "split": "train",
        "instructions": False,
        "base": "fixed",
        "res": 16,
    }
    pre = [{"stage": "preflight", "family": "open_drawer", "reason": "x"}]
    w.finish(
        len(ks) + 1,
        {
            "skipped_unreachable": 1,
            "skipped_error": 1,
            "skipped": 2,
            "skip_reasons": pre + [{"stage": "build", "task_id": "t", "reason": "boom"}],
        },
        {"shard": meta},
    )
    return root


def _fake_run(tmp_path, count=3, total=6, drop=()):
    for i in range(count):
        lo, hi = shard_range(total, i, count)
        _write_shard(tmp_path, i, count, total, [k for k in range(lo, hi) if k not in drop])
    return tmp_path


@pytest.mark.parametrize(
    "name,content,match",
    [
        ("episodes.jsonl", "{broken}\n", "episodes.jsonl: line 1 is not JSON"),
        ("episodes.jsonl", "[]\n", "episodes.jsonl: line 1 must be a JSON object"),
        ("info.json", "{broken}", "info.json: invalid JSON"),
    ],
)
def test_merge_reports_corrupt_shard_metadata(tmp_path, name, content, match):
    src = _fake_run(tmp_path / "source", count=1, total=1)
    (src / "shards" / "0of1" / "datasets" / "room_sim" / "meta" / name).write_text(content)
    with pytest.raises(ShardError, match=match):
        merge_shards([src / "shards"], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_synthetic_shards_reindexes_everything_and_validates(tmp_path):
    src, out = _fake_run(tmp_path / "src", drop=(1,)), tmp_path / "merged"
    res = merge_shards([src / "shards"], out)  # the folder holding <i>of<n>
    ds = res["dataset"]
    assert ds == out / "datasets" / "room_sim" and validate(ds) == []
    info = res["info"]
    assert info["total_episodes"] == 5 and info["arrival_merged_shards"] == 3 and "arrival_shard" not in info
    rows = [json.loads(line) for line in (ds / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [r["episode_index"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["arrival_task_id"] for r in rows] == [f"sim_open_drawer_{k:04d}" for k in (0, 2, 3, 4, 5)]  # global order
    assert info["total_frames"] == sum(5 + k for k in (0, 2, 3, 4, 5))
    tasks = [json.loads(line)["task"] for line in (ds / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks == ["open drawer 0", "open drawer 1"] and info["total_tasks"] == 2
    # pre-flight rejections are the same in every shard: counted once; build errors are per shard: summed
    assert info["arrival_skipped_unreachable"] == 1 and info["arrival_skipped_error"] == 3
    assert info["arrival_attempts"] == (1 + 1) + (2 + 1) + (
        2 + 1
    )  # each shard's episodes plus the one it ran and dropped
    assert sum(r.get("stage") == "preflight" for r in info["arrival_skip_reasons"]) == 1
    stats = json.loads((ds / "meta" / "stats.json").read_text())
    assert stats["episode_index"]["max"] == [4.0] and stats["index"]["max"] == [info["total_frames"] - 1.0]
    # the merged parquet of an episode carries the merged indices but the shard's own values
    import pyarrow.parquet as pq

    t = pq.read_table(ds / "data" / "chunk-000" / "episode_000003.parquet")
    assert set(t.column("episode_index").to_pylist()) == {3}
    assert t.column("index").to_pylist()[0] == sum(5 + k for k in (0, 2, 3))
    assert res["report"] is None  # these shards left no report


def test_merge_is_chunked_like_the_validator_expects(tmp_path, monkeypatch):
    import arrival.sim.export as sx

    monkeypatch.setattr(sx, "CHUNKS_SIZE", 2)
    monkeypatch.setattr(sx, "chunk_of", lambda i, size=2: i // size)
    monkeypatch.setattr(shards, "chunk_of", lambda i, size=2: i // size)
    src = _fake_run(tmp_path / "src", count=2, total=6)
    res = merge_shards([src / "shards" / "0of2", src / "shards" / "1of2"], tmp_path / "m")
    assert res["info"]["total_chunks"] == 3 and res["info"]["chunks_size"] == 2
    assert (res["dataset"] / "data" / "chunk-002" / "episode_000004.parquet").is_file()
    assert validate(res["dataset"]) == []


def test_merge_refuses_incomplete_or_mixed_sets(tmp_path):
    src = _fake_run(tmp_path / "src")
    s = src / "shards"
    with pytest.raises(ShardError, match="missing \\[1\\]"):
        merge_shards([s / "0of3", s / "2of3"], tmp_path / "m")
    with pytest.raises(ShardError, match="some given twice|got \\[0, 0, 1\\]"):
        merge_shards([s / "0of3", s / "0of3", s / "1of3"], tmp_path / "m")
    shutil.rmtree(s / "1of3" / "datasets" / "room_sim" / "meta")  # a shard killed before finish: no info.json
    with pytest.raises(ShardError, match="no dataset here"):
        merge_shards([s / "0of3", s / "1of3", s / "2of3"], tmp_path / "m")
    other = _fake_run(tmp_path / "other")
    info_p = other / "shards" / "2of3" / "datasets" / "room_sim" / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["arrival_shard"]["seed"] = 9
    info_p.write_text(json.dumps(info))
    with pytest.raises(ShardError, match="not from one run"):
        merge_shards(sorted((other / "shards").iterdir()), tmp_path / "m")
    plain = tmp_path / "plain"
    SimDatasetWriter(plain / "datasets" / "room_sim", "room", 20, 16, CAMS).finish(0)
    with pytest.raises(ShardError, match="not a shard dataset"):
        merge_shards([plain], tmp_path / "m")
    with pytest.raises(ShardError, match="would overwrite"):
        merge_shards(
            sorted((_fake_run(tmp_path / "again") / "shards").iterdir()), tmp_path / "again" / "shards" / "0of3"
        )


def test_merge_refuses_episode_owned_by_another_shard(tmp_path):
    src = tmp_path / "src"
    _write_shard(src, 0, 2, 4, [2])  # task 2 belongs to shard 1
    _write_shard(src, 1, 2, 4, [2, 3])
    out = tmp_path / "merged"
    with pytest.raises(ShardError, match="out-of-range task"):
        merge_shards([src / "shards"], out)
    assert not out.exists()


def test_missing_shard_video_does_not_erase_existing_merge(tmp_path):
    src = _fake_run(tmp_path / "src")
    video = (
        src
        / "shards"
        / "0of3"
        / "datasets"
        / "room_sim"
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000000.mp4"
    )
    video.unlink()
    out = tmp_path / "merged"
    old = out / "datasets" / "room_sim" / "keep.txt"
    old.parent.mkdir(parents=True)
    old.write_text("previous result")
    with pytest.raises(ShardError, match="missing source file"):
        merge_shards([src / "shards"], out)
    assert old.read_text() == "previous result"


def test_merge_shards_cli_exit_codes(tmp_path, capsys):
    src = _fake_run(tmp_path / "src")
    assert cli.main(["merge-shards", str(src), "--out", str(tmp_path / "o")]) == cli.EXIT_OK  # `src` holds shards/
    assert "OK: merged 3 shards" in capsys.readouterr().out
    assert cli.main(["merge-shards", str(src / "shards" / "0of3"), "--out", str(tmp_path / "o2")]) == cli.EXIT_USAGE
    assert "merge-shards:" in capsys.readouterr().err  # one shard of three is not a whole run


def test_sim_rejects_a_bad_shard_spec(tmp_path, capsys):
    pytest.importorskip("mujoco")
    assert cli.main(["sim", "--shard", "3/3", "--out", str(tmp_path)]) == cli.EXIT_USAGE
    assert "--shard i/n needs" in capsys.readouterr().err


# ---- the real thing: shards of the sim against one run ----

mujoco = pytest.importorskip("mujoco")

from conftest import SCANS  # noqa: E402

from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import collect as collect_mod  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

RUN = dict(families=["counter_to_sink", "open_drawer"], episodes=4, people=0, res=32, seed=3, log=lambda *_: None)


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", SCANS))


@pytest.fixture(scope="module")
def runs(tmp_path_factory, twin):
    """One process over 4 tasks, and the same 4 tasks as 2 shards of 2, merged."""
    root = tmp_path_factory.mktemp("shards")
    single = collect_mod.collect(twin, out_dir=root / "single", **RUN)
    parts = [collect_mod.collect(twin, out_dir=root / "sharded", shard=(i, 2), **RUN) for i in range(2)]
    merged = merge_shards([root / "sharded"], root / "merged", log=lambda *_: None)
    return {"root": root, "single": single, "parts": parts, "merged": merged}


def _rows(ds):
    return [json.loads(line) for line in (ds / "meta" / "episodes.jsonl").read_text().splitlines()]


def _frames(ds):
    import pyarrow.parquet as pq

    return [pq.read_table(p).to_pydict() for p in sorted(ds.glob("data/chunk-*/episode_*.parquet"))]


def test_two_shards_of_two_merge_to_the_dataset_of_one_run_of_four(runs):
    single, merged = runs["single"], runs["merged"]
    a, b = single["dataset"], merged["dataset"]
    assert validate(a) == [] and validate(b) == []
    ia, ib = single["info"], merged["info"]
    assert ia["total_episodes"] == ib["total_episodes"] and ia["total_episodes"] >= 3  # a real comparison
    for k in (
        "total_frames",
        "total_tasks",
        "total_videos",
        "arrival_attempts",
        "arrival_successes",
        "arrival_families",
        "arrival_skipped",
        "arrival_skipped_unreachable",
        "arrival_skipped_error",
        "features",
        "splits",
    ):
        assert ia[k] == ib[k], k
    ra, rb = _rows(a), _rows(b)
    assert [r["episode_index"] for r in rb] == list(range(len(rb)))
    assert ra == rb  # same task ids, wording, lengths, people, safety numbers, in the same order
    assert (a / "meta" / "tasks.jsonl").read_text() == (b / "meta" / "tasks.jsonl").read_text()
    for ea, eb in zip(_frames(a), _frames(b)):
        assert ea == eb  # states, actions, task, episode and global frame indices
    sa, sb = (json.loads((d / "meta" / "stats.json").read_text()) for d in (a, b))
    for k in sa:
        assert (
            np.allclose(sa[k]["mean"], sb[k]["mean"])
            and np.allclose(sa[k]["std"], sb[k]["std"])
            and sa[k]["count"] == sb[k]["count"]
        )
    assert len(list(b.glob("videos/chunk-*/*/*.mp4"))) == 2 * len(rb)


def test_shard_outputs_are_separate_and_cover_disjoint_tasks(runs):
    parts = runs["parts"]
    assert [p["dataset"].parent.parent.name for p in parts] == ["0of2", "1of2"]
    ids = [[r["arrival_task_id"] for r in _rows(p["dataset"])] for p in parts]
    assert not set(ids[0]) & set(ids[1])
    assert [p["info"]["arrival_shard"]["task_range"] for p in parts] == [[0, 2], [2, 4]]
    assert all(p["report_data"]["shard"]["total_tasks"] == 4 for p in parts)
    assert (runs["root"] / "sharded" / "shards" / "0of2" / "progress").is_dir()
    rep = json.loads(runs["merged"]["report"].read_text())
    assert rep["attempts"] == runs["single"]["report_data"]["attempts"] and rep["merged_shards"] == 2


def test_resumed_shard_equals_an_uninterrupted_one(runs, twin, tmp_path, monkeypatch):
    """Kill shard 0 during its second task, resume it, and get the shard an uninterrupted run wrote."""
    real, calls = collect_mod.run_episode, []

    def killed(env, policy, task, **kw):
        calls.append(task["task_id"])
        if len(calls) == 2:
            raise KeyboardInterrupt  # not an Exception: collect must not swallow it as a skipped task
        return real(env, policy, task, **kw)

    monkeypatch.setattr(collect_mod, "run_episode", killed)
    with pytest.raises(KeyboardInterrupt):
        collect_mod.collect(twin, out_dir=tmp_path, shard=(0, 2), **RUN)
    monkeypatch.undo()
    journal = tmp_path / "shards" / "0of2" / "progress" / "hero_kitchen_sim.jsonl"
    assert len(journal.read_text().splitlines()) == 2  # fingerprint + the one task that finished
    ds = tmp_path / "shards" / "0of2" / "datasets" / "hero_kitchen_sim"
    n_done = len(journal.read_text().splitlines()) - 1
    stray = [
        ds / "data" / "chunk-000" / f"episode_{n_done:06d}.parquet",  # a kill between the files and the journal line
        ds / "videos" / "chunk-000" / "observation.images.wrist" / f"episode_{n_done:06d}.mp4",
    ]
    for f in stray:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"partial")
    with pytest.raises(ShardError, match="different seed"):
        collect_mod.collect(twin, out_dir=tmp_path, shard=(0, 2), resume=True, **{**RUN, "seed": 4})
    res = collect_mod.collect(twin, out_dir=tmp_path, shard=(0, 2), resume=True, **RUN)
    ref = runs["parts"][0]
    assert res["report_data"]["resumed_tasks"] == 1
    assert validate(res["dataset"]) == []
    assert _rows(res["dataset"]) == _rows(ref["dataset"])
    assert _frames(res["dataset"]) == _frames(ref["dataset"])
    for k in ("total_episodes", "total_frames", "arrival_attempts", "arrival_successes", "arrival_skipped"):
        assert res["info"][k] == ref["info"][k], k
    untimed = [
        {f: {k: v for k, v in c.items() if k != "seconds"} for f, c in r["report_data"]["families"].items()}
        for r in (res, ref)
    ]
    assert untimed[0] == untimed[1]
    done_secs = json.loads(journal.read_text().splitlines()[1])["seconds"]
    assert (
        sum(c["seconds"] for c in res["report_data"]["families"].values()) > done_secs
    )  # the task run before the kill counts
    # an interrupted second task left no episode behind that the resumed run kept twice
    assert len(list(res["dataset"].glob("data/chunk-*/*.parquet"))) == res["info"]["total_episodes"]


def test_report_times_each_family_so_cost_can_compare_with_teleop(runs):
    from conftest import REPO

    from arrival.eval import cost

    rep = runs["single"]["report_data"]
    fams = rep["families"]
    assert set(fams) == {"counter_to_sink", "open_drawer"}
    assert all(f["seconds"] > 0 for f in fams.values())
    # the total and each family are rounded to 0.01 s on their own, so the family sum can run 0.005 s per family over
    # (CI: 169.55 against 169.54 + 0.01, which is 169.54999... in floating point)
    assert sum(f["seconds"] for f in fams.values()) <= rep["seconds"] + 0.005 * len(fams) + 1e-9
    merged = json.loads(runs["merged"]["report"].read_text())
    for fam, f in merged["families"].items():
        parts = [
            p["report_data"]["families"][fam]["seconds"] for p in runs["parts"] if fam in p["report_data"]["families"]
        ]
        assert f["seconds"] == pytest.approx(sum(parts), abs=0.02)
    baseline = REPO / "docs" / "results" / "p3-teleop-2026-10-21" / "teleop_baseline.json"
    c = cost.build(runs["root"] / "single", baseline_path=baseline)["comparison"]
    assert c["measured"] and c["task"] == "counter_to_sink" and c["generated"]["scope"] == "task"
    assert c["generated"]["seconds_per_success"] == pytest.approx(
        fams["counter_to_sink"]["seconds"] / fams["counter_to_sink"]["successes"], abs=0.001
    )
    assert c["seconds_ratio_teleop_over_generated"] is not None


def test_merge_of_a_shard_report_without_family_seconds_leaves_them_unknown(tmp_path):
    fam = {"attempts": 1, "successes": 1, "human_contacts": 0, "fridge_visits": 0, "door_contacts": 0}
    rep = {"room_id": "room", "seconds": 2.0, "human_contacts": 0}
    old, new = {**rep, "families": {"open_drawer": fam}}, {**rep, "families": {"open_drawer": {**fam, "seconds": 2.0}}}
    info = {"total_episodes": 2, "arrival_attempts": 2}
    for pair, want in (([new, new], 4.0), ([old, new], None), ([new, old], None)):
        path = shards._merged_report([{"report": r} for r in pair], info, {}, tmp_path / "r.json", 2)
        assert json.loads(path.read_text())["families"]["open_drawer"]["seconds"] == want
