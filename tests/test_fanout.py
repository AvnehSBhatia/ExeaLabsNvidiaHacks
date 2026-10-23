"""C1 offline: fanout to a fake local backend with resumable storage (arrival/jobs/fanout.py)."""

import json
import sys

import pytest

pytest.importorskip("mujoco")

from arrival.export.validate import validate  # noqa: E402
from arrival.jobs.fanout import FanoutConfig, LocalBackend, NebiusBackendStub, marker_key, run_fanout  # noqa: E402
from arrival.jobs.shards import ShardError  # noqa: E402
from conftest import SCANS  # noqa: E402

CFG = FanoutConfig("r1", 2, "hero_kitchen", "counter_to_sink,open_drawer", 4, 0, 32, 3, str(SCANS))
quiet = lambda *_: None  # noqa: E731
CMD_TRUE = [sys.executable, "-c", ""]
CMD_FALSE = [sys.executable, "-c", "import sys; sys.exit(1)"]


def _rows(ds):
    return [json.loads(x) for x in (ds / "meta" / "episodes.jsonl").read_text().splitlines()]


def _frames(ds):
    import pyarrow.parquet as pq

    return [pq.read_table(p).to_pydict() for p in sorted(ds.glob("data/chunk-*/episode_*.parquet"))]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("ref")
    be = LocalBackend(root / "store", root / "scratch")
    return run_fanout(be, CFG, root / "out", log=quiet)


@pytest.mark.render  # the sim renders in a child process, where --no-render's patch does not reach
def test_failed_shard_is_retried_and_dataset_is_valid(reference, tmp_path):
    be = LocalBackend(tmp_path / "store", tmp_path / "scratch", fail_first={1})
    res = run_fanout(be, CFG, tmp_path / "out", log=quiet)
    assert (1, 2) in be.submissions and (0, 2) not in be.submissions  # only shard 1 was retried
    assert validate(res["dataset"]) == [] and res["ran"] == [0, 1]
    assert _rows(res["dataset"]) == _rows(reference["dataset"])
    assert _frames(res["dataset"]) == _frames(reference["dataset"])


@pytest.mark.render  # the sim renders in a child process, where --no-render's patch does not reach
def test_killed_run_resumes_skipping_done_shards(reference, tmp_path):
    be = LocalBackend(tmp_path / "store", tmp_path / "scratch")

    def kill(i):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_fanout(be, CFG, tmp_path / "out", parallel=1, on_shard_done=kill, log=quiet)
    assert be.exists("r1/done/0of2.json") and not be.exists("r1/done/1of2.json")
    be2 = LocalBackend(tmp_path / "store", tmp_path / "scratch")  # a new process, same storage
    res = run_fanout(be2, CFG, tmp_path / "out", log=quiet)
    assert res["skipped"] == [0] and res["ran"] == [1]
    assert [s for s, _ in be2.submissions] == [1]  # shard 0 was not run again
    assert validate(res["dataset"]) == []
    assert _rows(res["dataset"]) == _rows(reference["dataset"])
    assert _frames(res["dataset"]) == _frames(reference["dataset"])
    assert res["info"]["total_frames"] == reference["info"]["total_frames"]


def test_gives_up_after_max_attempts_and_refuses_other_settings(tmp_path):
    be = LocalBackend(tmp_path / "store", tmp_path / "s", command=lambda c, i, o: CMD_FALSE)
    with pytest.raises(ShardError, match="failed 2 times"):
        run_fanout(be, CFG, tmp_path / "out", max_attempts=2, log=quiet)
    with pytest.raises(ShardError, match="different settings"):
        run_fanout(be, FanoutConfig("r1", 3), tmp_path / "out", log=quiet)


@pytest.mark.parametrize("content", [b"{broken", b"\xff"])
def test_corrupt_run_manifest_refuses_resume_before_submitting(tmp_path, content):
    be = LocalBackend(tmp_path / "store", tmp_path / "scratch")
    be.put_object("r1/run.json", content)
    with pytest.raises(ShardError, match="invalid run manifest r1/run.json"):
        run_fanout(be, CFG, tmp_path / "out", log=quiet)
    assert be.submissions == []


def test_nebius_stub_is_not_a_backend():
    with pytest.raises(NotImplementedError):
        NebiusBackendStub().submit(CFG, 0)


@pytest.mark.parametrize("run_id", ["", "..", "../escape", "a/b", "/tmp/escape", "a" * 65])
def test_run_id_cannot_escape_scratch_or_storage(run_id):
    with pytest.raises(ShardError, match="run_id"):
        FanoutConfig(run_id, 2)


@pytest.mark.parametrize("patch,name", [
    ({"shards": 0}, "shards"), ({"shards": 1.5}, "shards"),
    ({"episodes": 0}, "episodes"), ({"people": -1}, "people"),
    ({"res": True}, "res"), ({"seed": 1.2}, "seed"),
    ({"room": " "}, "room"), ({"families": "open_drawer,"}, "families"),
    ({"scans": " "}, "scans"),
])
def test_invalid_sim_settings_fail_before_submission(patch, name):
    with pytest.raises(ShardError, match=name):
        FanoutConfig(**{"run_id": "run", "shards": 2, **patch})

@pytest.mark.parametrize("controls", [
    {"parallel": 0}, {"parallel": -1}, {"parallel": True},
    {"max_attempts": 0}, {"max_attempts": -1},
    {"poll_seconds": -0.1}, {"poll_seconds": float("nan")}, {"poll_seconds": float("inf")},
])
def test_bad_fanout_controls_fail_before_submitting(controls, tmp_path):
    be = LocalBackend(tmp_path / "store", tmp_path / "scratch")
    with pytest.raises(ShardError):
        run_fanout(be, CFG, tmp_path / "out", **controls, log=quiet)
    assert be.submissions == []
    assert not be.exists("r1/run.json")

def test_unknown_backend_state_does_not_count_as_a_failed_attempt(tmp_path):
    class UnknownBackend(LocalBackend):
        def poll(self, job_id):
            return "queued"

    be = UnknownBackend(tmp_path / "store", tmp_path / "scratch", command=lambda c, i, o: CMD_TRUE)
    with pytest.raises(ShardError, match="unknown state 'queued'"):
        run_fanout(be, CFG, tmp_path / "out", poll_seconds=0, log=quiet)
    assert be.submissions == [(0, 1), (1, 1)]

@pytest.mark.parametrize("marker", [b"{broken", b'{"shard": 1, "attempts": 1}',
                                    b'{"shard": 0, "attempts": 0}'])
def test_invalid_done_marker_cannot_skip_a_shard(tmp_path, marker):
    be = LocalBackend(tmp_path / "store", tmp_path / "scratch", command=lambda c, i, o: CMD_TRUE)
    be.put_object(marker_key(CFG, 0), marker)
    with pytest.raises(ShardError, match="invalid done marker"):
        run_fanout(be, CFG, tmp_path / "out", log=quiet)
    assert be.submissions == []


def test_valid_marker_without_shard_dataset_is_not_skipped(tmp_path):
    be = LocalBackend(tmp_path / "store", tmp_path / "scratch", command=lambda c, i, o: CMD_TRUE)
    be.put_object(marker_key(CFG, 0), b'{"shard": 0, "attempts": 1}')
    with pytest.raises(ShardError, match="missing .*info.json"):
        run_fanout(be, CFG, tmp_path / "out", log=quiet)
    assert be.submissions == []

def test_local_backend_refuses_another_shards_output(tmp_path):
    be = LocalBackend(tmp_path / "store", tmp_path / "scratch", command=lambda c, i, o: CMD_TRUE)
    job = be.submit(CFG, 0)
    wrong = tmp_path / "scratch" / CFG.run_id / "0of2" / "shards" / "1of2"
    wrong.mkdir(parents=True)
    (wrong / "result.txt").write_text("wrong shard")
    with pytest.raises(ShardError, match="expected shard folder"):
        be.fetch_result(job, f"{CFG.run_id}/shards/0of2")
    assert be.list_objects(f"{CFG.run_id}/shards/0of2") == []


def test_sim_options_reach_every_shard_command():
    cfg = FanoutConfig("r1", 2, "hero_kitchen", "open_drawer", 2, 0, 32, 3, str(SCANS), 0.025, "mobile", True)
    args = cfg.sim_args(1)
    assert args[0] == "sim"
    assert args[args.index("--dart") + 1] == "0.025"
    assert args[args.index("--base") + 1] == "mobile"
    assert "--exocentric" in args
    plain = CFG.sim_args(0)
    assert not {"--dart", "--base", "--exocentric"} & set(plain)  # defaults leave the command unchanged


@pytest.mark.parametrize("patch,name", [({"dart": -0.1}, "dart"), ({"dart": float("nan")}, "dart"),
                                        ({"base": "legs"}, "base"), ({"exocentric": 1}, "exocentric")])
def test_bad_sim_options_are_refused(patch, name):
    with pytest.raises(ShardError, match=name):
        FanoutConfig(**{"run_id": "run", "shards": 2, **patch})


@pytest.mark.render
def test_dart_and_exocentric_shards_merge_into_a_valid_dataset(tmp_path):
    cfg = FanoutConfig("r2", 2, "hero_kitchen", "open_drawer", 2, 0, 32, 3, str(SCANS), 0.02, "fixed", True)
    res = run_fanout(LocalBackend(tmp_path / "store", tmp_path / "scratch"), cfg, tmp_path / "out", log=quiet)
    assert validate(res["dataset"]) == []
    assert res["info"]["total_episodes"] == 2
    assert any("overview" in k for k in res["info"]["features"])
