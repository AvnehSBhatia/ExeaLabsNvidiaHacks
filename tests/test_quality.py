"""Per-episode quality checks and the cleaned-dataset writer (arrival.export.quality, `arrival validate --episodes`)."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from arrival import cli
from arrival.export import quality
from arrival.export.validate import validate
from arrival.sim.export import SimDatasetWriter
from arrival.sim.robot import RANGES

RES = 16  # mp4 frames must be multiples of 16
MID = np.array([(lo + hi) / 2 for lo, hi in RANGES])


def trace(n=64, phase=0.0):
    """A smooth in-range arm motion plus a gripper column; the command leads the state by one small step."""
    t = np.linspace(0, 2 * np.pi, n)[:, None]
    q = MID + 0.3 * np.sin(t + phase + np.arange(7) * 0.4)
    base = np.zeros((n, 3))  # base x, y, yaw: the dataset schema carries them even for a fixed base
    s = np.concatenate([q, 0.02 + 0.01 * np.sin(t), base], 1)
    a = np.concatenate([q + 0.01, 0.02 + 0.01 * np.sin(t), base], 1)
    return s.astype(np.float32), a.astype(np.float32)


def video(n, blank=False, seed=0):
    rng = np.random.default_rng(seed)
    return (np.zeros((n, RES, RES, 3), np.uint8) if blank else rng.integers(0, 255, (n, RES, RES, 3), dtype=np.uint8))


def episode(s, a, blank_wrist=False, seed=0):
    n = len(s)
    return {"states": s, "actions": a, "videos": {"ego_view": video(n, seed=seed), "wrist": video(n, blank_wrist, seed + 1)},
            "detail": {"ok": True}, "safety": {"min_human_distance_m": 1.0, "human_contact": False},
            "people_xy": [[1.0, 1.0]] * n}


def build(root, defects, bad_first=False):
    """Six good open_drawer episodes, and one episode per entry of `defects` (name -> episode dict), after them
    (or before, with bad_first). Returns name -> episode index of each defect."""
    w = SimDatasetWriter(root, "qa_room", 20, RES, ("ego_view", "wrist"))
    task = {"instruction": "open the drawer", "family": "open_drawer", "lighting": "day"}
    names = {}

    def bad():
        for name, ep in defects.items():
            names[name] = w.add_episode(ep, task)

    if bad_first:
        bad()
    for i in range(6):
        s, a = trace(60 + 2 * i, phase=i * 0.1)
        w.add_episode(episode(s, a, seed=i), task)
    if not bad_first:
        bad()
    w.finish(len(w.episodes))
    return names


def defects():
    s, a = trace()
    nan_s = s.copy()
    nan_s[10, 2] = np.nan
    still = np.repeat(s[:1], 64, 0)
    long_s, long_a = trace(120, 0.3)
    stuck_s = long_s.copy()
    stuck_s[20:100] = stuck_s[20]  # frozen while the command keeps moving away
    lim_s, lim_a = trace()
    lim_s = lim_s.copy()
    lim_s[5:60, 6] = RANGES[6][1]  # joint 7 pinned at its stop
    big_s, big_a = trace(400, 0.2)
    return {
        "nan": episode(nan_s, a),
        "zero_motion": episode(still, np.repeat(a[:1], 64, 0)),
        "stuck_arm": episode(stuck_s, long_a),
        "joint_at_limit": episode(lim_s, lim_a),
        "length_outlier": episode(big_s, big_a),
        "blank_video": episode(s, a, blank_wrist=True),
    }


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("q") / "ds"
    return root, build(root, defects())


def flags_of(report, ep):
    return {f["check"] for f in report["results"][ep]["flags"]}


def test_good_episodes_pass_and_dataset_is_structurally_valid(tmp_path):
    build(tmp_path / "ds", {})
    assert validate(tmp_path / "ds") == []
    rep = quality.check_episodes(tmp_path / "ds", frames=True)
    assert rep["flagged"] == 0 and rep["kept"] == 6
    assert set(rep["checks_run"]) >= {"nonfinite", "zero_motion", "stuck_arm", "joint_at_limit", "blank_video"}
    assert "carried_object_tilt" in rep["not_checked"]  # honest: not measured


def test_each_defect_is_flagged_on_its_own_episode(ds):
    root, idx = ds
    rep = quality.check_episodes(root, frames=True)
    assert [r["episode_index"] for r in rep["results"] if r["ok"]] == list(range(6))  # defects come last
    for name, check in (("nan", "nonfinite"), ("zero_motion", "zero_motion"), ("stuck_arm", "stuck_arm"),
                        ("joint_at_limit", "joint_at_limit"), ("length_outlier", "length_outlier"),
                        ("blank_video", "blank_video")):
        assert check in flags_of(rep, idx[name]), name
    assert flags_of(rep, idx["nan"]) == {"nonfinite"}  # NaN short-circuits: no noise from the other checks
    assert flags_of(rep, idx["joint_at_limit"]) == {"joint_at_limit"}
    assert flags_of(rep, idx["stuck_arm"]) == {"stuck_arm"}
    assert rep["flags_by_check"]["blank_video"] == 1


def test_blank_video_needs_frames_flag(ds):
    root, idx = ds
    assert quality.check_episodes(root, frames=False)["results"][idx["blank_video"]]["ok"]


def test_thresholds_are_boundaries_not_hair_triggers():
    s, a = trace()
    # joint 7 at its stop for exactly LIMIT_STEPS steps passes, one more flags
    s = s.copy()
    s[:, 6] = MID[6]
    s[:quality.LIMIT_STEPS, 6] = RANGES[6][1]
    assert not [f for f in quality.check_trace(s, a, list(RANGES)) if f["check"] == "joint_at_limit"]
    s[quality.LIMIT_STEPS, 6] = RANGES[6][1]
    assert [f for f in quality.check_trace(s, a, list(RANGES)) if f["check"] == "joint_at_limit"]
    # near the stop but outside the 3% band is fine
    lo, hi = RANGES[6]
    s[:, 6] = hi - 0.05 * (hi - lo)
    assert not [f for f in quality.check_trace(s, a, list(RANGES)) if f["check"] == "joint_at_limit"]
    # no limits (placeholder joints): never flagged
    s[:, 6] = hi
    assert not [f for f in quality.check_trace(s, a, None) if f["check"] == "joint_at_limit"]


def test_length_outliers_skip_small_families():
    lengths = {0: 60, 1: 62, 2: 64, 3: 400}
    assert list(quality.length_outliers(lengths, dict.fromkeys(lengths, "f"))) == [3]
    small = {0: 60, 1: 400}
    assert quality.length_outliers(small, dict.fromkeys(small, "f")) == {}


def test_drop_writes_a_valid_reindexed_dataset(ds, tmp_path):
    root, idx = ds
    rep = quality.drop_flagged(root, tmp_path / "clean", frames=True)
    out = tmp_path / "clean"
    assert validate(out) == []
    assert rep["cleaned"]["kept"] == 6 and sorted(rep["cleaned"]["dropped"]) == sorted(idx.values())
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 6 and info["total_videos"] == 12 and info["arrival_dropped_by_validator"] == 6
    assert info["arrival_attempts"] == 12  # collection history is not rewritten
    t = pq.read_table(out / "data" / "chunk-000" / "episode_000005.parquet")
    assert set(t.column("episode_index").to_pylist()) == {5}
    # kept episodes are byte-for-value copies of the originals
    a = pq.read_table(root / "data" / "chunk-000" / "episode_000003.parquet")
    b = pq.read_table(out / "data" / "chunk-000" / "episode_000003.parquet")
    assert a.column("observation.state").to_pylist() == b.column("observation.state").to_pylist()
    assert quality.check_episodes(out, frames=True)["flagged"] == 0
    dropped = json.loads((out / "meta" / "dropped_episodes.json").read_text())
    assert dropped["dropped_reasons"][str(idx["nan"])][0]["check"] == "nonfinite"
    assert len(list((out / "videos").rglob("*.mp4"))) == 12
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert stats["observation.state"]["count"] == [info["total_frames"]]
    rows = np.concatenate([np.asarray(pq.read_table(f).column("observation.state").to_pylist())
                           for f in sorted((out / "data").rglob("*.parquet"))])
    assert np.allclose(stats["observation.state"]["q01"], np.quantile(rows, 0.01, axis=0))
    assert np.allclose(stats["observation.state"]["q99"], np.quantile(rows, 0.99, axis=0))


def test_drop_in_the_middle_reindexes_and_keeps_validity(tmp_path):
    s, a = trace()
    names = build(tmp_path / "ds", {"zero_motion": episode(np.repeat(s[:1], 64, 0), a)}, bad_first=True)
    assert names["zero_motion"] == 0
    rep = quality.check_episodes(tmp_path / "ds")
    assert rep["flagged_episodes"] == [names["zero_motion"]]
    quality.drop_flagged(tmp_path / "ds", tmp_path / "clean", rep)
    assert validate(tmp_path / "clean") == []
    t = pq.read_table(tmp_path / "clean" / "data" / "chunk-000" / "episode_000000.parquet")
    old = pq.read_table(tmp_path / "ds" / "data" / "chunk-000" / "episode_000001.parquet")  # first good episode
    assert t.column("observation.state").to_pylist() == old.column("observation.state").to_pylist()
    assert t.column("index").to_pylist() == list(range(t.num_rows))


def test_drop_refuses_existing_output_inside_source_and_all_flagged(ds, tmp_path):
    root, _ = ds
    (tmp_path / "exists").mkdir()
    with pytest.raises(quality.QualityError, match="already exists"):
        quality.drop_flagged(root, tmp_path / "exists")
    with pytest.raises(quality.QualityError, match="outside"):
        quality.drop_flagged(root, root / "clean")
    s, a = trace()
    build_dir = tmp_path / "allbad"
    w = SimDatasetWriter(build_dir, "r", 20, RES, ("ego_view", "wrist"))
    w.add_episode(episode(np.repeat(s[:1], 64, 0), a), {"instruction": "x", "family": "f", "lighting": "d"})
    w.finish(1)
    with pytest.raises(quality.QualityError, match="nothing to write"):
        quality.drop_flagged(build_dir, tmp_path / "none")
    assert not (tmp_path / "none").exists()


def test_not_a_dataset_is_a_typed_error(tmp_path):
    with pytest.raises(quality.QualityError):
        quality.check_episodes(tmp_path)


def test_cli_plain_validate_ignores_quality(tmp_path):
    s, a = trace()
    build(tmp_path / "ds", {"zero_motion": episode(np.repeat(s[:1], 64, 0), a)})
    assert cli.main(["validate", str(tmp_path / "ds")]) == cli.EXIT_OK  # structurally fine, just not worth training on
    assert cli.main(["validate", str(tmp_path / "ds"), "--episodes"]) == cli.EXIT_INVALID


def test_cli_episodes_exit_codes_and_drop(ds, tmp_path, capsys):
    root, idx = ds
    assert cli.main(["validate", str(root)]) == cli.EXIT_INVALID  # structural check: the NaN episode is already caught
    assert cli.main(["validate", str(root), "--episodes", "--json", str(tmp_path / "r.json")]) == cli.EXIT_INVALID
    out = capsys.readouterr().out
    assert f"episode {idx['nan']}" in out and "nonfinite" in out
    assert json.loads((tmp_path / "r.json").read_text())["flagged"] >= 5
    assert cli.main(["validate", str(root), "--drop", str(tmp_path / "clean")]) == cli.EXIT_OK
    assert cli.main(["validate", str(tmp_path / "clean"), "--episodes"]) == cli.EXIT_OK
    assert cli.main(["validate", str(tmp_path / "missing"), "--episodes"]) == cli.EXIT_INVALID


def _write_episodes(root, rng, lengths):
    import pyarrow as pa
    from arrival.export import lerobot

    (root / "data" / "chunk-000").mkdir(parents=True)
    stats = []
    for i, n in enumerate(lengths):
        s, a = (rng.normal(size=(n, 11)).astype(np.float32) for _ in range(2))
        t = pa.Table.from_arrays([pa.array(list(s), type=pa.list_(pa.float32())),
                                  pa.array(list(a), type=pa.list_(pa.float32())),
                                  pa.array(np.arange(n, dtype=np.float32) / 20)], names=["observation.state", "action", "timestamp"])
        pq.write_table(t, root / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
        stats.append({"stats": {"observation.state": lerobot._stats(s), "action": lerobot._stats(a),
                                  "timestamp": lerobot._stats(np.arange(n, dtype=np.float32) / 20)}})
    return stats


def test_dataset_stats_quantiles_match_the_python_list_reading(tmp_path):
    from arrival.export.lerobot import QUANTILES, dataset_stats

    stats = _write_episodes(tmp_path, np.random.default_rng(3), [40, 1, 77])
    got = dataset_stats(tmp_path, stats)
    for key in ("observation.state", "action", "timestamp"):
        rows = [v for f in sorted(tmp_path.glob("data/chunk-*/episode_*.parquet"))
                for v in pq.read_table(f).column(key).to_pylist()]
        q01, q99 = np.quantile(np.asarray(rows, dtype=np.float64).reshape(len(rows), -1), QUANTILES, axis=0)
        assert got[key]["q01"] == q01.tolist() and got[key]["q99"] == q99.tolist()


def test_list_column_matches_to_pylist_and_refuses_ragged_or_null_rows():
    import pyarrow as pa
    from arrival.export.lerobot import list_column

    rows = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    col = pa.chunked_array([pa.array(list(rows[:2]), pa.list_(pa.float32())), pa.array(list(rows[2:]), pa.list_(pa.float32()))])
    assert np.array_equal(list_column(col), np.array(col.to_pylist(), dtype=np.float64))
    with pytest.raises(ValueError):
        list_column(pa.chunked_array([pa.array([[1.0, 2.0], [3.0]], pa.list_(pa.float32()))]))
    with pytest.raises(ValueError):
        list_column(pa.chunked_array([pa.array([[1.0], None], pa.list_(pa.float32()))]))
