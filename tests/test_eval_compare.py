import json

import pytest

from arrival import cli
from arrival.eval.compare import CompareError, compare, sign_test_p


def write_run(path, outcomes, manifest="m1", checkpoint="ck"):
    path.mkdir(parents=True)
    rows = [{"run_id": path.name, "checkpoint_id": checkpoint, "manifest_id": manifest, "trial_index": i,
             "task_id": f"t{i}", "family": "counter_to_sink" if i % 2 else "open_drawer", "success": ok}
            for i, ok in enumerate(outcomes)]
    (path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def outcomes_17_4(n=48):
    """B wins 17 trials, A wins 4, both succeed on 22, both fail on 5: A 26/48, B 39/48."""
    a, b = [], []
    for kind, count in (("b", 17), ("a", 4), ("both", 22), ("none", 5)):
        for _ in range(count):
            a.append(kind in ("a", "both"))
            b.append(kind in ("b", "both"))
    return a, b


def test_sign_test_known_values():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0
    assert sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert sign_test_p(17, 4) == pytest.approx(0.0072, abs=5e-4)


def test_compare_counts_pairs_and_reports_p(tmp_path):
    a, b = outcomes_17_4()
    res = compare(write_run(tmp_path / "a", a, checkpoint="clean"), write_run(tmp_path / "b", b, checkpoint="dart"))
    t = res["total"]
    assert (t["trials"], t["a_successes"], t["b_successes"], t["a_won"], t["b_won"]) == (48, 26, 39, 4, 17)
    assert t["p"] == pytest.approx(0.0072, abs=5e-4)
    assert sum(f["trials"] for f in res["families"].values()) == 48
    assert res["a_checkpoint"] == "clean" and res["b_checkpoint"] == "dart"


def test_different_manifests_are_refused(tmp_path):
    a, b = outcomes_17_4()
    with pytest.raises(CompareError, match="different manifests"):
        compare(write_run(tmp_path / "a", a, manifest="m1"), write_run(tmp_path / "b", b, manifest="m2"))


def test_incomplete_run_is_refused(tmp_path):
    a, b = outcomes_17_4()
    with pytest.raises(CompareError, match="same trials"):
        compare(write_run(tmp_path / "a", a), write_run(tmp_path / "b", b[:-1]))


def test_cli_exit_codes(tmp_path, capsys):
    a, b = outcomes_17_4()
    ra, rb = write_run(tmp_path / "a", a), write_run(tmp_path / "b", b)
    assert cli.main(["eval-compare", str(ra), str(rb)]) == cli.EXIT_OK
    assert "0.007" in capsys.readouterr().out
    rc = write_run(tmp_path / "c", b, manifest="other")
    assert cli.main(["eval-compare", str(ra), str(rc)]) == cli.EXIT_USAGE
    assert "different manifests" in capsys.readouterr().err
    assert cli.main(["eval-compare", str(ra), str(tmp_path / "missing")]) == cli.EXIT_USAGE
