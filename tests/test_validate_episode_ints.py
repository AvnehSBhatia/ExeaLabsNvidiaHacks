"""episodes.jsonl episode_index and length are integers. A string crashes validate.

`validate` sums the lengths and formats `episode_index // chunks_size`. A length of `"2"`
raises `TypeError: unsupported operand type(s) for +: 'int' and 'str'`. An index of `"0"`
raises `TypeError: unsupported operand type(s) for //: 'str' and 'int'`. `True` is an int
in that arithmetic, so an index of `true` is formatted as episode 1 and the report says
the real parquet is unexpected and `episode_000001.parquet` is missing. The function's
contract is a list of error strings.
"""

import json

import numpy as np
import pytest

from arrival.cli import main
from arrival.export.lerobot import LeRobotWriter
from arrival.export.validate import validate

MODALITY = {
    "state": {"arm": {"start": 0, "end": 2}},
    "action": {"arm": {"start": 0, "end": 2, "absolute": True}},
    "annotation": {"human.task_description": {}},
}


def _dataset(tmp_path):
    writer = LeRobotWriter(tmp_path / "ds", "room", "none", state_dim=2, action_dim=2, modality=MODALITY)
    writer.add_episode(np.zeros((2, 2)), np.zeros((2, 2)), np.arange(2) / 20, "task", True)
    writer.finish(1)
    return tmp_path / "ds"


def _set(root, field, value):
    path = root / "meta" / "episodes.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("length", "2"),
        ("episode_index", "0"),
        ("episode_index", True),
        ("length", True),
        ("length", 2.0),
    ],
)
def test_episode_index_and_length_must_be_integers(tmp_path, field, value):
    root = _dataset(tmp_path)
    assert validate(root) == []
    _set(root, field, value)
    errors = validate(root)
    assert any("episode_index and length must be integers" in error for error in errors), errors
    assert not any("episode_000001" in error for error in errors), errors


def test_validate_cli_reports_a_string_length(tmp_path, capsys):
    root = _dataset(tmp_path)
    _set(root, "length", "2")
    assert main(["validate", str(root)]) == 1
    err = capsys.readouterr().err
    assert "INVALID:" in err
    assert "episode_index and length must be integers, got 0 and '2'" in err
    assert "Traceback" not in err
