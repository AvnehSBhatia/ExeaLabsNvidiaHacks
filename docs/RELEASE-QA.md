# Release QA (DEM-3)

Run this before each demo, from a fresh clone, and record the result at the bottom. Nothing here needs a GPU, a key or the network beyond the clone and the install.

## Checklist

1. Fresh clone and install exactly as the README says: `git clone`, `python -m venv .venv`, activate, `python -m pip install -e ".[dev,sim]"`. Note the Python version.
2. `python -m pytest -q`: all green. Note the count and the time.
3. `python -m arrival sim --episodes 8 --people 2 --save-preview`: 8/8, no human contacts, `out/previews/` has the contact sheet and the overview mp4. No `MUJOCO_LOG.TXT` in the repo root.
4. `python -m arrival validate out/datasets/hero_kitchen_sim`: `OK:` line, exit 0.
5. `python -m arrival sim --families all --episodes 16 --people 2 --seed 1`: every family in `SIM_FAMILIES` passes except the ones the README lists as skipped or open; compare with the status table.
6. `python -m arrival demo --episodes 4` and `validate out/datasets/hero_kitchen`: the placeholder path still works and still says `arrival_physically_valid: false`.
7. `python -m arrival eval --episodes 2 --people 2`: table printed, `out/eval/hero_kitchen_scripted.json` written.
8. `python -m arrival serve` then `python -m arrival work --once --episodes 2` in another terminal with an upload: job goes `queued` to `done` with a `dataset.zip`.
9. README: the status table, every task bullet's Done/Left text, the folder table and the command comments match what the code does today. REFERENCE: every command and flag exists (`python -m arrival <cmd> --help`). 04-contracts: field names match the files the writers produce.
10. `git status` is clean after all of it (`out/` is ignored).

## Last run

Date: Oct 17 2026. Commit: main 37dcbe9 (after PRs #2, #3, #4, #5, #9, #10, #11, #12, #13 and #19). Machine: Windows 11, Python 3.13.3, laptop CPU, mujoco 3.16.0.

| Step | Result |
|---|---|
| Install | clean; the editable install builds in a few seconds, MuJoCo 3.16.0 wheels fetched |
| Tests | 145 passed in 2 min 0 s |
| `sim --episodes 8 --people 2 --save-preview` | 8/8 (counter_to_sink 4/4, open_drawer 4/4), 0 contacts, 16.3 s, contact sheet and mp4 written, no MUJOCO_LOG.TXT |
| `validate` | OK, 8 episodes, 1288 frames, 6 tasks, 16 videos, physically_valid=True |
| `sim --families all --episodes 16 --people 2 --seed 1` | 16/16: counter_to_sink 3/3, open_drawer 3/3, open_door, close_drawer, close_door, put_in_drawer, wipe_counter 2/2 each; counter_to_counter skipped; dataset validates |
| `sim --families open_door,close_door --episodes 4 --people 2` | 4/4; the cabinet doors end at 1.24 to 1.27 rad (71 to 73 deg) |
| Steps 6 to 8 | covered by the suite (`tests/test_dataset.py`, `tests/test_eval.py`, `tests/test_worker.py`); not rerun by hand |
| Claims | fixed in this commit, listed in BUILD-LOG |
