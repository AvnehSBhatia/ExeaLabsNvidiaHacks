# DEM-4: one-command demo

`python scripts/demo_all.py --quick` runs the whole story in one process and prints PASS or FAIL with wall time per stage:

1. scan normalize + compile (`arrival.cli.run_demo`, one placeholder episode, no video)
2. twin layout PNG (`scripts/render_rooms.render`) to `<out>/renders/<room>_top.png`
3. sim dataset (`arrival.sim.collect.collect`, families `counter_to_sink,open_drawer`, previews on)
4. validate (`arrival.export.validate.validate`)
5. loader check (`arrival.export.loader_check.check_loader`: the files a LeRobot v2 loader reads, with the mp4s decoded)
6. closed-loop eval of the scripted expert, 2 episodes per family (`arrival.eval.harness.evaluate`)
7. contact sheet check (`<out>/previews/<room>_sim_contact.png`, written by stage 3)

| Flag | Meaning |
|---|---|
| `--quick` (default) | 2 episodes, 1 person, 64 px; about 25 s, 0.58 GB peak RSS on a laptop |
| `--full` | 8 episodes, 2 people, 128 px |
| `--room` | any room under `--scans` (default `hero_kitchen`) |
| `--out`, `--scans` | as in `python -m arrival` |
| `--serve` | start the local page after every stage passed |

Outputs: `<out>/demo_summary.md` and `<out>/demo_summary.json` (dataset path, episodes, frames, `physically_valid`, `joints_are_placeholder`, per-family eval successes, demos/hour, the honest label, and cost rows left `null`). Exit code 1 if any stage fails (a one-line `demo_all: FAIL <stage>: <reason>` goes to stderr), 0 otherwise.

`joints_are_placeholder` is read from `meta/embodiment.json`, where the writers put it, not from `info.json` (which carries `arrival_physically_valid`). Demos/hour is the collection rate on whatever machine ran it. The label is: scripted expert in simulation, no policy trained.

Test: `python -m pytest -p no:cacheprovider tests/test_demo_all.py`.
