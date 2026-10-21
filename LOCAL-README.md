# What is left, as of 2026-10-21

Everything doable on a laptop is built and on `feat/demo-site` (pushed). The to-do list is `site/data/todo.json`; run `python scripts/render_todo_readme.py` after editing it, and the README and the demo site both follow.

Setup note: this machine has a stray `tests` package in `~/miniconda3/lib/python3.13/site-packages`. Use a venv (`python -m venv .venv && .venv/bin/pip install -e ".[dev,sim]"`) before running the suite.

## 1. Land it (needs you)

1. Merge `feat/demo-site` into `main` (open a PR and merge). The attempt to fast-forward `main` from this session was denied by the permission check, so it has not been done. Once `main` contains the branch, PRs #96 to #100 and #102 to #117 close as merged.
2. Close the superseded PRs #25, #27, #33, #36, #49, #57, #75, #77. Decide on #101 (it ignores `site/`, which the branch tracks).
3. Close the stale issues #18, #22, #24, #26, #28, #30, #32, #34.
4. Flip the merged rows in `site/data/todo.json` from `review` to `done`, then regenerate the README.

## 2. Check by hand (laptop, about an hour)

| Check | How |
|---|---|
| Fresh clone on a clean venv | `make install && make test`, then `make demo` in under 5 minutes |
| Two workers on one queue | `arrival serve`, two `arrival work`, upload the hero scan twice |
| Shard, kill, resume | `arrival sim --shard 0/2`, interrupt, `--resume`, `merge-shards`, `validate` |
| The demo site in a browser | `make site`, open http://localhost:8765 (P4) |
| The job page buttons | upload, then Cancel and Delete (L30) |
| Docker | `docker compose up --build`, then `python scripts/smoke_hosted.py http://127.0.0.1:8000 --wait 600 --require-healthz` |

## 3. Known gaps from the laptop work

- Open-door is 32/32 over seeds 0, 5, 7 and 11 with two people (Oct 21); the earlier walker contacts have not recurred since L27. On `counter_b` link7 still grazes the east wall (22 steps, 0.9 mm) every episode.
- 3 of 28 episodes in the G1 run (seed 5, all families) are flagged for a joint at a limit, down from 5: open_oven twice (joint 5 on its stop for 71 steps; every flap tilt and grasp the expert has hits a stop from the fixed pedestal) and one close_door on `counter_a` (joint 7 for 43 steps on the approach).
- The mirrored right-hinged fridge plan is not built; the scene only has a left hinge.
- The memory cap (`ARRIVAL_MAX_JOB_MB`) was not measured on Linux.
- `site/data` was not re-exported after the merged PRs, so its numbers are from the earlier pipeline run.

## 4. Blocked (needs a GPU, cloud account or people)

GPU: G1 fine-tune, G2 real server, G3 three-cell eval, G4 baseline, G7 MimicGen, G8 splat composite. G5's scripts are written and never run. Cloud: C1 real run, C2 deploy, C3 real cost, C4 detection, C5 Cosmos. People: P1 iOS recorder field test, P2 three consented real scans, P3 twenty teleop demos (the timer is built).

Anything that needs real data reports "not measured". Nothing counts as done until its PR is merged on `main`.
