# Reference

Detailed usage for Arrival. The overview is in the [README](../README.md).

## Honest label: the `demo` dataset

These bullets are about the `demo` dataset (`out/datasets/<room>`). The MuJoCo `sim` dataset is described under [Simulation (MuJoCo)](#simulation-mujoco) and carries its own flags in `meta/info.json`.

- The dataset folder is a real GR00T LeRobot v2 layout (parquet, mp4, `meta/*.json`).
- **The motion inside it is a scripted placeholder in simulation.** It comes from `arrival_generator = "scripted_retarget"`: seed trajectories retargeted to new object poses. It is **not physically valid**: the `demo` path uses no MuJoCo and no contact physics. It is **not MimicGen** output.
- Frames are 64×64 `sim_texture` schematics. **No Gaussian splat** is trained or composited.
- **No GR00T training** was run, and there is **no real-robot result**.
- `meta/info.json` records `"arrival_physically_valid": false`. `meta/embodiment.json` records `"joints_are_placeholder": true`, because the `single_arm` joint values are zeros until a physics generator fills them.
- **Do not fine-tune GR00T (or any policy) on this data while `arrival_physically_valid` is false or `joints_are_placeholder` is true.**

## Install

Requires Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"        # demo, page, tests
python -m pip install -e ".[dev,sim]"    # + MuJoCo for `python -m arrival sim`
python -m pytest -q
```

## Run

Smoke demo, 4 episodes, under a minute:

```bash
python -m arrival demo --episodes 4
python -m arrival validate out/datasets/hero_kitchen
```

The full demo writes 1,000 successful episodes covering both task families, with an mp4 for every episode (about 30 s on a laptop):

```bash
python -m arrival demo --episodes 1000
python -m arrival validate out/datasets/hero_kitchen
```

`--videos` defaults to `all`, which is what GR00T's loader needs. `--videos sparse` or `none` drops the video feature from the dataset and writes human-only previews to `out/datasets/<room>_previews/`.

Open the page:

```bash
python -m arrival serve        # http://127.0.0.1:8000
```

| Route | What it does |
|---|---|
| `GET /` | Lists rooms under `scans/` with `geometry_source`, task counts per family, dataset path, and download link. Has an upload form |
| `GET /rooms/<id>` | Fixtures, the two task families, `arrival_physically_valid`, and the run report |
| `GET /download/<id>` | Zip of `out/datasets/<id>` |
| `POST /jobs` | Multipart upload with file field `scan` (max 200 MB, larger gets 413). Writes `out/jobs/<id>/` and returns `202 {"status":"queued",...}`. Nothing is compiled or generated in the request; `python -m arrival work` drains the queue |
| `GET /jobs/<id>` | The job's `status.json` |
| `GET /jobs/<id>/download` | `out/jobs/<id>/dataset.zip` once the job is `done`, else 404 with the current status |

The index page lists every job under `out/jobs/` with its status, stage, room id, episode count and a download link. The server binds `127.0.0.1` only and has no authentication. Uploads are streamed to a temp file; a body shorter than `Content-Length`, a missing closing boundary, an empty file, or a part with no filename gets 400. If `out/datasets/<room>_sim` exists, the room page links it and shows its `arrival_generator`, `arrival_physically_valid`, episode count and families.

From a RoomPlan export instead of the authored scan (`scans/<room>/roomplan.json`):

```bash
python -m arrival demo --room hero_kitchen --from-roomplan --episodes 4
```

The normalizer unwraps Apple's keyed enums (`{"sink": {}}` -> `sink`, `washerDryer` -> `washer_dryer`), drops `low`-confidence objects, puts the floor at z = 0 (from `floors[]`, else the lowest wall bottom), rotates the room so its longest wall runs along +X with the wall bounding box at (0, 0), and snaps near-90° yaws. RoomPlan captures no small objects, so a mug, a bowl, a plate, a cup and an apple are placed (seeded, each at a seeded size within 8% of its catalog size; one that does not fit the counter is skipped) on the counter nearest a sink; nothing is borrowed from the authored scan. A scan with no sink, stove or counter-like fixture yields no tasks and the command says so. `--scans` defaults to `./scans`, falling back to the repo's `scans/` when run from elsewhere.

Doors, windows and openings are kept in the bundle and twin with the id of their wall, `polygonCorners` floors set the floor extent, and a merged multi-story export yields one story (`roomplan.load(path, room, story=n)`; default the story with the most objects). To eyeball every room as a top-down plan:

```bash
python scripts/render_rooms.py                              # out/renders/<room>_top.png and <room>_roomplan_top.png
python scripts/render_rooms.py tests/fixtures/roomplan/amal.json
```

## Simulation (MuJoCo)

`python -m arrival sim --episodes N` builds a MuJoCo scene from the compiled twin and records episodes to `out/datasets/<room>_sim` (LeRobot v2 layout). Needs the `sim` extra. The code lives in `arrival/sim/`:

- **Articulated kitchen** from the twin: counters with drawers (slide joints), cabinet and wall-cabinet doors (hinge joints), a sink with a faucet lever, a stove with a bottom-hinged oven door, and fridge, dishwasher and microwave doors. `arrival.sim.scene.part_names(twin)` lists every joint with its handle site and an open position. Objects collide with the moving parts, so a mug rests on a drawer tray or an open oven door.
- **Arm**: a procedurally built 7-DoF Panda-like arm (not the Franka asset) driven through IK, so joint states in the dataset are the simulated joint positions, plus a parallel gripper that holds objects by friction contact.
- **People**: kinematic bodies walking seeded loops through the room: a free point, a lane that crosses behind the robot base, a 2 to 4 s stop at the counter, back out. They move and occlude; they are not simulated humans and they do not collide. `arrival/sim/safety.py` measures the horizontal clearance between every robot collision geom and each person every control step; an episode whose minimum reaches 0 is a human contact and is dropped from the dataset even when the task succeeded. The value is written per episode as `arrival_min_human_distance_m` and `arrival_human_contact` in `meta/episodes.jsonl`.
- **Cameras**: rendered with MuJoCo's offscreen renderer.
- **Policies** behind one interface: a scripted expert (default), a VLM planner via Nebius Token Factory (`--planner vlm`; needs `TOKEN_FACTORY_API_KEY`, falls back to the rule parser without one, and caches responses in `arrival/models/nemotron_cache.json`), and a VLA client for a GR00T policy server (placeholder; no server is shipped or run here).
- **Success is checked in physics** (object resting in the target region, drawer/door joint past threshold), not asserted by the script.

Run it:

```bash
python -m pip install -e ".[dev,sim]"
python -m arrival sim --episodes 8 --people 2 --save-preview   # contact sheet + mp4 in out/previews/
python -m arrival validate out/datasets/hero_kitchen_sim
```

Status as of Oct 17 (scripted expert, friction grasps, physics-checked; 16/16 over these seven families with `--families all --episodes 16 --people 2 --seed 1` on a fresh clone):

| Family | Success | Note |
|---|---|---|
| `counter_to_sink` | 100% | mug, and the bowl held by its rim |
| `open_drawer` | 100% | both counters |
| `open_door` | 100% | both cabinet doors, 8/8 over two seeds; the door ends at about 70° (threshold 60°) |
| `counter_to_counter` | skipped | pick point and target about 2 m apart, base is fixed per episode. Needs a mobile base |
| `close_drawer` | 100% | starts open 0.30 m, pushed shut on the handle bar; success is the slide joint under 0.03 m |
| `close_door` | 100% | starts open 70°, pushed shut along its arc; success is the hinge under 5° |
| `put_in_drawer` | 100% | drawer starts open 0.14 m, mug set on the exposed part of the tray; success is the mug resting inside the tray |
| `wipe_counter` | 100% | sponge pinched and dragged two strokes; success needs 0.4 m travelled pressed to the top and the sponge back on it |

Defaults are `counter_to_sink,open_drawer`. `--families all` runs every family; `counter_to_counter` is skipped with a log line until there is a mobile base (MAN-3), and any failed attempt is dropped and counted. `--planner vlm` sends each instruction through the Nemotron planner (POL-1), `--instructions nemotron --split train|eval` draws generated wordings with a held-out set (POL-2). The VLA client has never run against a real GR00T server.

### Closed-loop eval

```bash
python -m arrival eval --policy scripted --families counter_to_sink,open_drawer --episodes 4 --people 2
python -m arrival eval --holdout            # wording template 4 and the [0, +0.08] start offset (docs/07-eval.md)
python -m arrival eval --policy vla         # needs GR00T_POLICY_URL; exits 2 with a message when unset or unreachable
```

`eval` runs `--episodes` episodes per family through the same `SimEnv` and `run_episode` as `sim`, writes no dataset, prints a table, and writes `out/eval/<room>_<policy>.json` (shape in [04-contracts.md](04-contracts.md)). Per family: episodes, successes, success rate with a Wilson 95% interval, mean sim seconds (frames / 20), mean wall seconds, human contacts, the smallest robot-human distance seen, and the largest IK residual. A human contact counts as a failure. The eval is sim-in-the-twin only; nothing here scores a real robot.

What this is not: MuJoCo frames are not a Gaussian splat of the room, episodes are not MimicGen output, and no policy was trained on them. Read `arrival_generator` and `arrival_physically_valid` in the sim dataset's `meta/info.json` before using it.

## Upload worker

`python -m arrival work` runs in a second terminal next to `serve`. It polls `out/jobs/*/status.json` every `--poll` seconds (default 2), takes the oldest `queued` job, marks it `running`, and runs these stages, rewriting `status.json` at each one: `normalize` (the upload is a RoomPlan CapturedRoom export if it has `walls`/`objects` with `transform`, or a `scan.json` if it has `schema_version`; anything else fails here), `twin` (`compile_twin`), `objects` (`place_default_objects` when the scan has no objects, as RoomPlan exports never do; then `out/twins/<room>/`), `sim` (`arrival.sim.collect.collect`, same options as `sim`: `--episodes 8`, `--families counter_to_sink,open_drawer`, `--people 1`, `--res 128`, `--seed 0`), `validate`, and `zip` (`out/jobs/<id>/dataset.zip` holding `<room>_sim/`). The room id is the upload's filename stem in `[a-z0-9_]` plus the job id, so two uploads of `kitchen.json` never share a twin or dataset folder. A job with zero successful episodes or a failing validator ends `failed`. Failures are written to `status.json` (`error`) and `error.txt` (traceback); the worker then moves to the next job. `--once` processes at most one job and exits. Needs the `sim` extra; without MuJoCo every job fails at `sim`.

```bash
python -m arrival work                   # until Ctrl+C
python -m arrival work --once --episodes 2
```

Measured on a laptop CPU: a `scan.json` upload with `--episodes 2` took 5.9 s end to end, 2/2 episodes, 322 frames, zip 77,624 bytes. One worker runs at a time; there is no lock against two workers claiming the same job, no retry, and nothing runs on Nebius (OPS-2).

## Output layout

`out/` is gitignored. Nothing under it is committed.

```
out/
  twins/hero_kitchen/       twin.json, layout.yaml (RoboCasa subset), tasks.json
  datasets/hero_kitchen/    GR00T LeRobot v2
    meta/info.json          arrival_generator, arrival_physically_valid, arrival_successes, ...
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/modality.json      single_arm [0,7], gripper [7,8], ego_view, task description
    meta/embodiment.json
    meta/episodes_stats.jsonl, meta/stats.json
    meta/arrival_cartesian.jsonl   cartesian sidecar (eef/object traces)
    data/chunk-XXX/episode_XXXXXX.parquet   (1,000 episodes per chunk)
    videos/chunk-XXX/observation.images.ego_view/episode_XXXXXX.mp4
  datasets/hero_kitchen_sim/ MuJoCo sim dataset (same layout, ego_view + wrist videos)
  previews/                 sim contact sheets and overview mp4s (--save-preview)
  reports/hero_kitchen.json attempts, successes, dropped, seconds, demos/hour, dollars
  eval/hero_kitchen_scripted.json   closed-loop eval: per-family success, Wilson 95%, time, human contacts
  jobs/<job_id>/            upload.bin, status.json; after `work`: dataset.zip (done) or error.txt (failed)
  twins/<stem>_<job_id>/, datasets/<stem>_<job_id>_sim/   what `work` compiled and simulated for an upload
```

The demo dataset's embodiment label is `robocasa_panda` (7 joints + 1 gripper, placeholder joints) and its two task families are `counter_to_sink` and `counter_to_counter`; the sim dataset is labelled `arrival_sim_panda` and lists its families in `arrival_families` (see the sim dataset notes in [04-contracts.md](04-contracts.md)). Both vary by object, start pose, lighting, and instruction wording.

## Credentials

The demo uses no credentials and makes no network calls. Later Nebius Token Factory wiring reads these environment variables, and only inside `arrival.models.token_factory`:

| Variable | Purpose |
|---|---|
| `TOKEN_FACTORY_API_KEY` | Bearer token |
| `TOKEN_FACTORY_BASE_URL` | Defaults to `https://api.tokenfactory.nebius.com/v1`. Check it against the live docs |
| `NEMOTRON_OMNI_MODEL`, `NEMOTRON_ULTRA_MODEL`, `NEMOTRON_SUPER_MODEL`, `NEMOTRON_LIGHTNING_MODEL` | Optional model ids |

Export them in your shell, or put them in an untracked `.env` (gitignored) in the repo root or cwd: `arrival.models.token_factory` reads it itself, only when `TOKEN_FACTORY_API_KEY` is unset, and never overrides variables already set. No other module loads `.env`. Never commit keys. Account links and what to request are in [docs/08-access.md](08-access.md).

## Cost

None of the Nebius products below were called. The only measured number is the local generation run, which cost $0 on a laptop CPU. That says nothing yet about how generated data compares with teleoperation, because the comparison has not been measured.

| Step | Nebius product | Measured USD |
|---|---|---|
| Read the scan | Token Factory, Nemotron 3 Nano Omni | not measured |
| Assemble the twin, fill gaps | Token Factory, Nemotron 3 Ultra | not measured |
| Propose tasks | Token Factory, Nemotron 3 Super | not measured |
| Wordings | Nemotron 3.5 Lightning | not measured |
| People and episode labels | Nemotron 3.5 Lightning | not measured |
| Build the twin and appearance | Serverless AI Job, 1 GPU | not measured |
| Generate demos | Serverless AI Jobs, many | not measured (local generation: $0, see `out/reports/hero_kitchen.json`) |
| Restyle frames | Serverless Endpoint, Cosmos 3 | not measured |
| Fine-tune | Serverless AI Job, H100, GR00T N1.7 | not measured |
| Store artifacts | Object Storage | not measured |
| Host the app | Serverless Endpoint, CPU | not measured |

## Later: GR00T fine-tune (not run)

This step comes later. It was **not** run, and nothing in this repo was trained. **Do not run this while `joints_are_placeholder` is true** (or while `arrival_physically_valid` is false). It needs 40 GB+ VRAM and the Isaac-GR00T repo:

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path out/datasets/hero_kitchen \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path <to-be-written> \
  --num-gpus 1 \
  --output-dir out/checkpoints/gr00t \
  --max-steps 2000 \
  --global-batch-size 32
```

## Known gaps

Not built yet, and not claimed anywhere in the outputs:

- iPhone capture app (LiDAR + RoomPlan). Uploads are processed only by a local `python -m arrival work` process.
- MimicGen generation (`arrival/generate/mimicgen.py` is a stub).
- Gaussian splat training or compositing (`arrival/render/splat.py` is a stub).
- Cosmos restyle of frames.
- Nebius Serverless fan-out of generation jobs.
- GR00T fine-tune (command above, not run). `python -m arrival eval` scores policies in the sim; it has only been run with the scripted expert.
- RoboCasa-native layout conventions: `layout.yaml` is a subset emitter (world-frame `pos`, `z_rot` on oblique walls), not loadable by RoboCasa as-is.

