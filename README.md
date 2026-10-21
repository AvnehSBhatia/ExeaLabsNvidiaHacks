# Arrival

**Scan a room once. Get robot training data for that exact room.**

[![Arrival demo: scan → digital twin → MuJoCo simulation → GR00T dataset](docs/media/demo.gif)](docs/media/demo.mp4)

*30-second demo, rendered from the real code (an early cut that predates the SIM-1 and MAN-1 visual fixes; DEM-2 refreshes it). Click for the full-quality mp4. Regenerate it with `python scripts/make_demo_video.py`.*

---

## What this is

General robot policies like NVIDIA GR00T are trained on generic kitchens. When a robot arrives in *your* kitchen, it has never seen your counters, your sink or your drawers. Arrival closes that gap:

1. **Scan.** An iPhone LiDAR scan (Apple RoomPlan) describes the room: walls, counters, sink, stove, fridge.
2. **Twin.** We compile the scan into a digital twin: a RoboCasa-style layout and a MuJoCo scene with working drawers and doors.
3. **Simulate.** A 7-joint Panda-style arm does tasks in the twin while people walk around. Grasps use friction physics, and success is checked by the simulator.
4. **Capture.** Every successful episode is written as a [GR00T](https://github.com/NVIDIA/Isaac-GR00T)-compatible LeRobot v2 dataset: joint states, actions, ego and wrist camera video, and language instructions.
5. **Fine-tune** *(next)*. Train GR00T N1.7 on that room's data so the robot is useful on day one.

Built for the NVIDIA × Nebius hackathon by Exea Labs and a team of 30. **Want to help? Jump to [Join in](#join-in-team-plan-for-30-people).** The product spec is [plan.pdf](plan.pdf).

## Does it work? (status, Oct 20 2026)

| Stage | Status |
|---|---|
| Scan → twin | ✅ Works on the sample kitchen (now tiled, no overlapping fixtures) and on 2 real public iPhone RoomPlan exports; a scan regression suite covers every room under `scans/`. The iPhone capture app (`ios/`, CAP-1) is merged but untested on a real kitchen here; field scanning protocol in `docs/SCANNING.md` |
| Simulation: kitchen, drawers, doors, oven, fridge, dishwasher, microwave, wall cabinets, faucet, people | ✅ Works. MuJoCo, built from the twin; every moving part is a named joint with a handle site. Panda-style arm and kitchen visuals (recessed fronts, slim handles, sink, gas range) landed with SIM-1 and MAN-1 |
| People: interactions + safety metric | ✅ Walkers stop at the counter and cross behind the robot. Every episode records the minimum robot-human distance; episodes with contact are dropped. A walker opens the fridge, stands in the doorway and closes it, and the held door counts for the safety metric |
| Closed-loop eval harness | ✅ `python -m arrival eval` scores N episodes per family (success, time, human contacts, Wilson 95%). Counts arm-fixture contacts too; held-out seeds are offset from training and an episode that raises counts as a failure. ⚠️ Only the scripted expert has been scored; the VLA cell needs a GR00T server (POL-3) |
| Robot: pick mug → sink | ✅ 100% success, friction grasp, physics-checked |
| Robot: open drawer | ✅ 100% success |
| Robot: open door | ✅ The arm opens both cabinet doors: 32/32 with `eval --families open_door --episodes 8 --people 2` at seeds 0, 5, 7 and 11 (Oct 21), 0 human contacts, closest pass 0.043 m. The walker contacts on `counter_a_door` in the earlier check (3/4 and 1/2) have not recurred since walkers yield to the arm (L27). Opens to about 63-66°: the hand holds its heading while the bar turns between the pads, and joint 7 stays inside its limit. The arm is mounted at 0.75 m, so a mobile base (L13) is what would open it wider. ⚠️ On `counter_b`, link7 grazes the east wall for 22 steps (0.9 mm deep) in every episode: the stand spot is close to that wall |
| Robot: counter → counter | ✅ With `--base mobile` (L13, in review, not on `main` yet): the pedestal drives along the counter front at 0.3 m/s and sets the object down 2 m away, 6/6 with two people walking in the hero kitchen. The default base is still fixed, and skips the family |
| Robot: close drawer, close door, put mug in drawer, wipe counter | ✅ 100% each (32/32 across all families, seed 5, two people walking). The part starts open for the close tasks; the sponge must travel 0.4 m pressed to the top |
| Robot: bowl | ✅ Held by the rim (one finger inside the wall). 4/4 bowl → sink. The bowl is now hollow in physics |
| GR00T-format dataset + validator | ✅ Passes our validator, and the layout matches GR00T's loader source. ⚠️ Not yet loaded by GR00T itself since the latest fixes |
| Local web page: browse, download, upload | ✅ Works. Redesigned page: room cards with previews and stats, drag-and-drop upload, live job stages. Uploads are processed by `python -m arrival work` on the same machine (twin, sim dataset, zip); a killed worker's jobs are requeued. Hardening, worker limits, shardable sim, frames upload and a static export are in review (L3, L4, L5, L14, L23). ⚠️ No cloud fan-out yet; the page was checked with curl and tests, not in a browser |
| VLM planner (Nemotron) / VLA client (GR00T server) | ✅ VLM planner (Nemotron) wired into sim collection with Token Factory and offline cache fixture (POL-1); dynamic instruction generation with held-out eval set (POL-2). ⚠️ VLA client tested offline |
| GR00T fine-tune, MimicGen, splats, Cosmos, Nebius scale-out | ❌ Not built yet. This is what remains for the hackathon: the GR00T fine-tune (DAT-2), the GR00T policy server (POL-3) and real scans (CAP-3) |

There are two datasets. `python -m arrival sim` writes **physics-simulated** data with real joint states (`arrival_physically_valid: true`). `python -m arrival demo` writes a fast **scripted placeholder** with zero joint values (`arrival_physically_valid: false`). Never train on the placeholder.

## To-do

<!-- todo:start (generated by scripts/render_todo_readme.py from site/data/todo.json) -->

As of 2026-10-21: **52** shipped, **0** in review, **0** next, **18** blocked, of **70** items. 44 pull requests merged, 28 open. The same list is on the demo site (`site/data/todo.json`). A row is shipped only when its pull request is merged on `main`.

Sizes: S under half a day, M about a day, L several days. Priority: **must** (the demo or the core claim breaks without it), **should**, nice. Anything that needs real data reports "not measured" until the blocked rows land.

### Shipped on main (22)

Merged PRs, grouped by what they do.

| id | what | PRs |
|---|---|---|
| S01 | Sample kitchen tiles the wall with no overlapping fixtures | #5 |
| S02 | Scan regression suite and full RoomPlan coverage, two real public exports | #10 |
| S03 | Kitchen and Panda arm visuals; wrist camera framing and joint 7 off its stop | #2, #71 |
| S04 | Open drawer and open door, with collision boxes that match the drawn fronts | #4, #47 |
| S05 | Close drawer, close door, put mug in drawer, wipe counter; a bowl the gripper can hold | #11 |
| S06 | Plates by the rim and pans by the handle | #31 |
| S07 | The arm opens the fridge: pull by the bar, push the inside face | #78 |
| S08 | Oven, dishwasher, microwave and wall-cabinet doors, faucet lever, nine more objects | #19 |
| S09 | Key, fill and window lights, tile, wood and stone materials | #21 |
| S10 | Walking people: proportioned bodies, a gait, slow stops at the counter | #23, #76 |
| S11 | A person opens the fridge and the held door counts for safety | #73 |
| S12 | Closed-loop eval harness, human-contact metric, arm-fixture contacts | #13, #55 |
| S13 | Upload worker: queued scan to twin to sim dataset to zip, with atomic job status | #12, #51 |
| S14 | GR00T modality config; validator rejects NaN, inf and bad timestamps | #38, #52 |
| S15 | Nemotron planner in sim collection; Tavily appliance research | #3, #35 |
| S16 | iOS RoomPlan capture app and field scanning protocol | #43, #44 |
| S17 | CI workflow, Makefile and CPU Dockerfile; one-command demo pipeline | #82, #86 |
| S18 | Offline GR00T policy-server contract (mock server, shim, VLAClient fix) and loader conformance check | #83, #84 |
| S19 | Scan intake, RGB-D frame format, tape-measure accuracy harness, realism metrics | #87, #90, #92, #88 |
| S20 | Time and cost report, seed-replay retargeting, appearance-variation stage with honest labels | #85, #89, #93 |
| S21 | Docs and README claim checker, release QA, PyAV fix, refreshed demo video | #91, #29, #9, #50 |
| S22 | Integration of 16 PRs and the sprint PRs #82-#93, plus the to-do docs | #80, #94, #1, #81, #95 |

### In review (30)

Pull requests against main. All are merged into feat/demo-site and none is on main yet, so none counts as done.

| id | what | size | priority | status | PRs or issue |
|---|---|---|---|---|---|
| L1 | Fixtures on any wall (#49, #75, #77 folded into one PR) Replaces #49, #75 and #77. | M | must | Shipped | #99 |
| L2 | Failed task skips instead of crashing; pre-flight reach check | S | must | Shipped | #99 |
| L3 | Worker hard limits: subprocess timeout, size caps, stage-tagged errors | M | must | Shipped | #96 |
| L4 | Harden the web app for public hosting | M | must | Shipped | #97 |
| L5 | Shardable, resumable sim collection: --shard, --resume, merge-shards | M | must | Shipped | #104 |
| L6 | Fresh-clone matrix: macOS and Windows in CI, wheel smoke, docker job, Windows checks fixed, separate rendering gate | M | must | Shipped | #98, #108 |
| L7 | Docs and repo hygiene sweep Demo video and gif regenerated and the README status rows checked; the open-door row now states its measured rate. | S | must | Shipped | #98 |
| L8 | One-command offline demo from a fresh venv, GL fallback | S | must | Shipped | #98 |
| L9 | Non-kitchen rooms with room-appropriate objects (book, remote, laptop) Replaces #57. | M | should | Shipped | #105 |
| L10 | Twin compile fixes on the real U-shaped scan | S | should | Shipped | #100 |
| L11 | Scan robustness sweep: fuzzed RoomPlan variants | M | should | Shipped | #100 |
| L12 | Per-episode quality validator | S | should | Shipped | #102 |
| L13 | Mobile base on any wall: 11-float state and action through export, GR00T config and loader check Replaces #33 and #36. The fixed base stays the default; run the fixed-base GR00T check (G1) before training on the 11-float schema. | L | should | Shipped | #103 |
| L14 | Scan frames through the upload path, compared against held-out views in the worker | S | should | Shipped | #97, #110 |
| L15 | Provenance on twin objects: detected, default, authored | S | should | Shipped | #100 |
| L16 | Realism command: JSON and side-by-side PNG (real frames still missing) | S | should | Shipped | #100 |
| L17 | Reachability planner: IK solve of the expert's waypoints, fallback base grid | L | should | Shipped | #116 |
| L18 | Retention and deletion | S | should | Shipped | #96 |
| L19 | Worker concurrency: atomic claim, heartbeat, cancel | M | should | Shipped | #96 |
| L20 | Usage metering for the cost report; timed keyboard teleop baseline | M | should | Shipped | #96, #113 |
| L21 | CI extras: ruff, link check, Dependabot | S | should | Shipped | #98 |
| L22 | Secrets handling: single config loader, .env.example, redaction, arrival doctor | S | should | Shipped | #97, #114 |
| L23 | Static demo-site export | S | should | Shipped | #102 |
| L24 | Oven, dishwasher and faucet families on the any-wall chain Replaces #27. | M | nice | Shipped | #106 |
| L25 | Real Menagerie Franka behind --arm menagerie Replaces #25. The 34 MB meshes are fetched by script and stay out of git. | M | nice | Shipped | #107 |
| L26 | Thin-object (spatula) pinch grasp, in counter_to_sink and wipe, both arms | M | nice | Shipped | #117 |
| L27 | People pass: yield and stop from the safety distance, counter-use pause, 3D head and arm clearance check | M | nice | Shipped | issue #24 |
| L28 | Multi-room summary command (real rooms wait on P2) | S | nice | Shipped | #102 |
| L29 | Frozen evaluation trials: replay the same trials for any policy and keep every run | M | should | Shipped | #109 |
| L30 | Worker follow-ups: memory cap on Linux, DELETE /jobs/<id> with a retention note, cancel endpoint and per-episode progress | M | should | Shipped | issue #68 |

### Laptop: next (0)

Can be done on a laptop with no extra access.

| id | what | size | priority | status | PRs or issue |
|---|---|---|---|---|---|

### Needs a GPU (8)

A GPU host with 40 GB or more, or a Linux box.

| id | what | size | priority | status | PRs or issue |
|---|---|---|---|---|---|
| G1 | Real GR00T loader check and 100-step fine-tune on hero_kitchen_sim Offline slice done: 28 of 28 episodes at 224 px, validator passes, loader check passes with 4 warnings. GR00T itself has not loaded the data. | M | must | Blocked | issue #63 |
| G2 | Real GR00T policy server: chunking, both camera keys, golden-file test Wire-format golden test and the 8-float and 11-float schemas are done. The shim has not run against a real GR00T server. | M | must | Blocked | none |
| G3 | Three-cell held-out eval, 50 episodes per cell, Wilson intervals | L | must | Blocked | issue #64 |
| G4 | Generic-RoboCasa baseline dataset and fine-tune | M | must | Blocked | issue #64 |
| G5 | Dockerfile.gpu, run_finetune.sh, train and serve targets Dockerfile.gpu, scripts/run_finetune.sh and the train, eval-policy and serve-policy targets are written and never run on a GPU. Unverified items are listed in the build log. | L | must | Blocked | issue #63 |
| G6 | Prove layout.yaml loads in real RoboCasa Offline checker scripts/check_layout.py compares layout.yaml with the twin. Loading in real RoboCasa is not checked. | M | must | Blocked | issue #69 |
| G7 | Real MimicGen retargeting for counter_to_sink | L | should | Blocked | issue #62 |
| G8 | Scan-appearance composite replacing the splat stub | L | should | Blocked | issue #61 |

### Needs cloud (5)

Cloud accounts, billing or deploy access.

| id | what | size | priority | status | PRs or issue |
|---|---|---|---|---|---|
| C1 | Nebius Serverless Jobs fan-out with resume, then one real small run Fan-out client with a local backend, done-markers, retry and resume is built and tested. No Nebius client exists. | L | must | Blocked | issue #66 |
| C2 | Hosted HTTPS demo with a pre-seeded room Smoke test (#115), /healthz, docker-compose.yml and docs/DEPLOY.md are written and tested on a laptop. TLS, the domain, a real seeded room and a run on a host are still blocked. | M | must | Blocked | #115 |
| C3 | Measured time and cost per demo against the teleop baseline | S | should | Blocked | issue #67 |
| C4 | Detect small objects from posed frames with a VLM | M | should | Blocked | issue #60 |
| C5 | Real Cosmos variation on a tenth of validated frames | M | nice | Blocked | issue #70 |

### Needs people or an iPhone (5)

Physical rooms, a LiDAR iPhone, or a human.

| id | what | size | priority | status | PRs or issue |
|---|---|---|---|---|---|
| P1 | Field-test the iOS app on a LiDAR iPhone; RGB-D recorder in Swift | L | must | Blocked | issue #58 |
| P2 | Three consented real kitchen scans with tape measurements; licence check of the two public scans | L | must | Blocked | issue #59 |
| P3 | Twenty timed teleoperated counter_to_sink demos The timer tool is in review as part of L20 (#113); the demos need a person. | S | should | Blocked | issue #67 |
| P4 | Refresh the pitch video with real footage; check the demo page in a browser | S | should | Blocked | none |
| OWN | Close or update stale issues #18, #22, #24, #26, #28, #30, #32, #34 Repo owner only | S |  | Blocked | none |

### Pull requests not merged as they are

- #25, #27, #33, #36, #49, #57, #75, #77: Rebased into #96 to #110. Close them once those merge.
- #101: #101 ignores site/, which this branch tracks. Left out on purpose.

<!-- todo:end -->

### Known limits

- The GR00T loader has not read our dataset since the latest fixes; only our own validator and the offline conformance check have (G1).
- The mobile base (L13) is behind `--base mobile` and the real Franka behind `--arm menagerie`; the defaults are still the fixed Panda-style arm. Door grasps with the real meshes are not retuned.
- A corner fixture whose only stand spot is outside the room stays unreachable with a fixed base; the reach planner (L17) would replace the horizontal-envelope pre-flight.
- The web page was checked with curl and tests only, not in a browser. Uploads are processed by local workers that share one `out/` folder.
- No real kitchen has been scanned and run end to end, and nothing has trained on the data.

## Get started (5 minutes)

Requires Python 3.11+ on macOS, Linux or Windows (the suite and the MuJoCo sim run as is on Windows; use `.venv\Scripts\activate` there). No GPU and no API keys needed.

```bash
git clone https://github.com/AvnehSBhatia/ExeaLabsNvidiaHacks.git
cd ExeaLabsNvidiaHacks
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,sim]"
python -m pytest -q                       # should be all green
```

On a machine without an OpenGL context, `python -m pytest -q --no-render -rs` runs the portable checks and reports the rendering skips. CI requires these checks on Linux, macOS and Windows, plus the full suite and offline demo in a separate Linux rendering job. On Windows, the full suite and sim need a desktop session with a working graphics driver.

Run the simulation and look at what it recorded:

```bash
python -m arrival sim --episodes 8 --people 2 --save-preview
python -m arrival validate out/datasets/hero_kitchen_sim
open out/previews/                        # contact sheet + overview video
```

Then browse it in the local page:

```bash
python -m arrival serve                   # http://127.0.0.1:8000
python -m arrival work                    # in another terminal: turns uploaded scans into sim datasets
```

Other commands:

```bash
python -m arrival demo --episodes 1000                         # fast scripted placeholder dataset
python -m arrival demo --room hero_kitchen --from-roomplan     # build from a RoomPlan JSON export
python -m arrival sim --families all                           # include the not-yet-working tasks
python -m arrival eval --episodes 4 --people 2                 # score a policy closed-loop, no dataset written
```

Each evaluation writes a new `out/eval/<run_id>/` with `manifest.json`, `episodes.jsonl` and `report.json`. Replay the printed manifest with `--manifest PATH --policy vla --checkpoint-id LABEL` to compare checkpoints on the same twin and ordered trials. The checkpoint label is supplied by the caller; it is not verified against the server. See [the evaluation protocol](docs/07-eval.md).

Full usage, output layout, credentials and cost tracking are in **[docs/REFERENCE.md](docs/REFERENCE.md)**.

## How it works

```
iPhone LiDAR scan (RoomPlan JSON)          scans/<room>/roomplan.json
        │  arrival/scan        normalize: Apple enums, floor at z=0, align walls
        ▼
Digital twin                               out/twins/<room>/twin.json, layout.yaml
        │  arrival/twin        fixtures → RoboCasa types, synthesize counters
        │  arrival/tasks       tasks × objects × poses × lighting × wordings
        ▼
MuJoCo scene                               arrival/sim/scene.py
        │  drawers (slide joints), doors (hinges), sink basin, free objects
        │  people walking (arrival/sim/people.py), 3 cameras
        ▼
Robot + policy                             arrival/sim/robot.py, ik.py, policy.py
        │  7-DoF arm + gripper, IK, scripted expert │ VLM planner │ VLA client
        ▼
Physics success check → drop failures      arrival/sim/collect.py
        ▼
GR00T LeRobot v2 dataset                   out/datasets/<room>_sim/
        │  arrival/export      parquet + mp4 + meta, validator
        ▼
Fine-tune GR00T N1.7 (next)                docs/REFERENCE.md#later-gr00t-fine-tune-not-run
```

| Folder | What lives there |
|---|---|
| `arrival/scan/` | Scan loaders: RoomPlan JSON, authored ScanBundle, four-photo slot |
| `arrival/twin/`, `arrival/catalog/` | Twin compiler and category maps |
| `arrival/tasks/` | Task and variation proposer |
| `arrival/sim/` | MuJoCo scene, robot, IK, people, policies, data collection |
| `arrival/export/` | LeRobot v2 writer and validator |
| `arrival/generate/`, `arrival/render/` | Scripted placeholder generator, schematic renderer, MimicGen and splat stubs |
| `arrival/models/` | Nebius Token Factory (Nemotron) client |
| `arrival/web/`, `arrival/jobs/` | Local web page and upload queue |
| `scans/` | Sample rooms |
| `tests/` | pytest suite. No network, runs in about 15 s |
| `docs/` | Reference, design contracts, access links, build log |

## Join in: team plan for 30 people

Everything below is open work. **To claim a task:** open a GitHub issue titled with its ID (e.g. `[CAP-3] Scan 5 kitchens`), assign yourself, and post the link in the team channel. One owner per task. If a task is bigger than a day, split it into issues.

### Teams at a glance

| Team | People | Mission | Lead |
|---|---|---|---|
| **CAP**: Capture & LiDAR | 5 | Real iPhone scans of real rooms, and the app to capture them | _TBD_ |
| **TWN**: Twin & scan testing | 4 | Any real scan → a clean, correct digital twin | _TBD_ |
| **SIM**: Scene & realism | 4 | A kitchen that looks real, with every moving part working | _TBD_ |
| **MAN**: Robot & manipulation | 5 | Every task succeeding with a natural-looking arm | _TBD_ |
| **PPL**: People & safety | 2 | Believable moving people the robot has to work around | _TBD_ |
| **POL**: Policy (VLA/VLM) | 4 | Nemotron planner + GR00T policy closing the loop | _TBD_ |
| **DAT**: Data & training | 3 | GR00T fine-tune on our data, and evaluation | _TBD_ |
| **OPS**: Nebius scale & backend | 2 | Uploads processed in the cloud at scale, costs tracked | _TBD_ |
| **DEM**: Demo, web & QA | 1 + everyone | Hosted page, pitch video, release checks | _TBD_ |

Pick leads first. Each lead keeps their team's tasks below up to date.

### CAP: Capture & LiDAR (5)

LiDAR iPhones/iPads: iPhone 12 Pro or newer Pro models, or iPad Pro 2020+.

- **CAP-1: Capture app.** Build the Apple RoomPlan sample ("Create a 3D model of an interior room") with an export button. It writes `CapturedRoom` JSON via `JSONEncoder` and the USDZ. Add AirDrop/share, then upload to `POST /jobs`.
- **CAP-2: RGB-D frames.** Record ARKit RGB + depth frames + camera poses alongside RoomPlan (needed later for realism scoring and splats). Define the folder format in `docs/04-contracts.md`.
- **CAP-3: Scan real rooms.** Scan **at least 10 kitchens** (straight run, L-shape, U-shape, galley, island) plus **5 non-kitchen rooms** (office, living room, bathroom, garage, dorm). For each one:
  - commit `scans/<room_id>/roomplan.json`, a `notes.md` (location type, phone model, iOS version, lighting, clutter level) and 2 reference photos;
  - **measure 3 distances with a tape measure** (counter length, counter height, sink width) and record them in `notes.md`.
  - Remove anything personal from the photos.
- **CAP-4: Scan protocol.** Write `docs/SCANNING.md`: how to walk the room, speed, lighting, what to avoid (mirrors, glass, clutter), how to name rooms, consent for scanning other people's homes.
- **CAP-5: Licenses.** Check the licenses of the two public scans in `tests/fixtures/roomplan/`, and replace them with our own scans if unclear.

### TWN: Twin & scan testing (4)

- **TWN-1: LiDAR accuracy test.** Use the CAP-3 tape measurements to write a new test (planned as tests/test_lidar_accuracy.py, not in the repo yet) that compares twin dimensions with the tape measurements. Report the error per room and fail if it's over 5 cm.
- **TWN-2: Scan regression suite.** Every scan under `scans/` must normalize, compile and render with **no overlapping fixtures, nothing below the floor, and closed walls**. Done: `tests/test_scans.py` runs every `scans/<room>/scan.json` and `roomplan.json` through normalize, compile, `layout.yaml`, a footprint overlap check (a sink or stove inset in its counter is allowed), a floor check, a closed-wall check (1 cm) and a MuJoCo build; the hero `scan.json` overlaps are an expected failure until TWN-6 lands. `python scripts/render_rooms.py` writes a top-down PNG per room to `out/renders/`. New rooms from CAP-3 are picked up with no test changes.
- **TWN-3: RoomPlan coverage.** Handle doors, windows, openings, `polygonCorners` floors, islands, multi-story `story`, and every object category Apple emits (`dishwasher`, `oven`, `washerDryer`, `storage`, `table`, …). Done: doors, windows and openings land in the bundle with their wall and `is_open`, `polygonCorners` floors set the floor extent, a merged multi-story export yields one story per twin, counter-height blocks clear of the walls become `island: true` counters, all 16 Apple categories have a fixture type and a default size, one test per category (`tests/test_roomplan_coverage.py`). Left: non-kitchen categories (bed, sofa, bathtub, …) are a plain `box` in the layout because RoboCasa has no such fixture; doors and windows are in `twin.json` but not in the sim; curved walls and per-wall `polygonCorners` are ignored; no real multi-story export has been seen yet.
- **TWN-4: RoboCasa-native layout.** Make `layout.yaml` load in real RoboCasa (half extents, `wall_side`, groups). This needs a Linux machine with RoboCasa installed.
- **TWN-5: Object detection.** RoomPlan doesn't see mugs or bowls. Detect objects in CAP-2 photos (Nemotron Nano Omni via Token Factory, or any detector) and place them in the twin.
- **TWN-6: Fix the sample kitchen.** Done: fridge, counters and stove now tile the north wall with no overlap, the bowl is on the counter instead of inside the fridge, and `tests/test_roomplan.py` checks footprints (see `docs/BUILD-LOG.md`).

### SIM: Scene & realism (4)

- **SIM-1: Kitchen visuals.** A recessed sink basin with a faucet, cabinet fronts with gaps and slim handles, toe-kick, countertop overhang, backsplash, stove burners, and a proper fridge. Code is in `arrival/sim/scene.py`. **Done in #2** (recessed fronts, slim handles, backsplash, sink with faucet, gas range, fridge with freezer door). Lighting and materials landed with SIM-2.
- **SIM-2: Lighting & materials.** *Done (#21).* Key and fill lights, wood/stone/tile textures, and lighting variation that still matches the `dim/neutral/bright` axis.
- **SIM-3: More moving parts.** Dishwasher door, oven door, microwave, faucet handle, upper cabinets, and drawers in every base cabinet, all on joints with names the policy can target. **Done except the extra drawer row:** oven door, dishwasher and microwave doors, the faucet lever and wall-cabinet doors are named joints with handle sites; `part_names()` in `arrival/sim/scene.py` lists them. A drawer row above the door bay would sit inside the MAN-2 door panel, so it was left out.
- **SIM-4: More objects.** Plates, cups, bottles, pans, fruit, utensils, with realistic mass, friction and size variation. **Done:** cup, plate, bottle, pan, apple and spatula join the mug, bowl and sponge (`arrival/catalog/objects.py`), with mass scaled by volume, per-material friction and seeded size variation; the expert picks the cup, apple and bottle 4/4. The 1 cm spatula is pinched across its width with the pad tips just over the counter (L26) and goes counter to sink and wipes the counter.
- **SIM-5: Non-kitchen rooms.** *Open here; PR #57 needs a rebase.* Twin and sim support for at least one office and one living room from CAP-3 scans.
- **SIM-6: Splat appearance** *(stretch).* Gaussian splat of the scanned room composited with the sim. The design is in `docs/03-architecture.md`.

### MAN: Robot & manipulation (5)

- **MAN-1: Make the arm look and move like a Panda.** Real link shapes and proportions, a Franka-style hand, a low mount, IK biased to the Panda "ready" posture so the elbow doesn't flip, smooth min-jerk motion, and no clipping. Code is in `arrival/sim/robot.py` and `ik.py`. **Done in #2:** Panda-shaped visuals over the unchanged collision capsules, a nullspace pull toward the Panda home posture in IK, and min-jerk motion with via-point drawer pulls.
- **MAN-2: Open doors.** Done (see `docs/BUILD-LOG.md`). Both cabinet doors open at 100%; the fridge door is not a task yet (MAN-5). Check with `python -m arrival sim --families open_door --episodes 4 --save-preview`.
- **MAN-3: Mobile base.** *Open here; PRs #33 and #36 need a rebase, pick one.* Add an actuated base so `counter_to_counter` (about 2 m between pick point and target) is reachable, and add base joints to the state/action spec.
- **MAN-4: Grasp everything.** Bowl done: a hollow bowl and a rim grasp (see `docs/BUILD-LOG.md`). Cups, apples and bottles are picked by the body. Left: plates and pans are wider than the gripper opening.
- **MAN-5: New tasks.** Done: close drawer, close door, put the mug away in a drawer, wipe the counter with the sponge (`python -m arrival sim --families close_drawer,close_door,put_in_drawer,wipe_counter`). Open fridge landed (#78). Left: dishwasher load/unload and the faucet (PR #27 needs a rebase). The joints exist (SIM-3); no task drives them yet.
- **MAN-6: Real Franka model** *(optional; PR #25 needs a rebase).* Swap in the MuJoCo Menagerie Panda (Apache-2.0) behind a flag, keeping tests offline.

### PPL: People & safety (2)

- **PPL-1: Better humans.** Proportioned bodies with walking leg and arm swing, varied clothing colours and heights, and no clipping through counters.
- **PPL-2: Interactions.** People who stop at the counter, open the fridge, and cross the robot's path. Add a safety metric (minimum robot–human distance) to the episode data, and drop episodes with contact. *Done: walkers pause at the counter, cross behind the base and open the fridge (#73); `arrival/sim/safety.py` records the distance, contact episodes are dropped, and the held fridge door is not counted as an arm contact.*

### POL: Policy, VLA & VLM (4)

- **POL-1: VLM planner in the loop.** Done: `python -m arrival sim --planner vlm` runs `VLMPlanner` (Nemotron via Token Factory) inside `collect()`. Without a key it falls back to the rule parser. Responses cache in `arrival/models/nemotron_cache.json`.
- **POL-2: Instruction generation.** Done: `--instructions nemotron --split train|eval` draws varied wordings with a disjoint held-out set (`arrival/tasks/instructions.py`). The template path still keeps the wording and start-offset holdout from POL-4.
- **POL-3: GR00T policy server.** Run GR00T N1.7 inference (H100 or L40S), connect `VLAClient` via `GR00T_POLICY_URL`, and run closed-loop episodes in the sim.
- **POL-4: Closed-loop eval harness.** Score N episodes per task for any policy (success rate, time, collisions) in the scanned room. *Built: `python -m arrival eval` (`arrival/eval/harness.py`), see docs/07-eval.md. Arm-fixture contacts are counted. Still open: run it against a live GR00T server once POL-3 lands.*

### DAT: Data & training (3)

- **DAT-1: GR00T modality config.** Write the config for our 7+1 embodiment, then confirm GR00T's real data loader reads `out/datasets/hero_kitchen_sim`.
- **DAT-2: Fine-tune smoke test.** Run a 100-step GR00T N1.7 fine-tune on the sim dataset, then a full run. The command is in `docs/REFERENCE.md`.
- **DAT-3: Evaluation.** Compare base GR00T, GR00T fine-tuned on generic RoboCasa data, and GR00T fine-tuned on our room. Use 50 episodes per cell on held-out poses and wordings, following the protocol in `docs/07-eval.md`.
- **DAT-4: MimicGen / scale.** Generate thousands of physics demos per room from a few seeds, and track success and throughput.

### OPS: Nebius scale & backend (2)

- **OPS-1: Upload worker.** Done locally: `python -m arrival work` turns a queued scan (RoomPlan export or `scan.json`) into a twin, a sim dataset and `out/jobs/<id>/dataset.zip`, with stage updates in `status.json` and `GET /jobs/<id>/download`. Interrupted jobs are requeued when the worker restarts. Left: one worker process, no automatic retry, no multi-worker locking; running it on Nebius is OPS-2.
- **OPS-2: Nebius fan-out.** Serverless Jobs (one shard of episodes per job) and Object Storage for datasets, with checkpoint and resume.
- **OPS-3: Cost tracking.** Fill the `dollars` fields in the report from real Nebius usage per product.

### DEM: Demo, web & QA (1 + everyone)

- **DEM-1: Hosted page.** Deploy the web app, with a 3D or video viewer for each room and its dataset.
- **DEM-2: Pitch video.** Re-render `scripts/make_demo_video.py` once SIM-1 and MAN-1 land, and add real scan footage from CAP.
- **DEM-3: Release QA.** Before each demo, run a fresh-clone install, the full test suite, `sim` + `validate`, and check every README claim. Update the status table.

### Pull requests

Status of this branch (`integrate`), Oct 19 2026. **Done:** CAP-1, CAP-4, TWN-2, TWN-3, TWN-6, SIM-1, SIM-2, SIM-3, SIM-4, MAN-1, MAN-2, MAN-4 (bowl, cup, apple, bottle, plates and pans), MAN-5 (close drawer/door, put in drawer, wipe, open fridge), PPL-1, PPL-2, POL-1, POL-2, POL-4 (harness; not yet run on a live GR00T), DAT-1 (config; not yet loaded by GR00T), OPS-1 (local). **Partial:** CAP-2 (format open), CAP-5 (licenses unchecked), TWN-4, DEM-1 (local page only), DEM-2, DEM-3, MAN-5 (dishwasher and faucet tasks left). **Open:** CAP-3, TWN-1, TWN-5, SIM-5, SIM-6, MAN-3, MAN-6, POL-3, DAT-2, DAT-3, DAT-4, OPS-2, OPS-3. **Still needed for the hackathon:** GR00T fine-tune, the POL-3 policy server, and real scans (CAP-3).

**Merged into `main`:** #1 MVP handoff · #2 SIM-1, MAN-1 · #3 POL-1, POL-2 · #4 MAN-2 · #5 TWN-6 · #9 DEM-3 PyAV fix · #10 TWN-2, TWN-3 · #11 MAN-5, MAN-4 · #12 OPS-1 · #13 POL-4, PPL-2 · #19 SIM-3, SIM-4 · #21 SIM-2 · #23 PPL-1 · #29 DEM-3 release QA · #31 MAN-4 plates and pans · #35 TAV-1 · #38 DAT-1 · #43 CAP-1 · #44 CAP-4 · #47 MAN-2 collision boxes · #50 DEM-2 · #51 OPS-1 atomic status · #52 validator · #55 POL-4 arm contacts · #71 MAN-1 wrist camera · #73 PPL-2 fridge walker · #76 PPL-1 walker stops · #78 MAN-5 open fridge.

**Open, need a rebase onto `main` (they conflict in `collect.py`, `scene.py` or `people.py`):**

| PR | Task | What it adds |
|---|---|---|
| #77 | SIM | Fixtures on any wall, tasks on scanned cabinets (includes #75 and #49); conflicts with the new walker code |
| #57 | SIM-5 | A scanned office becomes a workable room |
| #33 | MAN-3 | Mobile base behind `--base mobile` |
| #36 | MAN-3 | Competing mobile base; pick one of #33 and #36 |
| #27 | MAN-5 | Oven, dishwasher and faucet tasks |
| #25 | MAN-6 | Menagerie Panda behind `--arm menagerie` |

Closed without merging: #45, #46, #48, #53.

### Everyone: definition of done for a PR

1. **Branch and PR into `main`**, e.g. `feat/MAN-2-door-opening`. Put the task ID in the PR title, keep it small, and get one review from your team lead.
2. **`python -m pytest -q` passes.** If you touched data, also run `sim` or `demo` and then `validate`.
3. **Show it.** For visual changes, attach a before/after image (`--save-preview` writes one).
4. **Follow the schemas** in [docs/04-contracts.md](docs/04-contracts.md). Don't invent parallel formats.
5. **Keep labels honest.** `arrival_physically_valid`, `arrival_generator` and `joints_are_placeholder` must describe what really happened. Never call scripted data MimicGen, trained, or real-robot.
6. **Never commit secrets, `out/`, or personal data from scans.** Put API keys in `.env`, which is gitignored. See [docs/08-access.md](docs/08-access.md) for accounts.

## Docs

- [docs/REFERENCE.md](docs/REFERENCE.md): every command, output layout, credentials, cost table, fine-tune command
- [docs/08-access.md](docs/08-access.md): Nebius, Token Factory, GR00T and RoboCasa accounts and links
- [docs/04-contracts.md](docs/04-contracts.md): data schemas
- [docs/09-later.md](docs/09-later.md): designs for MimicGen, splat, Cosmos and fine-tune
- [docs/BUILD-LOG.md](docs/BUILD-LOG.md): what changed and why
- [docs/00-START-HERE.md](docs/00-START-HERE.md): the original MVP build contract

## License

Apache 2.0. See [LICENSE](LICENSE). Copyright 2026 Exea Labs.
