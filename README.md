# Arrival

**Scan a room once. Get robot training data for that exact room.**

[![Arrival demo: scan → digital twin → MuJoCo simulation → GR00T dataset](docs/media/demo.gif)](docs/media/demo.mp4)

*30-second demo, rendered from the real code (an early cut: the room and arm visuals are being fixed under SIM-1 and MAN-1). Click for the full-quality mp4. Regenerate it with `python scripts/make_demo_video.py`.*

---

## What this is

General robot policies like NVIDIA GR00T are trained on generic kitchens. When a robot arrives in *your* kitchen, it has never seen your counters, your sink or your drawers. Arrival closes that gap:

1. **Scan.** An iPhone LiDAR scan (Apple RoomPlan) describes the room: walls, counters, sink, stove, fridge.
2. **Twin.** We compile the scan into a digital twin: a RoboCasa-style layout and a MuJoCo scene with working drawers and doors.
3. **Simulate.** A 7-joint Panda-style arm does tasks in the twin while people walk around. Grasps use friction physics, and success is checked by the simulator.
4. **Capture.** Every successful episode is written as a [GR00T](https://github.com/NVIDIA/Isaac-GR00T)-compatible LeRobot v2 dataset: joint states, actions, ego and wrist camera video, and language instructions.
5. **Fine-tune** *(next)*. Train GR00T N1.7 on that room's data so the robot is useful on day one.

Built for the NVIDIA × Nebius hackathon by Exea Labs and a team of 30. **Want to help? Jump to [Join in](#join-in-team-plan-for-30-people).** The product spec is [plan.pdf](plan.pdf).

## Does it work? (status, Oct 14 2026)

| Stage | Status |
|---|---|
| Scan → twin | ✅ Works on the sample kitchen and on 2 real public iPhone RoomPlan exports. ⚠️ No iPhone capture app yet |
| Simulation: kitchen, drawers, doors, oven, dishwasher, microwave, wall cabinets, faucet, people | ✅ Works. MuJoCo, built from the twin; every moving part is a named joint with a handle site |
| People: interactions + safety metric | ✅ Walkers stop at the counter and cross behind the robot. Every episode records the minimum robot-human distance; episodes with contact are dropped. ⚠️ Nobody opens the fridge yet |
| Closed-loop eval harness | ✅ `python -m arrival eval` scores N episodes per family (success, time, human contacts, Wilson 95%). ⚠️ Only the scripted expert has been scored; the VLA cell needs a GR00T server (POL-3) |
| Robot: pick mug, cup, apple, bottle, bowl, plate, pan → sink | ✅ 100% success, friction grasp (bowl and plate by the rim, pan by the handle), physics-checked |
| Robot: open drawer | ✅ 100% success |
| Robot: open door | ✅ 100% on both cabinet doors (8/8, two seeds, with and without people). Opens to about 70°: the arm is mounted at 0.75 m, so the far end of the handle arc sits at its reach limit. A mobile base (MAN-3) is what lifts that |
| Robot: counter → counter | ❌ Skipped. The pick point and the target counter are about 2 m apart and the base doesn't move |
| Robot: close drawer, close door, put mug in drawer, wipe counter | ✅ 100% each (32/32 across all families, seed 5, two people walking). The part starts open for the close tasks; the sponge must travel 0.4 m pressed to the top |
| Robot: bowl | ✅ Held by the rim (one finger inside the wall). 4/4 bowl → sink. The bowl is now hollow in physics |
| GR00T-format dataset + validator | ✅ Passes our validator, and the layout matches GR00T's loader source. ⚠️ Not yet loaded by GR00T itself since the latest fixes |
| Local web page: browse, download, upload | ✅ Works. Uploads are processed by `python -m arrival work` on the same machine (twin, sim dataset, zip). ⚠️ One local worker, no cloud fan-out yet |
| VLM planner (Nemotron) / VLA client (GR00T server) | ✅ VLM planner (Nemotron) wired into sim collection with Token Factory and offline cache fixture (POL-1); dynamic instruction generation with held-out eval set (POL-2). ⚠️ VLA client tested offline |
| GR00T fine-tune, MimicGen, splats, Cosmos, Nebius scale-out | ❌ Not built yet |

There are two datasets. `python -m arrival sim` writes **physics-simulated** data with real joint states (`arrival_physically_valid: true`). `python -m arrival demo` writes a fast **scripted placeholder** with zero joint values (`arrival_physically_valid: false`). Never train on the placeholder.

## Get started (5 minutes)

Requires Python 3.11+ on macOS, Linux or Windows (the suite and the MuJoCo sim run as is on Windows; use `.venv\Scriptsctivate` there). No GPU and no API keys needed.

```bash
git clone https://github.com/AvnehSBhatia/ExeaLabsNvidiaHacks.git
cd ExeaLabsNvidiaHacks
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,sim]"
python -m pytest -q                       # should be all green
```

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

- **TWN-1: LiDAR accuracy test.** Use the CAP-3 tape measurements to write `tests/test_lidar_accuracy.py`, which compares twin dimensions with the tape measurements. Report the error per room and fail if it's over 5 cm.
- **TWN-2: Scan regression suite.** Every scan under `scans/` must normalize, compile and render with **no overlapping fixtures, nothing below the floor, and closed walls**. Done: `tests/test_scans.py` runs every `scans/<room>/scan.json` and `roomplan.json` through normalize, compile, `layout.yaml`, a footprint overlap check (a sink or stove inset in its counter is allowed), a floor check, a closed-wall check (1 cm) and a MuJoCo build; the hero `scan.json` overlaps are an expected failure until TWN-6 lands. `python scripts/render_rooms.py` writes a top-down PNG per room to `out/renders/`. New rooms from CAP-3 are picked up with no test changes.
- **TWN-3: RoomPlan coverage.** Handle doors, windows, openings, `polygonCorners` floors, islands, multi-story `story`, and every object category Apple emits (`dishwasher`, `oven`, `washerDryer`, `storage`, `table`, …). Done: doors, windows and openings land in the bundle with their wall and `is_open`, `polygonCorners` floors set the floor extent, a merged multi-story export yields one story per twin, counter-height blocks clear of the walls become `island: true` counters, all 16 Apple categories have a fixture type and a default size, one test per category (`tests/test_roomplan_coverage.py`). Left: non-kitchen categories (bed, sofa, bathtub, …) are a plain `box` in the layout because RoboCasa has no such fixture; doors and windows are in `twin.json` but not in the sim; curved walls and per-wall `polygonCorners` are ignored; no real multi-story export has been seen yet.
- **TWN-4: RoboCasa-native layout.** Make `layout.yaml` load in real RoboCasa (half extents, `wall_side`, groups). This needs a Linux machine with RoboCasa installed.
- **TWN-5: Object detection.** RoomPlan doesn't see mugs or bowls. Detect objects in CAP-2 photos (Nemotron Nano Omni via Token Factory, or any detector) and place them in the twin.
- **TWN-6: Fix the sample kitchen.** Done: fridge, counters and stove now tile the north wall with no overlap, the bowl is on the counter instead of inside the fridge, and `tests/test_roomplan.py` checks footprints (see `docs/BUILD-LOG.md`).

### SIM: Scene & realism (4)

- **SIM-1: Kitchen visuals.** A recessed sink basin with a faucet, cabinet fronts with gaps and slim handles, toe-kick, countertop overhang, backsplash, stove burners, and a proper fridge. Code is in `arrival/sim/scene.py`. **The room currently looks buggy. This is a top priority.**
- **SIM-2: Lighting & materials.** Key and fill lights, wood/stone/tile textures, and lighting variation that still matches the `dim/neutral/bright` axis.
- **SIM-3: More moving parts.** Dishwasher door, oven door, microwave, faucet handle, upper cabinets, and drawers in every base cabinet, all on joints with names the policy can target. **Done except the extra drawer row:** oven door, dishwasher and microwave doors, the faucet lever and wall-cabinet doors are named joints with handle sites; `part_names()` in `arrival/sim/scene.py` lists them. A drawer row above the door bay would sit inside the MAN-2 door panel, so it was left out.
- **SIM-4: More objects.** Plates, cups, bottles, pans, fruit, utensils, with realistic mass, friction and size variation. **Done:** cup, plate, bottle, pan, apple and spatula join the mug, bowl and sponge (`arrival/catalog/objects.py`), with mass scaled by volume, per-material friction and seeded size variation; the expert picks the cup, apple and bottle 4/4. The 1 cm spatula sits under the grasp height, so it is scenery until the grasp rule changes.
- **SIM-5: Non-kitchen rooms.** Twin and sim support for at least one office and one living room from CAP-3 scans.
- **SIM-6: Splat appearance** *(stretch).* Gaussian splat of the scanned room composited with the sim. The design is in `docs/03-architecture.md`.

### MAN: Robot & manipulation (5)

- **MAN-1: Make the arm look and move like a Panda.** Real link shapes and proportions, a Franka-style hand, a low mount, IK biased to the Panda "ready" posture so the elbow doesn't flip, smooth min-jerk motion, and no clipping. Code is in `arrival/sim/robot.py` and `ik.py`. **The arm currently looks buggy. This is a top priority.**
- **MAN-2: Open doors.** Done (see `docs/BUILD-LOG.md`). Both cabinet doors open at 100%; the fridge door is not a task yet (MAN-5). Check with `python -m arrival sim --families open_door --episodes 4 --save-preview`.
- **MAN-3: Mobile base.** Add an actuated base so `counter_to_counter` (about 2 m between pick point and target) is reachable, and add base joints to the state/action spec.
- **MAN-4: Grasp everything.** Done: the bowl and the plate are hollow and pinched by the rim nearest the robot, the pan is pinched across its handle at the root, cups, apples and bottles are held by the body (see `docs/BUILD-LOG.md`). What limits it is the clamp: the position-controlled fingers hold about 5 to 8 N, so the pan hangs from its handle and the plate rides tilted while carried; both still land in the sink 4/4. The spatula is the one catalog object not picked.
- **MAN-5: New tasks.** Done: close drawer, close door, put the mug away in a drawer, wipe the counter with the sponge (`python -m arrival sim --families close_drawer,close_door,put_in_drawer,wipe_counter`). Left: dishwasher load/unload and the faucet. The joints exist (SIM-3); no task drives them yet.
- **MAN-6: Real Franka model** *(optional).* Swap in the MuJoCo Menagerie Panda (Apache-2.0) behind a flag, keeping tests offline.

### PPL: People & safety (2)

- **PPL-1: Better humans.** Proportioned bodies with walking leg and arm swing, varied clothing colours and heights, and no clipping through counters.
- **PPL-2: Interactions.** People who stop at the counter, open the fridge, and cross the robot's path. Add a safety metric (minimum robot–human distance) to the episode data, and drop episodes with contact. *Done except the fridge: walkers pause at the counter and cross behind the base, `arrival/sim/safety.py` records the distance, contact episodes are dropped. Opening the fridge needs the door hinge driven by a person, which is still open.*

### POL: Policy, VLA & VLM (4)

- **POL-1: VLM planner in the loop.** Done: `python -m arrival sim --planner vlm` runs `VLMPlanner` (Nemotron via Token Factory) inside `collect()`. Without a key it falls back to the rule parser. Responses cache in `arrival/models/nemotron_cache.json`.
- **POL-2: Instruction generation.** Done: `--instructions nemotron --split train|eval` draws varied wordings with a disjoint held-out set (`arrival/tasks/instructions.py`). The template path still keeps the wording and start-offset holdout from POL-4.
- **POL-3: GR00T policy server.** Run GR00T N1.7 inference (H100 or L40S), connect `VLAClient` via `GR00T_POLICY_URL`, and run closed-loop episodes in the sim.
- **POL-4: Closed-loop eval harness.** Score N episodes per task for any policy (success rate, time, collisions) in the scanned room. *Built: `python -m arrival eval` (`arrival/eval/harness.py`), see docs/07-eval.md. Still open: run it against a live GR00T server once POL-3 lands, and count arm-fixture collisions, not only human contact.*

### DAT: Data & training (3)

- **DAT-1: GR00T modality config.** Write the config for our 7+1 embodiment, then confirm GR00T's real data loader reads `out/datasets/hero_kitchen_sim`.
- **DAT-2: Fine-tune smoke test.** Run a 100-step GR00T N1.7 fine-tune on the sim dataset, then a full run. The command is in `docs/REFERENCE.md`.
- **DAT-3: Evaluation.** Compare base GR00T, GR00T fine-tuned on generic RoboCasa data, and GR00T fine-tuned on our room. Use 50 episodes per cell on held-out poses and wordings, following the protocol in `docs/07-eval.md`.
- **DAT-4: MimicGen / scale.** Generate thousands of physics demos per room from a few seeds, and track success and throughput.

### OPS: Nebius scale & backend (2)

- **OPS-1: Upload worker.** Done locally: `python -m arrival work` turns a queued scan (RoomPlan export or `scan.json`) into a twin, a sim dataset and `out/jobs/<id>/dataset.zip`, with stage updates in `status.json` and `GET /jobs/<id>/download`. Left: one worker process, no retries, no multi-worker locking; running it on Nebius is OPS-2.
- **OPS-2: Nebius fan-out.** Serverless Jobs (one shard of episodes per job) and Object Storage for datasets, with checkpoint and resume.
- **OPS-3: Cost tracking.** Fill the `dollars` fields in the report from real Nebius usage per product.

### DEM: Demo, web & QA (1 + everyone)

- **DEM-1: Hosted page.** Deploy the web app, with a 3D or video viewer for each room and its dataset.
- **DEM-2: Pitch video.** Re-render `scripts/make_demo_video.py` once SIM-1 and MAN-1 land, and add real scan footage from CAP.
- **DEM-3: Release QA.** Before each demo, run a fresh-clone install, the full test suite, `sim` + `validate`, and check every README claim. Update the status table.

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
