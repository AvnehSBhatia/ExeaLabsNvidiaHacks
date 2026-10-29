# Arrival

**Scan a room once. Get robot training data for that exact room.**

[![Arrival demo: scan → digital twin → MuJoCo simulation → GR00T dataset](docs/media/demo.gif)](docs/media/demo.mp4)

*A 30-second silent render from the code (`python scripts/make_demo_video.py`).*

**Open source under the [Apache License 2.0](LICENSE)** · Live site: https://arrival.exealabs.org · Pitch video: [YouTube, 2:41](https://youtu.be/_DsfMNQbM2w) · Fine-tuned model: [GR00T N1.7 on Hugging Face](https://huggingface.co/Ish1105/arrival-gr00t-n17-run4) · Track: **Physical AI**

**For judges:** [Results at a glance](#at-a-glance) · [Setup in 5 minutes](#get-started-5-minutes) · [NVIDIA Nemotron and NVIDIA open models](#nvidia-and-nebius-in-arrival) · [Where Token Factory accelerated our workflow](#where-nebius-token-factory-accelerated-our-workflow) · [Other Nebius tools and services](#other-nebius-tools-and-services) · [Results and limits](#does-it-work-status-oct-27-2026)

---

## At a glance

- **A real room in, training data out.** An iPhone RoomPlan scan of a real kitchen becomes a MuJoCo twin in which every drawer and door is a working joint. In that twin the arm completes 12 of 12 tasks across 11 task families, and simulated people walk around it; only demos that pass the physics check are kept.
- **GR00T learns the room.** NVIDIA GR00T N1.7-3B fine-tuned on one room's data: 39 of 48 trials against 26 of 48 for the clean-data baseline, on identical trials (paired sign test p = 0.007; small samples, [docs/07-eval.md](docs/07-eval.md)).
- **Nemotron plans every task, live on Nebius Token Factory.** On indirect requests ("let me check if we're out of milk") Nemotron 3 Super picks the right task 13 times in 15; the keyword parser it replaced, 6 ([benchmark](docs/results/planner-benchmark-2026-10-26.json)).
- **Hours become minutes.** 1,000 teleoperated demos would take a person 17.1 hours (61.5 s each, [measured](docs/results/p3-teleop-2026-10-21/README.md)); Arrival makes 1,000 physics-checked demos in about 55 minutes on one laptop (3.3 s each).
- **Sim to real look, geometry kept.** NVIDIA Cosmos Transfer2.5-2B restyled sim episodes into photoreal kitchens; all 279 frames kept the physics' geometry ([results](docs/results/cosmos-transfer-2026-10-27/README.md)).
- **Built in the open.** 418 merged pull requests from 16 contributors, about 2,280 tests, CI on Linux, macOS and Windows.

## What this is

General robot policies like NVIDIA GR00T are trained on generic kitchens. When a robot arrives in *your* kitchen, it has never seen your counters, your sink or your drawers. Arrival closes that gap:

1. **Scan.** An iPhone LiDAR scan (Apple RoomPlan) describes the room: walls, counters, sink, stove, fridge.
2. **Twin.** We compile the scan into a digital twin: a RoboCasa-style layout and a MuJoCo scene with working drawers and doors.
3. **Simulate.** A 7-joint Panda-style arm does tasks in the twin while people walk around. Grasps use friction physics, and success is checked by the simulator.
4. **Capture.** Every successful episode is written as a [GR00T](https://github.com/NVIDIA/Isaac-GR00T)-compatible LeRobot v2 dataset: joint states, actions, ego and wrist camera video, and language instructions.
5. **Fine-tune.** GR00T N1.7 ([weights](https://huggingface.co/Ish1105/arrival-gr00t-n17-run4), NVIDIA Open Model License) has been trained on that room's data (39 of 48 against 26 of 48; small samples, and the base-GR00T and generic-RoboCasa cells are not run).

Built in the open by Exea Labs for the Nebius × NVIDIA Global AI Hackathon, Physical AI track: 53 people on the team, 16 of them with merged code ([who built this](#who-built-this)). There is no physical robot: everything the robot does happens in the simulated twin of a real, scanned room. The product spec is [plan.pdf](plan.pdf).

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
open out/previews/                        # contact sheet + overview video (Windows: start out\previews, Linux: xdg-open)
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
python -m arrival cost                                         # time and cost report vs P3 teleoperation baseline
```

Each evaluation writes a new `out/eval/<run_id>/` with `manifest.json`, `episodes.jsonl` and `report.json`. Replay the printed manifest with `--manifest PATH --policy vla --checkpoint-id LABEL` to compare checkpoints on the same twin and ordered trials. The checkpoint label is supplied by the caller; it is not verified against the server. See [the evaluation protocol](docs/07-eval.md).

### With the live NVIDIA and Nebius APIs (optional)

Nothing above needs a key. To run the real models, copy `.env.example` to `.env` (it is git-ignored) and set:

```bash
TOKEN_FACTORY_API_KEY=...                                  # Nebius Token Factory (tokenfactory.nebius.com)
NEMOTRON_SUPER_MODEL=nvidia/nemotron-3-super-120b-a12b     # plans each task
NEMOTRON_LIGHTNING_MODEL=nvidia/Nemotron-3_5-Lightning     # writes training and held-out instruction wordings
NEMOTRON_OMNI_MODEL=nvidia/Nemotron-3.5-Super-VL           # finds small objects in scan frames
TAVILY_API_KEY=...                                         # appliance research passed to the planner
```

Then:

```bash
python -m arrival doctor                  # which keys are set, without printing them
python scripts/smoke_live_apis.py         # one real call per service, saved under docs/results/
python -m arrival sim --planner vlm --instructions nemotron --tavily \
    --families counter_to_sink,open_drawer,open_fridge --episodes 3 --people 2
```

The sim prints `planned by token_factory` for each task and `[tavily] ... tavily_api` for each appliance it researched.

Full usage, output layout, credentials and cost tracking are in **[docs/REFERENCE.md](docs/REFERENCE.md)**.

## NVIDIA and Nebius in Arrival

| What | Where in the code | What it does here | Run for real? |
|---|---|---|---|
| **GR00T N1.7-3B** (NVIDIA open model) | `configs/gr00t/`, `scripts/prepare_gr00t.py`, `arrival/sim/policy.py` (`VLAClient`), `arrival/policy_server/` | Fine-tuned on a room's sim dataset, then scored closed-loop in the same twin through a policy server | Yes: 39 of 48 against 26 of 48 ([weights](https://huggingface.co/Ish1105/arrival-gr00t-n17-run4), `docs/07-eval.md`). Trained on one 96 GB GPU on Cosmos, not on Nebius |
| **Nemotron on Nebius Token Factory** | `arrival/models/token_factory.py`, `VLMPlanner` in `arrival/sim/policy.py`, `arrival/tasks/instructions.py`, `arrival/twin/detect.py` | Plans each task from the instruction and the room inventory (`sim --planner vlm`), writes varied train and held-out instruction wordings (`--instructions nemotron`), and finds small objects in scan frames | Yes (Oct 26): `nvidia/nemotron-3-super-120b-a12b` plans and `nvidia/Nemotron-3_5-Lightning` writes wordings, about 2.6 s a plan, and `nvidia/Nemotron-3.5-Super-VL` found the mug in rendered scan frames and placed it 1.2 cm from its true centre (`docs/results/live-detect-2026-10-26.json`). On 15 indirect requests ("let me check if we're out of milk") the Nemotron planner picks the right task 13 times, the keyword parser 6; on 54 Lightning-written wordings, 52 against 47 (`scripts/planner_benchmark.py`, `docs/results/planner-benchmark-2026-10-26.json`) |
| **Nebius AI Cloud: Serverless Jobs and Object Storage** | `arrival/jobs/nebius.py`, `arrival fanout --backend nebius` | Splits collection into resumable shards, one Serverless Job each, with results in a bucket, then merges them | Not yet (C1). Tested against in-process fakes of both APIs |
| **Cosmos Transfer2.5-2B** (NVIDIA open model) | `arrival/render/variation.py` (geometry check), `docs/results/cosmos-transfer-2026-10-27/` | Restyles a rendered sim episode into a photoreal kitchen, guided by the episode's own edges, so the arm, the fixtures and the people stay where the physics put them | Yes (Oct 27), on one RTX PRO 6000 we rented: three 93-frame episodes (open_fridge, counter_to_sink, open_drawer), about 8 minutes each. All 279 frames passed our edge check (correlation 0.74 to 0.83 against a 0.6 threshold) and NVIDIA's guardrail. Not yet run over a whole dataset |
| **Tavily** | `arrival/models/tavily.py`, `arrival/tasks/tavily_research.py` | Looks up the scanned appliances (hinges, clearances, safety) and passes the findings to the Nemotron planner (`sim --planner vlm --tavily`) | Yes (Oct 26): e.g. the fridge query returns the federal limit of 66.7 N (15 lb) to open an empty door, citing ecfr.gov and GE (`docs/results/live-apis-2026-10-26.json`). The safety guidelines next to it are still our built-in table, labelled `built_in_table` |

### Where Nebius Token Factory accelerated our workflow

Token Factory is the Nebius service Arrival runs on, live: one OpenAI-compatible endpoint and one key serve all three Nemotron models, so the client is small (`arrival/models/token_factory.py`) and changing a model is an environment variable.

- **A model plans every task, so we stopped extending keyword rules.** `nvidia/nemotron-3-super-120b-a12b` turns a request and the room's inventory into a task plan in about 2.6 s (median). On 15 indirect requests ("let me check if we're out of milk") it picked the right task 13 times; the keyword parser it replaced got 6.
- **The training language writes itself.** `nvidia/Nemotron-3_5-Lightning` writes varied training wordings and a held-out evaluation set for each task family, where the offline path uses fixed templates. The planner got 52 of 54 of those wordings right, the keyword parser 47.
- **Scan frames are read without hand labels.** `nvidia/Nemotron-3.5-Super-VL` finds the small objects that RoomPlan does not report: on rendered frames of the twin it found the mug in every frame that has one and nothing in the empty ones, and the mug was placed 1.2 cm from its true centre.
- **Live research reaches the planner.** Tavily's findings for each scanned appliance go into the Nemotron prompt (`sim --planner vlm --tavily`), so the fridge plan knows the door opens with at most 66.7 N.
- **Fast to iterate.** No rate limit was hit in more than 150 calls in about ten minutes; answers are cached under `out/cache/`, so re-runs cost nothing; and with no key every path falls back offline and says so, so the 2,280-test suite needs no key.

The evidence is committed: `docs/results/live-apis-2026-10-26.json`, `docs/results/planner-benchmark-2026-10-26.json` and `docs/results/live-detect-2026-10-26.json`. Re-run them with `python scripts/smoke_live_apis.py` and `python scripts/planner_benchmark.py`.

### Other Nebius tools and services

- **Nebius AI Cloud Serverless Jobs and Object Storage.** `arrival fanout --backend nebius` (`arrival/jobs/nebius.py`) splits data collection into resumable shards, runs each one as a Serverless Job, keeps the results in an Object Storage bucket and merges them. It is built and tested against in-process fakes of both APIs, and it has not run on Nebius yet, so we don't count it as used.
- The GR00T N1.7 fine-tune ran on one 96 GB GPU on Cosmos, not on Nebius.

Keys go in an untracked `.env` (see `.env.example`); `python -m arrival doctor` reports which are set without printing them. With no keys every path falls back to an offline one, and its output says which source answered. Answers are cached under `out/cache/`, never in the package, so a configured key always reaches the API on a fresh checkout. `python scripts/smoke_live_apis.py` checks all three keys with one real call each.

## Does it work? (status, Oct 27 2026)

| Stage | Status |
|---|---|
| Scan → twin | ✅ Works on the sample kitchen (now tiled, no overlapping fixtures) on one real kitchen scanned with the capture app (`ios/`, CAP-1) on an iPhone 13 Pro Max, and on 2 authored RoomPlan fixtures (office, study; task P5); a scan regression suite covers every room under `scans/`. Field scanning protocol in `docs/SCANNING.md` |
| Simulation: kitchen, drawers, doors, oven, fridge, dishwasher, microwave, wall cabinets, faucet, people | ✅ Works. MuJoCo, built from the twin; every moving part is a named joint with a handle site. Panda-style arm and kitchen visuals (recessed fronts, slim handles, sink, gas range) landed with SIM-1 and MAN-1 |
| People: interactions + safety metric | ✅ Walkers stop at the counter and cross behind the robot. Every episode records the minimum robot-human distance; episodes with contact are dropped. A walker opens the fridge, stands in the doorway and closes it, and the held door counts for the safety metric |
| Closed-loop eval harness | ✅ `python -m arrival eval` scores N episodes per family (success, time, human contacts, Wilson 95%). Counts arm-fixture contacts too; held-out seeds are offset from training and an episode that raises counts as a failure. ⚠️ The scripted expert is scored; GR00T checkpoints were scored through a live server: 100 steps got 0 of 8 (a plumbing check); a 6,000-step fine-tune on 320 episodes got 10 of 24 (42%, Wilson 95% 25 to 61%) on `counter_to_sink` and `open_drawer`, no people, so far from the scripted expert (docs/ref/POL-3.md) |
| Robot: pick mug → sink | ✅ 100% success, friction grasp, physics-checked |
| Robot: open drawer | ✅ 100% success |
| Robot: open door | ✅ The arm opens both cabinet doors (`eval --families open_door --episodes 8 --people 2`, seeds 0 to 3: 8/8 each, no walker contact; before the door keep-out one of those four seeds lost 1 of 8). Opens to about 63-66°: the hand holds its heading while the bar turns between the pads, and joint 7 stays inside its limit. The arm is mounted at 0.75 m, so the mobile base (L13, `--base mobile`) is what would open it wider |
| Robot: counter → counter | ✅ With `--base mobile` (L13): the pedestal drives along the counter front at 0.3 m/s and sets the object down 2 m away, 6/6 with two people walking in the hero kitchen. The default base is still fixed, and skips the family |
| Robot: close drawer, close door, put mug in drawer, wipe counter | ✅ 100% each (32/32 across all families, seed 5, two people walking). The part starts open for the close tasks; the sponge must travel 0.4 m pressed to the top |
| Robot: bowl | ✅ Held by the rim (one finger inside the wall). 4/4 bowl → sink. The bowl is now hollow in physics |
| GR00T-format dataset + validator | ✅ Passes our validator, and loads in Isaac-GR00T's own `LeRobotEpisodeLoader` (`dac54b3`, Oct 27, on main `03690dd`): the fixed-arm, mobile-base and three-camera datasets at 224 px, every episode, decoded video, finite state and action, an instruction per row; GR00T's `stats.py` writes their stats (`scripts/check_gr00t_loader.py`, [docs/results/gr00t-loader-2026-10-27](docs/results/gr00t-loader-2026-10-27/README.md)). No fine-tune has run on a dataset this new |
| Local web page: browse, download, upload | ✅ Works. Redesigned page: room cards with previews and stats, drag-and-drop upload, live job stages. Uploads are processed by `python -m arrival work` on the same machine (twin, sim dataset, zip); a killed worker's jobs are requeued. The static demo page is live at https://arrival.exealabs.org and was checked in a browser at 1024 px and 375 px (#195). ⚠️ The upload-to-dataset page is not hosted, and the Nebius fan-out has not run for real |
| VLM planner (Nemotron) / VLA client (GR00T server) | ✅ VLM planner (Nemotron) wired into sim collection with Token Factory and offline cache fixture (POL-1); dynamic instruction generation with held-out eval set (POL-2). The VLA client (`--policy vla`) has scored GR00T checkpoints through a live GR00T server (`docs/ref/POL-3.md`; scores in the eval row above) |
| GR00T N1.7 fine-tune and paired evaluation | ✅ Fine-tunes of GR00T N1.7-3B on `hero_kitchen_sim` ran on one 96 GB GPU on Cosmos (not Nebius) and were scored in the sim on identical trials (`docs/07-eval.md`, "Results so far"). Best so far 39 of 48 trials (81%) against 26 of 48 for the clean-data baseline (paired sign test p = 0.007): clean data plus DART demos with the marginal drawer pulls removed. ⚠️ Small samples; the generic-RoboCasa and base-GR00T cells of the usefulness protocol (DAT-3) are not run, so `usefulness` stays `null` |
| Demo multiplication, splats, Cosmos, Nebius scale-out | 🟡 Partly built, none run at scale. Demos are multiplied in physics by our own retargeting, not upstream MimicGen: seed-demo replay (DAT-4a, `arrival/generate/retarget.py`) and object-centric SE(3) retargeting of `counter_to_sink` (G7, `arrival/generate/se3_retarget.py`), which turned 14 human teleop seeds into 104 physics-checked episodes from 140 attempts; `arrival/generate/mimicgen.py` only probes for RoboCasa. `arrival/render/splat.py` composites a static view and the moving geometry by depth; no Gaussian splat is trained. NVIDIA Cosmos Transfer2.5-2B restyled three sim episodes into photoreal kitchens on Oct 27, and every frame passed the restyle stage's edge check (`docs/results/cosmos-transfer-2026-10-27/`); the dataset run through `arrival/render/variation.py` (SIM-7a) is still open. `arrival fanout --backend nebius` is tested against in-process fakes and has not run on Nebius. Still open for the hackathon: the DAT-3 baseline cells and real scans (CAP-3) |

There are two datasets. `python -m arrival sim` writes **physics-simulated** data with real joint states (`arrival_physically_valid: true`). `python -m arrival demo` writes a fast **scripted placeholder** with zero joint values (`arrival_physically_valid: false`). Never train on the placeholder.

## Known limits

- GR00T trains on our datasets (three fine-tunes, `docs/07-eval.md`), but every score is from 24 to 48 paired trials and the same checkpoint has scored 2 of 12 and 7 of 12 on the same 12 trials: only large paired differences mean anything.
- The mobile base (L13) is behind `--base mobile` and the real Franka behind `--arm menagerie`; the defaults are still the fixed Panda-style arm. The Menagerie arm was measured over all families at seed 5 with no people and, on Oct 27 (main `897aa05`), at seeds 0 and 3 with two people walking: 12 of 14 and 13 of 14, `open_oven` missing in both. Its steepest flap tilt put the real hand's body into the stove front; without that tilt, the same runs on main `5898e89` are 14 of 14 and 14 of 14, no human contacts, no episode flagged.
- With a fixed base, drawers and `close_door` on the U-kitchen's corner cabinet stay unreachable and are skipped. The rotated kitchen's fridge, whose right side is in the room's corner, is hinged on that side since the hinge is inferred from the walls (`hinge_source: inferred_from_walls`), and opens. Only the corner cabinet's `open_door` takes a shifted stand spot. The reach planner checks the arm's IK, not collisions.
- The job memory limit (`ARRIVAL_MAX_JOB_MB`) counts resident memory across the process tree (`/proc` on Linux, `Toolhelp32`/`psapi` on Windows, `ps` on macOS; `docs/ref/MEMCAP.md`).
- The static demo page is live at https://arrival.exealabs.org and was checked in a browser at desktop and phone width (#195). The upload page's buttons were checked in headless Chromium; it is not hosted. Uploads are processed by local workers that share one `out/` folder.
- One real kitchen, scanned with the capture app on an iPhone 13 Pro Max, has run end to end (`tests/fixtures/roomplan/kitchen_real_01.json`: 12 tasks over 11 families, 9 of 12 episodes, valid). With every family at 12 episodes (seed 0, no people) it was 6 of 12 on Oct 26, every cabinet door failing, and is 12 of 12 since the blind-corner and narrow-cabinet layout fix (`docs/BUILD-LOG.md`). It has no tape measurements yet, and GR00T has not trained on it; the GR00T scores above are from the authored hero kitchen. The three measured, consented scans (P2) are not done.
- `tests/fixtures/roomplan/office.json` and `study.json` are clean authored test fixtures under Apache 2.0 (P5, #310). Real phone captures with doors and windows remain open under P5 to validate placement against real-device sensor noise.

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
Fine-tune GR00T N1.7, score in the twin   docs/GR00T-FINETUNE.md, docs/07-eval.md
```

| Folder | What lives there |
|---|---|
| `arrival/scan/` | Scan loaders: RoomPlan JSON, authored ScanBundle, four-photo slot |
| `arrival/twin/`, `arrival/catalog/` | Twin compiler and category maps |
| `arrival/tasks/` | Task and variation proposer |
| `arrival/sim/` | MuJoCo scene, robot, IK, people, policies, data collection |
| `arrival/export/` | LeRobot v2 writer and validator |
| `arrival/generate/`, `arrival/render/` | Scripted placeholder, physics seed replay and native SE(3) retargeting; external MimicGen and splat stubs |
| `arrival/models/` | Nebius Token Factory (Nemotron) client |
| `arrival/web/`, `arrival/jobs/` | Local web page and upload queue |
| `scans/` | Sample rooms |
| `tests/` | pytest suite. No network, about 20 minutes on one core (CI splits it into three shards, `-n auto`) |
| `docs/` | Reference, design contracts, access links, build log |

## Paired robot and room views

Export synchronized ego, wrist and external room views from the **same** MuJoCo demonstration with
`--exocentric`. A shared frame index and timestamp align each camera with the recorded joint state and action.
This extends the existing collection pipeline; the default two-camera workflow is unchanged.

```bash
python -m arrival sim --families open_drawer --episodes 2 --people 1 --res 128 --seed 3 --exocentric --save-preview --out out/multiview
python -m arrival validate out/multiview/datasets/hero_kitchen_sim
```

![Synchronized robot ego and external room views from one MuJoCo rollout](docs/media/ego-exo.png)

The paired preview adds labels only for viewing; exported training videos stay unannotated. An optional GR00T
modality config requests all three cameras. This is scripted simulation data; no three-view model training or
performance gain is claimed. See [capture, output layout and limitations](docs/REFERENCE.md#synchronized-ego-and-exocentric-capture).

<!-- contributors:start (generated by scripts/render_contributors.py) -->

## Who built this

Pull requests opened against this repository, by author (merged, or consolidated into the rebased PRs). The full list, with commit counts, is in [docs/CONTRIBUTORS.md](docs/CONTRIBUTORS.md); each to-do row below names who worked on it.

| who | pull requests | some of what they did |
|---|---|---|
| @Aadi-Tiwari | 117 | Open doors: tilted grasp, clearance path, via-point pull; Sample kitchen: fridge, counters and stove tile the wall wit; Scan regression suite and full RoomPlan coverage |
| @aarochu | 96 | CAP-4: Add field scanning protocol; DEM-3: Refresh release status and setup guidance; TWN: Avoid duplicate counters over scanned base cabinets |
| @vrun-systems26 | 41 | SIM: open_fridge push follows the fixture's yaw (side-wall f; SIM: real kitchen cabinet doors work in a blind corner and o; TWN: hinge a fridge on the side with less room in front (rot |
| @Anandb71 | 40 | close_door, open_oven: keep every arm joint off its stops; Draw RoomPlan doors, windows and openings in the sim as visu; Upload worker: retry a failed job a bounded number of times |
| @AvnehSBhatia (repo owner, with Claude Code) | 40 | Add tonight's MVP handoff; Integrate 16 open PRs, 40+ bug fixes, new web page, refreshe; Before the deadline: blocked items, laptop sprint, known bug |
| @rishith-c | 37 | CAP-2: Preserve complete frame archives when ZIP creation fa; CAP-1: Reject unsupported capture server URL schemes; CAP-1: Match iOS capture UI to Arrival and protect saved sur |
| @amanx98 | 19 | [POL-2] Wire Nemotron VLMPlanner into sim colle…; Implement Tavily AI Research Agent for appliance manuals and; Mobile Manipulator Base & 100% Physics-Checked Counter-to-Co |
| @AKSHAJ-SHELL | 14 | Cabinet fronts collide where they are drawn; arm clipping te; Count arm-fixture contacts in eval; A scanned office becomes a room the arm can work in |
| @JmarbinGG | 9 | iOS RoomPlan capture app: scan, share, upload to /jobs; Timed keyboard teleop baseline for the cost comparison; Single config loader, .env.example and secret redaction |
| @IshaanM05 | 6 | DEM-3: pin video writer to the FFMPEG backend; DAT-1: GR00T N1.7 modality config for the sim Panda; POL-3: gr00t_shim run against real GR00T, plus GPU run notes |
| @man-coding-j | 5 | Include arm-link bodies in walker safety yield under --base ; Object-centric SE(3) trajectory retargeting for counter_to_s; TWN-3: Draw visual door leaf and knob for open RoomPlan door |
| @raunak2007 | 2 | Planner benchmark v2: more inventories, blind requests, reje; Planner prompt lists the room's families, not 9 of 15: unlis |
| @AdvayRoongta | 1 | Cleanup: no import cycles, cohesive modules, docstrings, one |
| @Kr4ugs | 1 | Hygiene: imports at the top of the module, one import per li |
| @julieyixuanye-vm | 1 | docs: align architecture with implementation and formalize T |
| @wwpatel | 1 | SIM-1, MAN-1: kitchen and Panda arm visuals |
| @rishith-c (fork branch) | 0 | Synchronized ego, wrist and exocentric capture, side-by-side preview, optional t |

<!-- contributors:end -->

## To-do

<!-- todo:start (generated by scripts/render_todo_readme.py from site/data/todo.json) -->

As of 2026-10-28: **67** shipped, **0** in review, **0** next, **13** blocked, of **80** items. 418 pull requests merged, 2 open. The same list is on the demo site (`site/data/todo.json`). A row is shipped only when its pull request is merged on `main`.

Sizes: S under half a day, M about a day, L several days. Priority: **must** (the demo or the core claim breaks without it), **should**, nice. Anything that needs real data reports "not measured" until the blocked rows land.

### Shipped on main (25)

Merged PRs, grouped by what they do.

| id | what | PRs | who |
|---|---|---|---|
| S01 | Sample kitchen tiles the wall with no overlapping fixtures | #5 | @Aadi-Tiwari |
| S02 | Scan regression suite and full RoomPlan coverage, two real public exports | #10 | @Aadi-Tiwari |
| S03 | Kitchen and Panda arm visuals; wrist camera framing and joint 7 off its stop | #2, #71 | @wwpatel, @Aadi-Tiwari |
| S04 | Open drawer and open door, with collision boxes that match the drawn fronts | #4, #47, #338 | @Aadi-Tiwari, @aarochu, @vrun-systems26 |
| S05 | Close drawer, close door, put mug in drawer, wipe counter; a bowl the gripper can hold | #11 | @Aadi-Tiwari |
| S06 | Plates by the rim and pans by the handle | #31 | @Aadi-Tiwari |
| S07 | The arm opens the fridge: pull by the bar, push the inside face | #78, #298, #342 | @Aadi-Tiwari, @vrun-systems26 |
| S08 | Oven, dishwasher, microwave and wall-cabinet doors, faucet lever, nine more objects | #19 | @Aadi-Tiwari |
| S09 | Key, fill and window lights, tile, wood and stone materials | #21 | @Aadi-Tiwari |
| S10 | Walking people: proportioned bodies, a gait, slow stops at the counter | #23, #76 | @Aadi-Tiwari |
| S11 | A person opens the fridge and the held door counts for safety | #73 | @Aadi-Tiwari |
| S12 | Closed-loop eval harness, human-contact metric, arm-fixture contacts | #13, #55 | @Aadi-Tiwari, @AKSHAJ-SHELL |
| S13 | Upload worker: queued scan to twin to sim dataset to zip, with atomic job status | #12, #51 | @Aadi-Tiwari, @aarochu |
| S14 | GR00T modality config; validator rejects NaN, inf and bad timestamps | #38, #52 | @IshaanM05, @aarochu |
| S15 | Nemotron planner in sim collection; Tavily appliance research | #3, #35 | @amanx98 |
| S16 | iOS RoomPlan capture app and field scanning protocol | #43, #44 | @JmarbinGG, @aarochu |
| S17 | CI workflow, Makefile and CPU Dockerfile; one-command demo pipeline | #82, #86 | @AvnehSBhatia |
| S18 | Offline GR00T policy-server contract (mock server, shim, VLAClient fix) and loader conformance check | #83, #84 | @AvnehSBhatia |
| S19 | Scan intake, RGB-D frame format, tape-measure accuracy harness, realism metrics | #87, #90, #92, #88, #393 | @AvnehSBhatia, @vrun-systems26 |
| S20 | Time and cost report, seed-replay retargeting, appearance-variation stage with honest labels | #85, #89, #93 | @AvnehSBhatia |
| S21 | Docs and README claim checker, release QA, PyAV fix, refreshed demo video | #91, #29, #9, #50 | @AvnehSBhatia, @Aadi-Tiwari, @IshaanM05, @aarochu |
| S22 | Integration of 16 PRs and the sprint PRs #82-#93, plus the to-do docs | #80, #94, #1, #81, #95 | @AvnehSBhatia |
| S23 | Integration of #220-#250: scan, sim and docs fixes from the sprint | #251 | @aarochu, @Anandb71, @amanx98, @AvnehSBhatia |
| S24 | Integration of #201-#219 and the dependabot bumps, the Devpost feedback draft | #252 | @aarochu, @Anandb71, @AvnehSBhatia |
| S25 | Integration of #253-#301: CLI usage errors instead of tracebacks, fridge push on any wall, walkers yield to arm links, eval-compare, cost CLI, Windows memory cap, empty shipped API caches | #302 | @Aadi-Tiwari, @amanx98, @AKSHAJ-SHELL, @vrun-systems26, @AvnehSBhatia, Claude Code |

### On main (#194) (31)

Integrated on feat/more-prs-2 and merged to main in #194. A row is shipped only when main has it.

| id | what | size | priority | status | PRs or issue | who |
|---|---|---|---|---|---|---|
| L1 | Fixtures on any wall (#49, #75, #77 folded into one PR) Replaces #49, #75 and #77. The U-kitchen corner cabinet door now opens from a stand spot 0.25 m along its run. Its drawers, its close_door and the rotated kitchen fridge stay rejected. | M | must | Shipped | #99 | @AvnehSBhatia |
| L2 | Failed task skips instead of crashing; pre-flight reach check | S | must | Shipped | #99 | @AvnehSBhatia |
| L3 | Worker hard limits: subprocess timeout, size caps, stage-tagged errors | M | must | Shipped | #96 | @AvnehSBhatia |
| L4 | Harden the web app for public hosting | M | must | Shipped | #97 | @AvnehSBhatia |
| L5 | Shardable, resumable sim collection: --shard, --resume, merge-shards | M | must | Shipped | #104 | @AvnehSBhatia |
| L6 | Fresh-clone matrix: macOS and Windows in CI, wheel smoke, docker job, Windows checks fixed, separate rendering gate | M | must | Shipped | #98, #108 | @AvnehSBhatia, @Aadi-Tiwari |
| L7 | Docs and repo hygiene sweep Demo video and gif regenerated and the README status rows checked; the open-door row now states its measured rate. | S | must | Shipped | #98 | @AvnehSBhatia |
| L8 | One-command offline demo from a fresh venv, GL fallback | S | must | Shipped | #98 | @AvnehSBhatia |
| L9 | Non-kitchen rooms with room-appropriate objects (book, remote, laptop) Replaces #57. | M | should | Shipped | #105 | @AvnehSBhatia |
| L10 | Twin compile fixes on the real U-shaped scan | S | should | Shipped | #100 | @AvnehSBhatia |
| L11 | Scan robustness sweep: fuzzed RoomPlan variants | M | should | Shipped | #100 | @AvnehSBhatia |
| L12 | Per-episode quality validator | S | should | Shipped | #102 | @AvnehSBhatia |
| L13 | Mobile base on any wall: 11-float state and action through export, GR00T config and loader check Replaces #33 and #36. The fixed base stays the default; run the fixed-base GR00T check (G1) before training on the 11-float schema. | L | should | Shipped | #103 | @AvnehSBhatia |
| L14 | Scan frames through the upload path, compared against held-out views in the worker | S | should | Shipped | #97, #110 | @AvnehSBhatia, @Aadi-Tiwari |
| L15 | Provenance on twin objects: detected, default, authored | S | should | Shipped | #100 | @AvnehSBhatia |
| L16 | Realism command: JSON and side-by-side PNG (real frames still missing) | S | should | Shipped | #100 | @AvnehSBhatia |
| L17 | Reachability planner: IK solve of the expert's waypoints, fallback base grid | L | should | Shipped | #116 | @Aadi-Tiwari |
| L18 | Retention and deletion | S | should | Shipped | #96 | @AvnehSBhatia |
| L19 | Worker concurrency: atomic claim, heartbeat, cancel | M | should | Shipped | #96 | @AvnehSBhatia |
| L20 | Usage metering for the cost report; timed keyboard teleop baseline | M | should | Shipped | #96, #113 | @AvnehSBhatia, @JmarbinGG |
| L21 | CI extras: ruff, link check, Dependabot | S | should | Shipped | #98 | @AvnehSBhatia |
| L22 | Secrets handling: single config loader, .env.example, redaction, arrival doctor | S | should | Shipped | #97, #114 | @AvnehSBhatia, @JmarbinGG |
| L23 | Static demo-site export | S | should | Shipped | #102 | @AvnehSBhatia |
| L24 | Oven, dishwasher and faucet families on the any-wall chain Replaces #27. | M | nice | Shipped | #106 | @AvnehSBhatia |
| L25 | Real Menagerie Franka behind --arm menagerie Replaces #25. The 34 MB meshes are fetched by script and stay out of git. The close-door push standoff is retuned for the real hand: 14 of 14 on seed 5 with no people. Not measured over other seeds, rooms or people. | M | nice | Shipped | #107 | @AvnehSBhatia |
| L26 | Thin-object (spatula) pinch grasp, in counter_to_sink and wipe, both arms | M | nice | Shipped | #117 | @Aadi-Tiwari |
| L27 | People pass: yield and stop from the safety distance, counter-use pause, 3D head and arm clearance check On feat/demo-site with no pull request yet. Walkers also keep out of the door hinge disc (open_door 8 of 8 on seeds 0 to 3, was 7 of 8 on one), and fridges can be right-hinged with hinge: "right". Yield uses the closest of the base and the arm-link bodies. The hero fridge with a right hinge is rejected by the pre-flight because its free edge is in the corner. | M | nice | Shipped | issue #24 | @AvnehSBhatia, Claude Code |
| L28 | Multi-room summary command (real rooms wait on P2) | S | nice | Shipped | #102 | @AvnehSBhatia |
| L29 | Frozen evaluation trials: replay the same trials for any policy and keep every run | M | should | Shipped | #109 | @Aadi-Tiwari |
| L30 | Worker follow-ups: memory cap on Linux, DELETE /jobs/<id> with a retention note, cancel endpoint and per-episode progress DELETE /jobs/<id>, cancel and delete buttons, episode progress and an optional memory cap. The memory cap counts resident memory: the worker reads the job's process tree every 0.5 s and kills it past ARRIVAL_MAX_JOB_MB with a MemoryLimitError naming the stage (#192). The old RLIMIT_AS cap followed the core count and crashed jobs (docs/ref/MEMCAP.md). | M | should | Shipped | #192 | @AvnehSBhatia, Claude Code, @AKSHAJ-SHELL |
| L31 | Synchronized ego, wrist and exocentric capture from one simulation state, with a side-by-side preview and an optional three-camera GR00T config From rishith-c/feat/ego-exocentric-capture, merged on feat/more-prs-2 and on main via #194. Offline key and schema checks only; no three-view GR00T training or inference has been run. | M | nice | Shipped | none | @rishith-c |

### Landed after #194 (2)

Merged to main in #202.

| id | what | size | priority | status | PRs or issue | who |
|---|---|---|---|---|---|---|
| L32 | Demo page clicks: stage buttons, the scan/twin divider, and the phone disclaimer Checked in a browser at 1024 px and 375 px, both themes. The faded hero was still taking the stage-button clicks, the hidden dock cards were taking the divider drag, and on a phone the intro covered the disclaimer. Offline asset checks already passed. Not deployed (C2). | S | should | Shipped | #195 | @AKSHAJ-SHELL |
| L33 | Re-export site/data and stop the exporter deleting todo.json State rows are 11 wide; the page packs the first 8 (arm and gripper). Absolute paths are stripped. todo.json is kept across a re-export. Checked with the exporter tests and a headless page load. | S | should | Shipped | #196 | @JmarbinGG |

### Laptop: next (0)

Can be done on a laptop with no extra access.

| id | what | size | priority | status | PRs or issue | who |
|---|---|---|---|---|---|---|

### Needs a GPU (8)

A GPU host with 40 GB or more, or a Linux box.

| id | what | size | priority | status | PRs or issue | who |
|---|---|---|---|---|---|---|
| G1 | Real GR00T loader check and fine-tune on hero_kitchen_sim GR00T N1.7-3B fine-tuned on one 96 GB GPU and scored in the sim on identical trials. Best 39 of 48 (81%) against 26 of 48 for the clean-data baseline (paired sign test p = 0.007): clean data plus DART demos with the marginal drawer pulls removed. Self-reported in #165 and #63; small samples, and the base-GR00T and generic-RoboCasa cells (G3, G4) are not run, so `usefulness` stays null. Weights: https://huggingface.co/Ish1105/arrival-gr00t-n17-run4 (NVIDIA Open Model License, not the repo's Apache 2.0). The same card reports clean plus unfiltered DART at 22 of 48. The fine-tunes ran on one 96 GB GPU on Cosmos, not on Nebius. | M | must | Shipped | #165, #383 | @IshaanM05, @vrun-systems26 |
| G2 | Real GR00T policy server: chunking, both camera keys, golden-file test Ran against real GR00T on Oct 21 (Isaac-GR00T dac54b3, N1.7-3B, one 96 GB GPU): /act returns 16-step action chunks for a request with both camera keys, and eval --policy vla scored the fine-tunes through it (docs/ref/POL-3.md). The wire-format golden test and the 8-float and 11-float schemas are done. | M | must | Shipped | #83, #164 | @AvnehSBhatia, Claude Code |
| G3 | Three-cell held-out eval, 50 episodes per cell, Wilson intervals | L | must | Blocked | issue #64 | unassigned |
| G4 | Generic-RoboCasa baseline dataset and fine-tune | M | must | Blocked | issue #64 | unassigned |
| G5 | Dockerfile.gpu, run_finetune.sh, train and serve targets Dockerfile.gpu, scripts/run_finetune.sh and the train, eval-policy and serve-policy targets are written and never run on a GPU. Unverified items are listed in the build log. | L | must | Blocked | issue #63 | @AvnehSBhatia, Claude Code |
| G6 | Prove layout.yaml loads in real RoboCasa Built in real RoboCasa 1.0.1 (Oct 27): layout.yaml as written loads but RoboCasa reads its walls, floor, fronts and sink differently; arrival.twin.robocasa_layout converts it, and the hero kitchen and RoomPlan kitchen_real_01 then build with every fixture centre within 2.5 cm horizontally and no added overlap (scripts/check_robocasa_layout.py, docs/results/robocasa-layout-2026-10-27). The sink height is RoboCasa's (flush with the counter top). | M | must | Shipped | #418 | @vrun-systems26, @AvnehSBhatia, Claude Code |
| G7 | Real MimicGen retargeting for counter_to_sink Native object-centric SE(3) retargeting of counter_to_sink landed in #356 (arrival/generate/se3_retarget.py, physics-checked). Issue #62's benchmark ran on Oct 27: 14 of the 20 P3 teleop demos became seeds (retarget --teleop), and 140 attempts gave 104 physics-checked episodes that validate with 0 flagged, at 15.2 s each against 61.5 operator seconds per teleop demo (docs/results/g7-teleop-seeds-2026-10-27). Upstream MimicGen is still not integrated (arrival/generate/mimicgen.py only probes for RoboCasa). | L | should | Blocked | #356, #439 | @man-coding-j, @vrun-systems26 |
| G8 | Scan-appearance composite replacing the splat stub The depth compositor landed in #194 (arrival/render/splat.py: a static view and the moving geometry, same camera, by depth). No Gaussian splat is trained and no scan appearance is captured yet. | L | should | Blocked | #194 | unassigned |

### Needs cloud (7)

Cloud accounts, billing or deploy access.

| id | what | size | priority | status | PRs or issue | who |
|---|---|---|---|---|---|---|
| C6 | One real Nemotron call on Token Factory, kept as evidence Live on Oct 26: Nemotron Super plans (about 2.6 s each) and Lightning writes wordings through Token Factory. On 15 indirect requests the Nemotron planner picks the right task 13 times against 6 for the keyword parser, and on 54 Lightning-written wordings 52 against 47 (docs/results/planner-benchmark-2026-10-26.json). Evidence: docs/results/live-apis-2026-10-26.json. | S | must | Shipped | none | @AvnehSBhatia, Claude Code |
| C7 | One real Tavily call, for the Best Use of Tavily bonus Live on Oct 26: queries rewritten against the live API so they return figures (e.g. the fridge's federal 66.7 N opening-force limit, citing ecfr.gov), at the advanced depth. The safety guidelines beside them are still our built-in table, labelled built_in_table. | S | should | Shipped | none | @AvnehSBhatia, Claude Code |
| C1 | Nebius Serverless Jobs fan-out with resume, then one real small run Fan-out with done-markers, retry and resume is built. arrival fanout --backend nebius runs shards as Serverless AI Jobs with an Object Storage bucket, tested only against in-process fakes of both APIs. It has never run against Nebius. The hackathon rules count a project as running on Nebius only if it calls the Token Factory inference API or runs on Nebius AI Cloud (Serverless Jobs, Serverless Endpoints, or DevPods). Neither has been done. Not pursued for this submission: C6 meets the rule. | L | must | Blocked | issue #66 | @AvnehSBhatia, Claude Code |
| C2 | Hosted HTTPS demo with a pre-seeded room The static demo site is live at https://arrival.exealabs.org (Vercel). The hosted upload-to-dataset demo is not: smoke test (#115), /healthz, docker-compose.yml and docs/DEPLOY.md are written and tested on a laptop. The Devpost form does not require a demo URL on the Physical AI track. | M | must | Blocked | #115 | @JmarbinGG |
| C3 | Measured time and cost per demo against the teleop baseline Added arrival cost CLI subcommand with automated P3 teleop baseline comparison. Time is measured: retargeted counter_to_sink demos from P3's own seeds take 15.2 s of one CPU core each against 61.5 operator seconds, a 4.04 ratio from arrival cost (docs/results/g7-teleop-seeds-2026-10-27). Cloud dollar cost per demo remains to be measured with real usage. | S | should | Blocked | issue #67 | @amanx98, @vrun-systems26 |
| C4 | Detect small objects from posed frames with a VLM Live on Oct 26: Nemotron 3.5 Super VL on Token Factory read the six rendered fixture frames, boxed the mug in all three mug frames and nothing in the empty ones; the mug was placed 1.2 cm from its true centre (docs/results/live-detect-2026-10-26.json). Rendered frames, not a real scan: real RGB-D frames wait on P1. | M | should | Shipped | issue #60 | @AvnehSBhatia, Claude Code |
| C5 | Real Cosmos variation on a tenth of validated frames The restyle stage and its Cosmos client landed in #93 (arrival/render/variation.py, SIM-7a): about 10% of frames, edge and mask checks, honest labels. On Oct 27 NVIDIA Cosmos Transfer2.5-2B (edge control) ran for real on one RTX PRO 6000 and restyled three 93-frame sim episodes (open_fridge, counter_to_sink, open_drawer) into photoreal kitchens; all 279 frames passed the stage's edge check (correlation 0.74 to 0.83, threshold 0.6) and NVIDIA's guardrail (docs/results/cosmos-transfer-2026-10-27/). Still open: the dataset run, which calls the model through the HTTP client on a tenth of each dataset's frames. | M | nice | Blocked | #93 | unassigned |

### Needs people or an iPhone (6)

Physical rooms, a LiDAR iPhone, or a human.

| id | what | size | priority | status | PRs or issue | who |
|---|---|---|---|---|---|---|
| P1 | Field-test the iOS app on a LiDAR iPhone; RGB-D recorder in Swift First field test done: a kitchen scanned with the capture app on an iPhone 13 Pro Max ran end to end (tests/fixtures/roomplan/kitchen_real_01.json, 9 of 12 episodes valid, bd4e791). The Swift RGB-D recorder is not done. | L | must | Blocked | #156 | @AKSHAJ-SHELL |
| P2 | Three consented real kitchen scans with tape measurements; licence check of the two public scans Licence check done: neither public scan's source repo has a licence. Scans next. | L | must | Blocked | #156 | @AKSHAJ-SHELL |
| P5 | Replace the two unlicensed public RoomPlan fixtures with our own captures Licence risk resolved: unlicensed fixtures amal.json and shopyseek.json replaced with Apache 2.0 authored fixtures office.json and study.json (#310). Consented real phone captures of an office (table and chairs) and a bedroom (desk blocked by chair) are still wanted to validate phone noise on doors and windows. Uses: tests/fixtures/roomplan/README.md. | M | must | Shipped | #310 | @amanx98 |
| P3 | Twenty timed teleoperated counter_to_sink demos 20 successes in 26 attempts; 1230.359 operator seconds including failures. Basic validation passes; one long episode is retained. Evidence: docs/results/p3-teleop-2026-10-21/. C3 still needs the cloud comparison. | S | should | Shipped | #127 | @Aadi-Tiwari |
| P4 | Pitch video: at least 1 minute of the modules in action, under 3 minutes, public on YouTube Public on YouTube since Oct 29: https://youtu.be/_DsfMNQbM2w (2:41). | S | must | Shipped | #156, #195 | @AvnehSBhatia, @AKSHAJ-SHELL |
| OWN | Close issues that are done: #37 DAT-1, #63 DAT-2, #74 PPL-2, #79 MAN-5; decide #72 MAN-1 and #101 The earlier stale issues (#18 to #34) and superseded PRs are closed. #37 and #63 are covered by G1, #74 and #79 by the fridge work on main. #101 ignores site/, which the repo tracks and deploys: close it. | S |  | Blocked | none | unassigned |

### Devpost form (1)

Nebius x NVIDIA Global AI Hackathon, https://nebiusglobalaihackathon.devpost.com/. Deadline Friday, October 30, 2026, 10:00am Pacific. Physical AI is the track: a demo URL is not required, and with no robot the video needs at least one minute of the key modules (P4).

| id | what | size | priority | status | PRs or issue | who |
|---|---|---|---|---|---|---|
| SUB | Fill the Devpost form: Physical AI, description, public repo, and feedback Required on the form: the track, a text description of what it is, why, and how it works, a public GitHub URL whose About section shows the Apache 2.0 license, a README with setup instructions, the YouTube link (P4), and written feedback on Token Factory, AI Cloud, and the NVIDIA models used. Highlight GR00T and any Nemotron or Token Factory use. English. A demo URL is optional on this track (C2). If the project existed before August 26, 2026, say what changed during the submission period. The project has to run as the video and the text describe, and the rules require a real Token Factory call or a Nebius AI Cloud run (C1). | S | must | Blocked | none | unassigned |

### Pull requests not merged as they are

- #101, #472: #101 ignores site/, which the repo tracks and deploys to https://arrival.exealabs.org. Left out on purpose; close it. #472 (procedural drawer enclosure walls) fails the portable tests on Linux, macOS and Windows; it needs a fix before it can merge.

<!-- todo:end -->

## Join in: team plan for 30 people

Each task ends with its status and to-do id (see [To-do](#to-do)); credit is by PR. **To claim an open task:** open a GitHub issue titled with its ID (e.g. `[CAP-3] Scan 5 kitchens`), assign yourself, and post the link in the team channel. One owner per task. If a task is bigger than a day, split it into issues.

### Teams at a glance

| Team | People | Mission | Lead |
|---|---|---|---|
| **CAP**: Capture & LiDAR | 5 | Real iPhone scans of real rooms, and the app to capture them | _TBD_ |
| **TWN**: Twin & scan testing | 4 | Any real scan → a clean, correct digital twin | _TBD_ |
| **SIM**: Scene & realism | 4 | A kitchen that looks real, with every moving part working | _TBD_ |
| **MAN**: Robot & manipulation | 5 | Every task succeeding with a natural-looking arm | _TBD_ |
| **PPL**: People & safety | 2 | Believable moving people the robot has to work around | @vrun-systems26 |
| **POL**: Policy (VLA/VLM) | 4 | Nemotron planner + GR00T policy closing the loop | _TBD_ |
| **DAT**: Data & training | 3 | GR00T fine-tune on our data, and evaluation | _TBD_ |
| **OPS**: Nebius scale & backend | 2 | Uploads processed in the cloud at scale, costs tracked | _TBD_ |
| **DEM**: Demo, web & QA | 1 + everyone | Hosted page, pitch video, release checks | _TBD_ |

Pick leads first. Each lead keeps their team's tasks below up to date.

### CAP: Capture & LiDAR (5)

LiDAR iPhones/iPads: iPhone 12 Pro or newer Pro models, or iPad Pro 2020+.

- **CAP-1: Capture app.** Build the Apple RoomPlan sample ("Create a 3D model of an interior room") with an export button. It writes `CapturedRoom` JSON via `JSONEncoder` and the USDZ. Add AirDrop/share, then upload to `POST /jobs`. **Done in #43** (S16); field-tested on an iPhone 13 Pro Max (P1).
- **CAP-2: RGB-D frames.** Record ARKit RGB + depth frames + camera poses alongside RoomPlan (needed later for realism scoring and splats). **Partly done:** the frame format, loader and validator landed in #90 (S19), and the app has a recorder (`FrameRecorder.swift`, issue #58), but P1 still lists the Swift RGB-D recorder as not done.
- **CAP-3: Scan real rooms.** Scan **at least 10 kitchens** (straight run, L-shape, U-shape, galley, island) plus **5 non-kitchen rooms** (office, living room, bathroom, garage, dorm). For each one:
  - commit `scans/<room_id>/roomplan.json`, a `notes.md` (location type, phone model, iOS version, lighting, clutter level) and 2 reference photos;
  - **measure 3 distances with a tape measure** (counter length, counter height, sink width) and record them in `notes.md`.
  - Remove anything personal from the photos.

  **Partly done:** the intake tool landed in #87 (S19) and one real kitchen ran end to end (`tests/fixtures/roomplan/kitchen_real_01.json`, P1). The three measured, consented scans (P2) are open; no `scans/<room>/measurements.json` exists yet.
- **CAP-4: Scan protocol.** Write `docs/SCANNING.md`: how to walk the room, speed, lighting, what to avoid (mirrors, glass, clutter), how to name rooms, consent for scanning other people's homes. **Done in #44** (S16).
- **CAP-5: Licenses.** Check the licenses of the two public scans in `tests/fixtures/roomplan/`, and replace them with our own scans if unclear. **Done in #310** (P5): neither had a licence, so they were replaced with authored Apache 2.0 fixtures `office.json` and `study.json`.

### TWN: Twin & scan testing (4)

- **TWN-1: LiDAR accuracy test.** Use the CAP-3 tape measurements to write a test that compares twin dimensions with the tape measurements. Report the error per room and fail if it's over 5 cm. **Partly done:** the harness and `tests/test_lidar_accuracy.py` landed in #92 (S19), 5 cm default tolerance; no real room has tape measurements yet (P2).
- **TWN-2: Scan regression suite.** Every scan under `scans/` must normalize, compile and render with **no overlapping fixtures, nothing below the floor, and closed walls**. **Done in #10** (S02): `tests/test_scans.py` runs every `scans/<room>/scan.json` and `roomplan.json` through normalize, compile, `layout.yaml`, a footprint overlap check (a sink or stove inset in its counter is allowed), a floor check, a closed-wall check (1 cm) and a MuJoCo build; the hero `scan.json` overlaps are an expected failure until TWN-6 lands. `python scripts/render_rooms.py` writes a top-down PNG per room to `out/renders/`. New rooms from CAP-3 are picked up with no test changes.
- **TWN-3: RoomPlan coverage.** Handle doors, windows, openings, `polygonCorners` floors, islands, multi-story `story`, and every object category Apple emits (`dishwasher`, `oven`, `washerDryer`, `storage`, `table`, …). Done: doors, windows and openings land in the bundle with their wall and `is_open`, `polygonCorners` floors set the floor extent, a merged multi-story export yields one story per twin, counter-height blocks clear of the walls become `island: true` counters, all 16 Apple categories have a fixture type and a default size, one test per category (`tests/test_roomplan_coverage.py`). Doors, windows and openings are also drawn in the sim on their wall as visual-only geoms: a closed door is a frame, a leaf and a knob; an open door retains its frame and dark recess and adds a leaf and knob swung 90 degrees toward the room centre, following `hinge: "left"` or `"right"` (default left jamb when omitted); an opening is a frame around a dark recess; a window is a frame, a sill, a pane and a mullion. They never collide, so physics and task outcomes do not change (`tests/test_openings_sim.py`). A wall that carries `curve` (`startAngle`, `endAngle`, `radius`) is replaced by straight chords every 15° (at least two) instead of being dropped or treated as one straight run (`tests/test_roomplan_coverage.py`). Left: non-kitchen categories (bed, sofa, bathtub, …) are a plain `box` in the layout because RoboCasa has no such fixture; the wall behind a doorway stays solid, so walkers and the mobile base cannot go through it; no real multi-story export has been seen yet.
- **TWN-4: RoboCasa-native layout.** Make `layout.yaml` load in real RoboCasa (half extents, `wall_side`, groups). Done for the hero kitchen and RoomPlan `kitchen_real_01` (G6, Oct 27, RoboCasa 1.0.1): `layout.yaml` as written loads in RoboCasa's `KitchenArena` but builds the wrong room (walls and floor read as half extents on the wrong side, every front facing its wall, the sink floating in a solid counter, the stove 15 cm into the floor); `arrival.twin.robocasa_layout` turns it into RoboCasa's own form, and both kitchens then build and compile to MuJoCo with every fixture centre within 2.5 cm horizontally, walls exact, fronts matching the twin, and no overlap the twin does not already have (`scripts/check_robocasa_layout.py`, [docs/results/robocasa-layout-2026-10-27](docs/results/robocasa-layout-2026-10-27/README.md)). Left: RoboCasa sets the sink's height (flush with the counter top, 5.5 cm below the hero twin's sink box), `layout.yaml` itself still uses the twin's form, and no robot or task ran in the built room.
- **TWN-5: Object detection.** RoomPlan doesn't see mugs or bowls. Detect objects in CAP-2 photos (Nemotron Nano Omni via Token Factory, or any detector) and place them in the twin. **Partly done** (C4, issue #60): `arrival/twin/detect.py`; a live Nemotron VL run on rendered frames placed the mug 1.2 cm from its centre (`docs/results/live-detect-2026-10-26.json`). Real frames wait on P1.
- **TWN-6: Fix the sample kitchen.** **Done in #5** (S01): fridge, counters and stove now tile the north wall with no overlap, the bowl is on the counter instead of inside the fridge, and `tests/test_roomplan.py` checks footprints (see `docs/BUILD-LOG.md`).

### SIM: Scene & realism (4)

- **SIM-1: Kitchen visuals.** A recessed sink basin with a faucet, cabinet fronts with gaps and slim handles, toe-kick, countertop overhang, backsplash, stove burners, and a proper fridge. Code is in `arrival/sim/scene.py`. **Done in #2** (recessed fronts, slim handles, backsplash, sink with faucet, gas range, fridge with freezer door). Lighting and materials landed with SIM-2.
- **SIM-2: Lighting & materials.** *Done (#21).* Key and fill lights, wood/stone/tile textures, and lighting variation that still matches the `dim/neutral/bright` axis.
- **SIM-3: More moving parts.** Dishwasher door, oven door, microwave, faucet handle, upper cabinets, and drawers in every base cabinet, all on joints with names the policy can target. **Done in #19 (S08) except the extra drawer row:** oven door, dishwasher and microwave doors, the faucet lever and wall-cabinet doors are named joints with handle sites; `part_names()` in `arrival/sim/scene.py` lists them. A drawer row above the door bay would sit inside the MAN-2 door panel, so it was left out.
- **SIM-4: More objects.** Plates, cups, bottles, pans, fruit, utensils, with realistic mass, friction and size variation. **Done in #19** (S08): cup, plate, bottle, pan, apple and spatula join the mug, bowl and sponge (`arrival/catalog/categories.py`), with mass scaled by volume, per-material friction and seeded size variation; the expert picks the cup, apple and bottle 4/4. The 1 cm spatula is pinched across its width with the pad tips just over the counter (L26) and goes counter to sink and wipes the counter.
- **SIM-5: Non-kitchen rooms.** Twin and sim support for at least one office and one living room from CAP-3 scans. **Done in #105** (L9, replaces #57): non-kitchen rooms with room-appropriate objects (book, remote, laptop).
- **SIM-6: Splat appearance** *(stretch).* Gaussian splat of the scanned room composited with the sim. The design is in `docs/03-architecture.md`. **Open** (G8): the depth compositor `arrival/render/splat.py` landed in #194; no splat is trained.

### MAN: Robot & manipulation (5)

- **MAN-1: Make the arm look and move like a Panda.** Real link shapes and proportions, a Franka-style hand, a low mount, IK biased to the Panda "ready" posture so the elbow doesn't flip, smooth min-jerk motion, and no clipping. Code is in `arrival/sim/robot.py` and `ik.py`. **Done in #2:** Panda-shaped visuals over the unchanged collision capsules, a nullspace pull toward the Panda home posture in IK, and min-jerk motion with via-point drawer pulls.
- **MAN-2: Open doors.** **Done in #4 and #47** (S04, see `docs/BUILD-LOG.md`). Both cabinet doors open at 100%; the fridge opens too (#78, S07). Check with `python -m arrival sim --families open_door --episodes 4 --save-preview`.
- **MAN-3: Mobile base.** Add an actuated base so `counter_to_counter` (about 2 m between pick point and target) is reachable, and add base joints to the state/action spec. **Done in #103** (L13, replaces #33 and #36): `--base mobile`, 11-float state and action; the fixed base stays the default.
- **MAN-4: Grasp everything.** Bowl done: a hollow bowl and a rim grasp (see `docs/BUILD-LOG.md`). Cups, apples and bottles are picked by the body. Plates are picked by the rim and pans by the handle (#31). **Done in #11 and #31** (S05, S06).
- **MAN-5: New tasks.** **Done in #11** (S05): close drawer, close door, put the mug away in a drawer, wipe the counter with the sponge (`python -m arrival sim --families close_drawer,close_door,put_in_drawer,wipe_counter`). Open fridge landed (#78). Oven, dishwasher (open, load, unload) and faucet families run on the any-wall chain (L24, #106, which replaces #27).
- **MAN-6: Real Franka model** *(optional).* Swap in the MuJoCo Menagerie Panda (Apache-2.0) behind a flag, keeping tests offline. **Done in #107** (L25, replaces #25): `--arm menagerie`, meshes fetched by script.

### PPL: People & safety (2)

- **PPL-1: Better humans.** Proportioned bodies with walking leg and arm swing, varied clothing colours and heights, and no clipping through counters. **Done in #23 and #76** (S10).
- **PPL-2: Interactions.** People who stop at the counter, open the fridge, and cross the robot's path. Add a safety metric (minimum robot–human distance) to the episode data, and drop episodes with contact. *Done: walkers pause at the counter, cross behind the base and open the fridge (#73); `arrival/sim/safety.py` records the distance, contact episodes are dropped, and the held fridge door is not counted as an arm contact.*

### POL: Policy, VLA & VLM (4)

- **POL-1: VLM planner in the loop.** **Done in #3** (S15): `python -m arrival sim --planner vlm` runs `VLMPlanner` (Nemotron via Token Factory) inside `collect()`. Without a key it falls back to the rule parser. Responses cache in `out/cache/nemotron_cache.json`.
- **POL-2: Instruction generation.** **Done in #3:** `--instructions nemotron --split train|eval` draws varied wordings with a disjoint held-out set (`arrival/tasks/instructions.py`). The template path still keeps the wording and start-offset holdout from POL-4.
- **POL-3: GR00T policy server.** Run GR00T N1.7 inference (H100 or L40S), connect `VLAClient` via `GR00T_POLICY_URL`, and run closed-loop episodes in the sim. **Done** (G2, #83, #164): ran against real GR00T N1.7-3B on Oct 21 on one 96 GB GPU (`docs/ref/POL-3.md`).
- **POL-4: Closed-loop eval harness.** Score N episodes per task for any policy (success rate, time, collisions) in the scanned room. **Done in #13 and #55** (S12): `python -m arrival eval` (`arrival/eval/harness.py`), see docs/07-eval.md. Arm-fixture contacts are counted. It scored the GR00T fine-tunes through the live server (G2).

### DAT: Data & training (3)

- **DAT-1: GR00T modality config.** Write the config for our 7+1 embodiment, then confirm GR00T's real data loader reads `out/datasets/hero_kitchen_sim`. **Done:** config in #38 (S14), real loader check in #165 (G1).
- **DAT-2: Fine-tune smoke test.** Run a 100-step GR00T N1.7 fine-tune on the sim dataset, then a full run. The command is in `docs/REFERENCE.md`. **Done in #165** (G1): best fine-tune 39 of 48 against 26 of 48 for the clean-data baseline, small samples (`docs/07-eval.md`).
- **DAT-3: Evaluation.** Compare base GR00T, GR00T fine-tuned on generic RoboCasa data, and GR00T fine-tuned on our room. Use 50 episodes per cell on held-out poses and wordings, following the protocol in `docs/07-eval.md`. **Open** (G3, G4): the base-GR00T and generic-RoboCasa cells have not run.
- **DAT-4: MimicGen / scale.** Generate thousands of physics demos per room from a few seeds, and track success and throughput. **Partly done** (#89, #356, #439): a native same-object SE(3) generator retargets recorded `counter_to_sink` rollouts with physics checks (`arrival/generate/se3_retarget.py`). Issue #62's benchmark ran from 14 human teleop seeds: 104 physics-checked episodes from 140 attempts (`docs/results/g7-teleop-seeds-2026-10-27/`). Upstream MimicGen integration and scaling to thousands remain open (G7).

### OPS: Nebius scale & backend (2)

- **OPS-1: Upload worker.** **Done locally in #12 and #51** (S13): `python -m arrival work` turns a queued scan (RoomPlan export or `scan.json`) into a twin, a sim dataset and `out/jobs/<id>/dataset.zip`, with stage updates in `status.json` and `GET /jobs/<id>/download`. Several workers can share one queue (`claim.lock` and a heartbeat; a job whose worker died is requeued). A failed job is retried up to 2 times (`ARRIVAL_JOB_RETRIES` or `--retries`), but not for a bad upload, a request over a limit, a room the sim cannot use or a timeout; `status.json` counts `job_attempts` and keeps the earlier errors in `retry_errors`. A retry waits `ARRIVAL_JOB_RETRY_BACKOFF` seconds (default 5, doubling per attempt, capped at 300) before it can be claimed again. Left: running it on Nebius is OPS-2.
- **OPS-2: Nebius fan-out.** Serverless Jobs (one shard of episodes per job) and Object Storage for datasets, with checkpoint and resume. **Partly done** (C1): `arrival fanout --backend nebius` is built, tested only against fakes of both APIs, never run on Nebius; not pursued for this submission.
- **OPS-3: Cost tracking.** Fill the `dollars` fields in the report from real Nebius usage per product. **Partly done** (C3): the local time and cost report (#85) and `arrival cost` against the P3 teleop baseline (#127); dollar figures from real usage are open.

### DEM: Demo, web & QA (1 + everyone)

- **DEM-1: Hosted page.** Deploy the web app, with a 3D or video viewer for each room and its dataset. **Partly done** (C2): the static demo site is live at https://arrival.exealabs.org (L23, #102, #195); the hosted upload demo is not.
- **DEM-2: Pitch video.** Re-render `scripts/make_demo_video.py` once SIM-1 and MAN-1 land, and add real scan footage from CAP. **Partly done** (P4): the demo video was refreshed in #50, but `site/motion/pitch.mp4` is 24.4 s; the public YouTube video of at least a minute is open.
- **DEM-3: Release QA.** Before each demo, run a fresh-clone install, the full test suite, `sim` + `validate`, and check every README claim. Update the status table. **Ongoing:** release QA in #29, claim checker `scripts/check_docs.py` (S21).

### Pull requests

The PRs that were waiting on a rebase on Oct 19 are all resolved: #77 (with #49 and #75) landed through #99 (L1), #57 through #105 (L9), #33 and #36 through #103 (L13), #27 through #106 (L24) and #25 through #107 (L25). GitHub shows #49, #75, #77 and #57 as merged and #33, #36, #27 and #25 as closed; #45, #46, #48 and #53 were closed without merging. Which PRs closed which to-do is in the [To-do](#to-do) list; who wrote them is in [docs/CONTRIBUTORS.md](docs/CONTRIBUTORS.md).

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
- [docs/DEVPOST-SUBMISSION.md](docs/DEVPOST-SUBMISSION.md) and [docs/DEVPOST-FEEDBACK.md](docs/DEVPOST-FEEDBACK.md): the hackathon form text and feedback, with the lines that still need a real run marked
- [docs/CONTRIBUTORS.md](docs/CONTRIBUTORS.md): who worked on what, by pull request and by commit
- [docs/SUBMISSION-CHECKLIST.md](docs/SUBMISSION-CHECKLIST.md): the hackathon rules, checked item by item before submitting (moved from the repo root's LOCAL-README.md)
- [P3 human baseline](docs/results/p3-teleop-2026-10-21/README.md): timed demos, attempt log, cost assumptions and dataset handoff
- [docs/00-START-HERE.md](docs/00-START-HERE.md): the original MVP build contract

## License

Apache License 2.0, an OSI-approved open-source license. See [LICENSE](LICENSE). Copyright 2026 Exea Labs. The fine-tuned GR00T N1.7 weights are under the NVIDIA Open Model License.
