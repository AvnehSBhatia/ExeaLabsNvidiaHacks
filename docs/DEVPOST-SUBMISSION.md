# Devpost submission: text to paste

Form: https://nebiusglobalaihackathon.devpost.com/ (deadline Friday, October 30, 2026, 10:00am Pacific). Each section below is one form field. Lines in **[brackets]** must be filled in or deleted before you submit; do not submit a bracket. Every number here comes from the README or `docs/07-eval.md`; if one of those changes, change it here too.

## Track

Physical AI.

## Name and tagline

Arrival: scan a room once, get robot training data for that exact room.

## Description

### The problem

General robot policies such as NVIDIA GR00T learn in generic simulated kitchens. When a robot arrives in a real home, it has never seen that kitchen's counters, sink, drawers or fridge, and teleoperating new demonstrations in every home does not scale.

### What Arrival does

Arrival turns one iPhone LiDAR scan of a room into GR00T-ready training data for that room:

1. **Scan.** An Apple RoomPlan export from an iPhone describes the walls, counters, sink, stove, fridge, doors and drawers. A small iOS app (`ios/`) captures and uploads the scan.
2. **Twin.** The scan is compiled into a digital twin: a RoboCasa-style layout and a MuJoCo scene where every drawer, door, fridge and oven is a named joint with a handle.
3. **Simulate.** A 7-joint arm does kitchen tasks in the twin (pick and place to the sink or another counter, open and close drawers and doors, put a mug in a drawer, wipe the counter, open the fridge, oven and dishwasher; the README's status table gives each one's measured rate) while simulated people walk through the kitchen. Grasps use friction physics, success is checked by the simulator, and episodes with human contact are dropped.
4. **Capture.** Every successful episode is written as a GR00T-compatible LeRobot v2 dataset: joint states, actions, ego, wrist and optional room cameras, and language instructions.
5. **Fine-tune and evaluate.** GR00T N1.7-3B fine-tuned on one room's data scored 39 of 48 paired closed-loop trials against 26 of 48 for the clean-data baseline (paired sign test p = 0.007). The checkpoint is public: https://huggingface.co/Ish1105/arrival-gr00t-n17-run4.

### NVIDIA and Nebius

- **GR00T N1.7-3B**: the policy we fine-tune and score, with a modality config for our embodiment and a policy-server client used by the closed-loop eval.
- **Nemotron on Nebius Token Factory**: plans each task from the instruction and the room's inventory (`sim --planner vlm`), writes varied, held-out instruction wordings (`--instructions nemotron`), and finds small objects in scan frames. Live, with `nvidia/nemotron-3-super-120b-a12b` for plans (about 2.6 s each) and `nvidia/Nemotron-3_5-Lightning` for wordings. On 15 indirect requests such as "let me check if we're out of milk", the Nemotron planner picked the right task 13 times against 6 for a keyword parser. On 54 wordings Lightning wrote, it was 52 against 47 (`scripts/planner_benchmark.py`). `nvidia/Nemotron-3.5-Super-VL` finds small objects in posed scan frames: on rendered frames of the twin it found the mug in every frame that has one, nothing in the empty ones, and the mug was placed 1.2 cm from its true centre.
- **Nebius AI Cloud Serverless Jobs and Object Storage**: `arrival fanout --backend nebius` splits simulation into resumable shards, each run as a Serverless Job, with results in an Object Storage bucket. Built and tested against fakes; not yet run on Nebius.
- **Tavily**: researches the appliances found in each scan (hinges, clearances, safety guidance) and passes the findings to the Nemotron planner (`sim --planner vlm --tavily`). Live: the fridge query, for example, returns the federal limit of 66.7 N (15 lb) to open an empty door, citing ecfr.gov and GE.

### How it is built

Python 3.11, MuJoCo, a LeRobot v2 writer and validator, a local web app with an upload queue and workers (cancel, retry, memory and time limits, retention), shardable and resumable collection, a frozen-trial evaluation harness with Wilson intervals and a paired comparison command, and a time and cost report against a human teleoperation baseline (20 timed teleop demos). It is tested on Linux, macOS and Windows in CI.

### Honest limits

The results are from simulation only: one room, two tasks in the scored fine-tune, one seed, one run per model, and small samples. Base GR00T and a generic RoboCasa baseline were not scored. It has not run on a real robot. One real kitchen, scanned on an iPhone 13 Pro Max with our capture app, ran end to end (9 of 12 episodes valid); the GR00T fine-tune used the authored sample kitchen. **[Update if more real scans land (P2).]**

### What's next

Real scans of consenting homes, the three-cell usefulness eval (room-specific vs generic vs base GR00T), and fanning simulation out on Nebius so a new room produces its data in minutes.

## Links

- Code: https://github.com/AvnehSBhatia/ExeaLabsNvidiaHacks (Apache 2.0)
- Demo site (optional on this track): https://arrival.exealabs.org
- Video: https://youtu.be/_DsfMNQbM2w (public, 2:41)
- Checkpoint: https://huggingface.co/Ish1105/arrival-gr00t-n17-run4

## Built during the submission period?

Yes. The repository's first commit is October 12, 2026, after the August 26, 2026 start, so no "what changed" note is needed. Say so if the form asks.

## Feedback field

Paste `docs/DEVPOST-FEEDBACK.md` after its **[after the real call]** lines are filled in or deleted.
