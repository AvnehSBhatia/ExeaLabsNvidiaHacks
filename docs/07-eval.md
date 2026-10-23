# Evaluation

Three measurements come from the spec. Tonight only the first one has numbers, and those numbers are for the scripted generator, not for teleoperation and not for MimicGen. The report schema is in [04-contracts.md](04-contracts.md). Leave `teleop_baseline`, `usefulness`, and `realism` as `null`.

## Speed and cost

Claim it supports: generated demonstrations are cheaper and faster than teleoperation.

How, when someone measures it for real:

- Count successful episodes and wall-clock seconds for a finished run.
- Dollars are the sum of Nebius invoices for that run, split by product: Token Factory, Serverless Jobs, Cosmos endpoint, Object Storage, app endpoint.
- Baseline: teleoperate the same task in the twin with RoboCasa's teleop tools, time 20 successes, and divide. That baseline is not part of tonight.

Tonight: fill successes, attempts, seconds, and demos per hour from the local run. Dollars stay unmeasured except local CPU at $0. The README must not convert that $0 into the claim "data is far cheaper." It may say the local placeholder cost $0 and the comparison is not yet measured.

## Usefulness

Claim it supports: the data teaches the room.

Protocol, later, in simulation of the scanned room only:

- Tasks: `counter_to_sink` and `counter_to_counter`.
- Held-out variations: object poses and wordings not used in training. Split by holding out wording template 4 and the start offset `[0, +0.08]`.
- Three cells, same held-out set, 50 episodes per cell:
  1. GR00T N1.7 base, no fine-tune.
  2. GR00T N1.7 fine-tuned on generic RoboCasa data.
  3. GR00T N1.7 fine-tuned on this room's dataset.
- Success: object within 0.05 m horizontally of the target point and released (gripper open at the end).
- If the gain of cell 3 over cell 2 is inside a 5 to 6 percent absolute band, report no gain. Do not fish for a seed that crosses the band.
- If GR00T N1.7 does not load the Panda embodiment, train the LeRobot policy from <https://github.com/nebius/serverless-ai-cookbook/tree/main/robotics/lerobot-finetune-job> and label the cell with that policy name.

Tonight: do not run this. `usefulness` is `null`.

### Results so far (Oct 22 2026): fine-tunes of our own data, not the three cells above

These are not the protocol's cells: there is no base-GR00T cell and no generic-RoboCasa cell, the trials are not the held-out wordings, and `usefulness` stays `null`. They compare our own fine-tunes against each other on identical trials (`eval --manifest`, one policy server per model, `counter_to_sink` and `open_drawer`, no people, 224 px), 24 trials per family.

| model | data | steps | counter_to_sink | open_drawer | total |
|---|---|---|---|---|---|
| run 2 | 480 clean episodes | 3,200 | 6/24 (25%) | 20/24 (83%) | 26/48 (54%) |
| run 3 | the same plus 646 DART episodes (`sim --dart 0.025`) | 4,000 | 13/24 (54%) | 9/24 (38%) | 22/48 (46%) |
| run 4 | 480 clean plus 544 filtered DART episodes (307 `counter_to_sink`, and the 237 drawer demos that open at least 0.28 m; `arrival subset --min-drawer-opening 0.28`) | 4,000 | 18/24 (75%) | 21/24 (88%) | **39/48 (81%)** |

Paired exact sign tests, run 3 against run 2: total 22 against 26 (p = 0.56, no reliable difference); counter_to_sink run 3 wins 11 trials and loses 4 (p = 0.12, suggestive: mugs 1 of 8 to 6 of 8); open_drawer run 3 wins 0 and loses 11 (p = 0.001). An earlier 6,000-step run on 320 clean episodes scored 10/24; its checkpoint was lost with the notebook.

Run 4 is the same recipe as run 3 (same steps, batch and settings) with the marginal drawer demos removed. Against run 2: total 17 trials won and 4 lost (p = 0.007); counter_to_sink 14 won, 2 lost (p = 0.004; bowls 14 of 16 against 5 of 16); open_drawer 3 won, 2 lost (p = 1.0: drawers are back at run 2's level). Against run 3: total 20 won, 3 lost (p < 0.001); open_drawer 12 won, 0 lost (p < 0.001); counter_to_sink 8 won, 3 lost (p = 0.23, so run 4 is not shown to beat run 3 there). Mugs are 4 of 8 (run 3: 6 of 8, run 2: 1 of 8); no mug difference is significant. Run 4's three failed drawers ended at 0.0, 0.002 and 0.128 m, against run 3's fifteen under-pulls of 0.05 to 0.15 m. The scoring that produced this table is a full re-score of the same checkpoint on the same manifest: the first run-4 scoring was lost when its notebook shut down (only a small part of it had been saved).

What was found, with how it was checked:

- The policy can see the object. Given frames from expert episodes, its first predicted target is within 0.3 cm of the expert's and within about 1 to 1.7 cm 0.4 to 0.75 s ahead (12 episodes, near the grasp). In closed loop, in the 4 failing mug rollouts I traced, it closed the gripper 2 to 8 cm from the mug, against a grasp tolerance of about 1 cm: errors compound into states the demonstrations never show.
- That is what DART addresses, and it helped the grasp task. It hurt drawers, and the cause is in the data. The clean expert ends every drawer pull at 0.300 m. With even 0.01 rad of execution noise about 30% of the kept drawer demos end under 0.28 m (36% at 0.025, 47% at 0.04), all above the 0.15 m success bar, so the physics check keeps them. In the 646-episode set 102 of 339 drawer demos are under 0.28 m. Run 3's failed drawers opened 5 to 15 cm: under-pulls, not misses.
- The same checkpoint scored 2/12 and 7/12 on the same 12 trials, so a 12-trial score cannot separate two models. Replanning every 4 steps, 16 denoising steps, and averaging 8 sampled chunks (`gr00t_shim --samples`, #161) each showed no closed-loop gain on counter_to_sink (2 of 24, against roughly 10 to 17% for the baseline, 24 trials each; replanning every 8 steps was not finished).
- Removing the 102 marginal drawer demos and retraining with the same recipe (run 4) restored the drawers (21/24) and kept the grasp-task gain (18/24): the data-quality cause held up under a test that could have failed. Only one retrain was scored, so no variant was picked from several.
- Scoring a checkpoint on the GPU that is training it slowed training from 1.4 to 0.15 steps/s; two clients on one policy server produced intermittent HTTP 500s (a lock fixes it).

Not shown: whether DART helps in general or in other rooms (one seed and one run per model; runs 3 and 4 also trained 800 steps longer than run 2), run 4's gain over run 3 on counter_to_sink, any result on held-out wordings, or anything about counter_to_counter. The 48 trials were also how the drawer problem was found; the 0.28 m bar comes from the datasets themselves (the clean expert ends every pull at 0.300 m), not from tuning on these trials.


## Realism

Claim it supports: the twin looks like the room.

Protocol, later: for scan frames that have RGB, depth, and a camera pose, render the twin from that pose and report image error and depth error. Do not invent a metric name until the frames exist. Candidate metrics to compute then, not tonight: mean L1 on depth in meters, and a pixel error on RGB. Mask out the robot if it is not in the scan photo.

Blocked when `blocks_claims` contains `realism` (the four-photo bundle).

Tonight: no scan frames with poses are in the repo. `realism` is `null`.

## What runs today

`python -m arrival eval --policy scripted|vla --families <list|all> --episodes N --seed S --people P [--holdout]` is the harness for the usefulness protocol, sim-in-the-twin only. It runs N episodes per family through the same MuJoCo env and physics success check as `sim`, writes no dataset, and reports per family: success rate with a Wilson 95% interval, mean sim time (frames / 20 Hz), mean wall time, human contacts, the smallest robot-human distance, and the largest IK residual. An episode in which the robot reaches a person is a failure. The split above is enforced in `arrival/sim/tasks.py`: `sim` datasets never contain wording template 4 or the start offset `[0, +0.08]`, and `--holdout` draws only those (`open_drawer` and `open_door` have three wordings and no start pose, so they keep all three either way). The sim applies half the y offset, so the held-out start is +0.04 m in y. The output is `out/eval/<run_id>/report.json`, shape in [04-contracts.md](04-contracts.md).

What it does not do yet: the success rule above (0.05 m of the target point, gripper open) is not what it checks; it uses the physics rule from `collect.py` (object resting inside the target fixture, or the joint past its threshold). Arm-fixture collisions are counted, not scored: an episode where an arm link (link1 to link7; the hand and fingers are meant to touch handles) touches anything but the robot, people or free objects is reported per family as `arm_contact_episodes` with the deepest penetration (`arm_contact_max_depth_m`), and the worst pair is logged. `--policy vla` has only been run far enough to confirm it stops with a message when `GR00T_POLICY_URL` is unset; the three cells need a GR00T server (POL-3) and a fine-tune (DAT-2). With 4 episodes per family the Wilson interval is 0.5 wide; the protocol's 50 per cell brings it to about 0.1.

## Replaying identical trials

The first evaluation freezes its full twin, ordered tasks, seed, people count, holdout split, resolution and fixed-base rollout settings in `out/eval/<run_id>/manifest.json`. Its SHA-256 ID covers the canonical JSON contents. Replaying this file does not compile the room or generate tasks again. A changed hash, inconsistent twin, duplicate task ID or unsupported rollout setting is refused before simulation.

Use `--manifest PATH --policy vla --checkpoint-id LABEL` for each policy server checkpoint. Copy the manifest path printed by the first run. `--manifest` cannot be combined with generation options such as `--room`, `--seed`, `--episodes`, `--people`, `--res` or `--holdout`; these values come from the file. VLA runs require a nonempty checkpoint label. This is a caller declaration, not evidence of which weights the remote server loaded.

`python -m arrival eval-compare out/eval/<run_a> out/eval/<run_b>` pairs two such runs by trial and prints, per family and in total, the successes of each run, the trials only one of them won and the exact two-sided sign test over those. It refuses runs with different manifest IDs or different trial sets, so a comparison cannot mix unlike trials. It needs only the two `episodes.jsonl` files, not the simulator.

Every invocation gets a new run directory, even with the same policy and checkpoint label. `episodes.jsonl` records each completed or failed attempt with its trial index, task ID, physics success, contact-adjusted success, safety measurements, times, error, manifest ID and checkpoint ID. The aggregate report is derived from these rows. Failed or invalid rollouts have unknown simulation time and safety measurements, shown as null. Means and contact counts use only `observed_episodes`; all attempts remain in the success-rate denominator. If the first server request fails, the attempt and manifest remain on disk but there is no completed `report.json`; the cost reader ignores that partial run. Families with no generated tasks are listed as omitted.

The manifest fixes inputs, not outcomes across software versions. Reports record Python, MuJoCo and NumPy versions; comparisons should also use the same source revision and policy settings. This implementation supports the fixed base only. Replay and mock-server tests establish evaluation bookkeeping, not learned-policy usefulness. The three-cell experiment still needs real checkpoints and GPU access.

## What a demo video may say

If a video is cut later, it may show the local page, the schematic frames, and the dataset folder. It must say the motion is a scripted placeholder in simulation. It must not show a physical robot, a splat, or a fine-tuned policy unless those were actually run.
