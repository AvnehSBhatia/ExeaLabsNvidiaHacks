# GR00T fine-tune preparation for Arrival sim data

The repo's existing `configs/gr00t/arrival_panda_config.py` and
`arrival_panda_mobile_config.py` register `NEW_EMBODIMENT` for fixed and
mobile sim data. Both use the two RGB cameras, 7-joint arm, gripper and task
instruction; the mobile config also uses planar base `(x, y, yaw)`. Sim
dataset targets are absolute. The GR00T config models arm targets relative to
current joints and gripper targets as absolute; the mobile config also models
base targets as absolute. Both use a 16-step action horizon, matching the current
[GR00T custom embodiment example](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md).

Generate an all-video sim dataset first, then run the preflight from this repo:

```bash
python -m arrival sim --episodes 100 --people 0 --out out/training
python scripts/prepare_gr00t.py out/training/datasets/hero_kitchen_sim \
  --gr00t-root /path/to/Isaac-GR00T --steps 100
```

The preflight validates the LeRobot dataset and refuses placeholder joints,
nonphysical trajectories, sparse video, incompatible modality keys and
nonabsolute dataset targets. It selects the fixed or mobile config from
`meta/embodiment.json` and prints a fine-tune command; it does not load GR00T,
download weights or start a GPU job. Run the printed command in the
Isaac-GR00T checkout after checking its current CLI and available hardware.
For a full run, choose the step count and evaluation plan from the held-out
protocol in [07-eval.md](07-eval.md). Keep the 100-step smoke test and full
run results separate.

The configs' keys follow the dataset's `meta/modality.json`; their
`annotation.human.task_description` language key follows the
[GR00T data preparation guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md).

## Published checkpoint (run 4)

The best fine-tune, 39 of 48 in the sim against 26 of 48 for the clean-data model (paired sign test p = 0.007), is on Hugging Face: <https://huggingface.co/Ish1105/arrival-gr00t-n17-run4>. It is not stored in this repo (about 12 GB of F32 weights).

- Data: 480 clean plus 544 DART episodes, with 102 marginal drawer pulls removed. Clean plus unfiltered DART scored 22 of 48, so the filtering is what matters.
- Licence: the weights are under the NVIDIA Open Model License, not this repo's Apache 2.0.
- Limits: simulation only, one room, two tasks, one seed, one run per model. Not for controlling a real robot.
