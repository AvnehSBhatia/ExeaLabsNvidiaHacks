# Contracts

Implement these shapes. Field names are exact.

## ScanBundle

`scans/<room_id>/scan.json`

```json
{
  "schema_version": "1",
  "room_id": "hero_kitchen",
  "units": "meters",
  "frame": "z_up",
  "geometry_source": "authored_sample",
  "blocks_claims": [],
  "walls": [
    {
      "id": "wall_north",
      "start": [0.0, 0.0],
      "end": [4.0, 0.0],
      "height": 2.4,
      "thickness": 0.02
    }
  ],
  "fixtures": [
    {
      "id": "counter_a",
      "category": "counter",
      "size": [1.45, 0.65, 0.92],
      "position": [1.475, 0.4, 0.46],
      "yaw": 0.0
    }
  ],
  "objects": [
    {
      "id": "mug_1",
      "label": "mug",
      "size": [0.08, 0.08, 0.1],
      "position": [1.0, 0.35, 0.97],
      "yaw": 0.0
    }
  ]
}
```

Rules:

- `size` is full extent `[x, y, z]`, not half-extent.
- `position` is the center.
- `walls[].start` and `end` are `[x, y]` on the floor. The wall runs from start to end. Thickness extends toward the room interior. Tonight's sample walls use the rectangle below; do not infer interior from a polygon library. Store thickness and use it as the wall box Y or X size in the layout.
- `geometry_source` is one of `authored_sample`, `roomplan`, `four_photo_unmeasured`.
- `blocks_claims` is a list of strings. Four-photo bundles set it to `["geometry", "realism"]`. Others use `[]`.
- Optional keys, written by the RoomPlan normalizer and absent from authored scans; loaders accept their absence. `doors`, `windows` and `openings` are lists of `{"id", "wall_id", "size", "position", "yaw"}` (doors also carry `"is_open"`); `wall_id` is the id of the wall the surface sits in, or `null` when the export's parent is not one of the walls; `size` is `[width, 0, height]` as scanned. `floors` is a list of `{"id", "z", "corners"}` with `corners` a list of `[x, y]` floor-polygon points in the room frame and `z` the floor's height above z = 0. `story` and `story_count` are integers (which story this bundle holds, and how many the export had).
- Reject files with `schema_version` other than `"1"`, or `units` other than `"meters"`, or `frame` other than `"z_up"`.

### Hero kitchen that must be committed

Walls, counterclockwise, floor rectangle 4.0 m by 3.0 m:

| id | start | end | height | thickness |
|---|---|---|---|---|
| wall_north | [0, 0] | [4, 0] | 2.4 | 0.02 |
| wall_east | [4, 0] | [4, 3] | 2.4 | 0.02 |
| wall_south | [4, 3] | [0, 3] | 2.4 | 0.02 |
| wall_west | [0, 3] | [0, 0] | 2.4 | 0.02 |

Fixtures:

| id | category | size | position | yaw |
|---|---|---|---|---|
| counter_a | counter | [1.45, 0.65, 0.92] | [1.475, 0.40, 0.46] | 0 |
| sink_1 | sink | [0.60, 0.50, 0.25] | [1.6, 0.40, 0.85] | 0 |
| counter_b | counter | [1.1, 0.65, 0.92] | [3.45, 0.40, 0.46] | 0 |
| fridge_1 | refrigerator | [0.75, 0.70, 1.80] | [0.375, 0.40, 0.90] | 0 |
| stove_1 | stove | [0.70, 0.65, 0.90] | [2.55, 0.40, 0.45] | 0 |

Objects:

| id | label | size | position |
|---|---|---|---|
| mug_1 | mug | [0.08, 0.08, 0.10] | [1.0, 0.35, 0.97] |
| bowl_1 | bowl | [0.14, 0.14, 0.06] | [1.15, 0.45, 0.95] |
| sponge_1 | sponge | [0.10, 0.06, 0.03] | [3.3, 0.35, 0.935] |

`room_id` is `hero_kitchen`. `geometry_source` is `authored_sample`.

## RoomPlan JSON → ScanBundle

Input is a subset of Apple's CapturedRoom JSON. Extra keys are ignored.

```json
{
  "walls": [
    {
      "identifier": "wall_north",
      "dimensions": [4.0, 2.4, 0.02],
      "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 2.0,1.2,-0.01, 1]
    }
  ],
  "floors": [],
  "doors": [],
  "windows": [],
  "openings": [],
  "objects": [
    {
      "identifier": "sink_1",
      "category": {"sink": {}},
      "confidence": {"high": {}},
      "dimensions": [0.6, 0.25, 0.5],
      "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 1.6,0.85,-0.40, 1]
    }
  ]
}
```

`transform` is 16 numbers, column-major, ARKit Y-up. Translation is indexes 12, 13, 14. Apple encodes Swift enums as keyed objects (`{"sink": {}}`, `{"washerDryer": {}}`, `{"high": {}}`); the normalizer also accepts plain strings, snake-cases camelCase categories, drops `low`-confidence objects, subtracts the floor height (`floors[].transform[13]`, else the lowest wall bottom), rotates the room so the longest wall runs along +X, translates the wall bounding box to (0, 0), and snaps yaws within 5° of a multiple of 90°. Walls with zero thickness get 0.1 m.

Conversion:

```
x = m[12]
y = -m[14]
z = m[13]
size = [dimensions[0], dimensions[2], dimensions[1]]
```

Yaw: the local +X axis (first column) mapped the same way, then `atan2` of its horizontal components. Identity rotation yields yaw `0`.

Walls: treat each wall as a Y-up box whose center converts like an object. Project the box onto the floor and store `start` and `end` as the two endpoints of the long horizontal edge. `height` is `dimensions[1]`. `thickness` is the smaller of `dimensions[0]` and `dimensions[2]`.

Doors, windows and openings: each is a Surface with `dimensions [width, height, 0]`, a `transform` like a wall's, and `parentIdentifier` naming the wall it sits in. They convert like objects (center, size, yaw, then the same rotation and translation as the room) into `doors`, `windows` and `openings`. `wall_id` is the parent identifier when it matches a wall in the export, else `null`. A door's category is `{"door": {"isOpen": bool}}`; that becomes `is_open`.

Floors: `polygonCorners` (iOS 17) lists the floor outline in the floor's local plane coordinates `[x, y, ~0]`, and the floor's `transform` maps them into the room. Each floor with 3 or more corners becomes a `floors` entry with the corners in the aligned Z-up frame and `z` relative to the floor height. On both public exports the polygon bounding box equals the wall bounding box. A floor without `polygonCorners` adds nothing, and the compiler falls back to the wall bounding box.

Stories: `story` is an integer on the room, on every surface and on every object (iOS 17). A `CapturedStructure` export (several scans merged) carries a `rooms` array of CapturedRooms; the normalizer flattens it, tagging each element with its room's `story`. One story is kept per bundle: `to_bundle(..., story=n)` or `load(path, room_id, story=n)`, default the story with the most objects (lowest index on a tie). The floor height is measured on the kept story only. `story` and `story_count` are recorded in the bundle.

Object categories pass through unchanged after snake-casing: `sink`, `refrigerator`, `stove`, `oven`, `dishwasher`, `storage`, `table`, `washer_dryer`, `chair`, `sofa`, `bed`, `bathtub`, `toilet`, `television`, `fireplace`, `stairs` (the full `CapturedRoom.Object.Category` list as of iOS 17). An object whose `dimensions` are missing or not positive gets the category's default size from `arrival.catalog.fixtures.DEFAULT_SIZES`. Counters are not a RoomPlan category. The compiler synthesizes counters; the normalizer does not invent them from nothing. If the sample RoomPlan fixture has no counters, the compiler still emits counter boxes for `sink` and `stove` as specified in the compiler section.

Commit `scans/hero_kitchen/roomplan.json` containing the sink object above (translation x=1.6, y=0.85, z=-0.40) plus the four walls of the hero kitchen. A unit test normalizes it and checks the sink center equals `[1.6, 0.40, 0.85]` within `1e-6`, and size equals `[0.6, 0.5, 0.25]`.

Checked-in `roomplan.json` does not have to rebuild every fixture. The authored `scan.json` is what `demo` loads. The RoomPlan test only locks the coordinate conversion.

## Four-photo input

`bundle_from_photos(paths: list[path], room_id: str) -> ScanBundle`

- Require exactly 4 files.
- Copy them to `scans/<room_id>/photos/`.
- Return a ScanBundle that copies the hero kitchen geometry.
- Set `geometry_source` to `four_photo_unmeasured`.
- Set `blocks_claims` to `["geometry", "realism"]`.
- Do not call the network.

## Fixture map

| RoomPlan or scan category | Layout `type` |
|---|---|
| counter | counter |
| sink | sink |
| refrigerator | fridge_bottom_freezer |
| stove | stove |
| oven | stove |
| dishwasher | dishwasher |
| storage | hinge_cabinet |
| table | counter |
| microwave | microwave |
| washer_dryer, chair, sofa, bed, bathtub, toilet, television, fireplace, stairs | box |
| anything else | box |

`table` uses `default_config_name: dining` when the type is counter. Unknown categories still appear in the layout so the scan is not silently dropped. RoboCasa has no bed, sofa or bathtub fixture, so those categories stay `box` with their scanned size; `source_category` keeps the name.

Default sizes (full extents `[x, y, z]`, meters), used when a scan object has no usable dimensions:

| category | size | category | size |
|---|---|---|---|
| counter | 1.8, 0.65, 0.92 | washer_dryer | 0.6, 0.6, 0.85 |
| sink | 0.6, 0.5, 0.25 | chair | 0.5, 0.5, 0.9 |
| refrigerator | 0.75, 0.7, 1.8 | sofa | 2.0, 0.9, 0.85 |
| stove | 0.7, 0.65, 0.9 | bed | 1.6, 2.0, 0.5 |
| oven | 0.6, 0.6, 0.6 | bathtub | 1.7, 0.75, 0.55 |
| dishwasher | 0.6, 0.6, 0.85 | toilet | 0.4, 0.7, 0.8 |
| storage | 0.9, 0.5, 2.0 | television | 1.2, 0.1, 0.7 |
| table | 1.4, 0.8, 0.75 | fireplace | 1.2, 0.4, 1.0 |
| microwave | 0.5, 0.4, 0.3 | stairs | 1.0, 3.0, 2.7 |

Island: a `counter` or `table` fixture between 0.85 m and 1.15 m tall whose footprint is more than 0.3 m from every wall compiles to type `counter` with `island: true` and no `default_config_name`. Dining tables (about 0.75 m) stay `counter` with `dining`. `storage` is never an island; a freestanding sideboard stays `hinge_cabinet`. The rule needs walls; a bundle without walls never flags an island.

Object labels:

| label | catalog_id |
|---|---|
| mug | mug |
| bowl | bowl |
| plate | plate |
| sponge | sponge |
| apple | apple |
| bottle | bottle |
| pan | pan |
| cup | cup |
| spatula | spatula |
| anything else | box |

Object physics come from `arrival/catalog/objects.py`: shape, nominal size, mass at that size (scaled by volume for any other size), friction and colour. `PICKABLE` lists the ids the scripted expert holds by the body (mug, cup, sponge, apple, bottle); the bowl, plate, pan and spatula are scene objects only. `place_default_objects` draws each default object's size within 8% of the nominal size from its seed.

Moving parts in the sim (`arrival.sim.scene.part_names(twin)`): each is a joint, with a site `<joint>_handle` on its handle bar, in the parent fixture's frame.

| fixture type | joints |
|---|---|
| counter | `<id>_drawer0`, `<id>_drawer1` (slide, 0 to 0.4 m), `<id>_door` (hinge, 0 to 1.9 rad), `<id>_upper_door<k>` (hinge; one wall-cabinet door per 0.6 m of counter run, none over tables, islands or counters under 0.8 m) |
| sink inside a counter | `<sink_id>_faucet` (hinge, 0 to 1.2 rad; the lever up is water on) |
| stove | `<id>_oven_door` (hinge about the bottom edge, 0 to 1.5 rad; closed is held by gravity) |
| fridge_bottom_freezer | `<id>_door` (hinge, 0 to 2.0 rad) |
| dishwasher | `<id>_door` (hinge about the bottom edge, 0 to 1.5 rad); floor-standing; a counter run whose end overlaps it is trimmed back, like a fridge or stove (an appliance wholly inside a run is not cut out) |
| microwave | `<id>_door` (hinge on the left edge, 0 to 1.9 rad); the fixture `position` is the box centre, so it sits on a counter top |

## Twin

`out/twins/<room_id>/twin.json`

```json
{
  "room_id": "hero_kitchen",
  "geometry_source": "authored_sample",
  "frame": "z_up",
  "units": "meters",
  "fixtures": [
    {
      "id": "counter_a",
      "type": "counter",
      "size": [1.45, 0.65, 0.92],
      "position": [1.475, 0.4, 0.46],
      "yaw": 0.0,
      "source_category": "counter"
    }
  ],
  "objects": [
    {
      "id": "mug_1",
      "catalog_id": "mug",
      "size": [0.08, 0.08, 0.1],
      "position": [1.0, 0.35, 0.97],
      "yaw": 0.0
    }
  ]
}
```

`layout.yaml` is a RoboCasa-subset document. RoboCasa's real kitchens use `main_group` / `align_to` chains (see <https://github.com/robocasa/robocasa/blob/main/robocasa/models/assets/scenes/kitchen_layouts/train/layout011.yaml>). Tonight's emitter uses absolute `pos` and `size`, which that format also accepts, and one group:

```yaml
room:
  walls:
    - {name: wall_north, type: wall, size: [4.0, 2.4, 0.02], pos: [2.0, 0.0, 1.2]}
  floor:
    - {name: floor, type: floor, size: [4.0, 3.0, 0.02], pos: [2.0, 1.5, 0.0]}
main_group:
  group_origin: [0, 0]
  group_pos: [0.0, 0.0]
  group_z_rot: 0.0
  bottom_row:
    - name: counter_a
      type: counter
      size: [1.45, 0.65, 0.92]
      pos: [1.475, 0.4, 0.46]
```

Wall `pos` is the segment midpoint at half height: `[ (start.x+end.x)/2, (start.y+end.y)/2, height/2 ]`. Wall `size` is `[length, thickness, height]` for walls whose long axis is X, and `[thickness, length, height]` for walls whose long axis is Y. Floor `size` is `[width, depth, 0.02]` from the wall bounding box, `pos` at the bbox center with z `0`. When the twin has `floors` with corners, the bounding box of those corners is used instead.

Counter synthesis (only when the scan has a sink or stove whose centre is more than 0.3 m from every counter footprint and whose own footprint is more than 0.05 m from every counter, so a stove built in between two counters gets none): emit a counter of size `[max(fixture.size[0] + 0.4, 1.2), 0.65, 0.92]`, centered on the fixture's x,y, with z `0.46`. The authored sample already has counters, so this branch is covered by a unit test with a sink-only bundle, not by the hero scan.

Do not emit backing walls.

The twin copies `doors`, `windows`, `openings`, `floors`, `story` and `story_count` from the bundle when present, unchanged, and a fixture flagged as an island carries `island: true`; that flag is also written on its `layout.yaml` row. Doors and windows are not written to `layout.yaml` and the sim does not draw them yet.

## Tasks

`out/twins/<room_id>/tasks.json`

```json
{
  "room_id": "hero_kitchen",
  "tasks": [
    {
      "task_id": "counter_to_sink",
      "family": "counter_to_sink",
      "object_id": "mug_1",
      "source_fixture_id": "counter_a",
      "target_fixture_id": "sink_1",
      "start_position": [1.0, 0.35, 0.97],
      "lighting": "neutral",
      "instruction": "Pick up the mug from the counter and place it in the sink.",
      "mimicgen_template": "PnPCounterToSink"
    }
  ]
}
```

Families:

| family | source | target | mimicgen_template |
|---|---|---|---|
| counter_to_sink | a counter fixture | the sink | PnPCounterToSink |
| counter_to_counter | counter_a | counter_b | null |

`mimicgen_template` for counter-to-counter stays `null` unless a later agent finds a real RoboCasa template name. Do not invent one.

Variation generation, RNG seed 0, for listing tasks (the demo generator may subsample):

- Objects: every object whose horizontal center lies on `counter_a` (mug_1, bowl_1). Sponge is on counter_b and is not a source object.
- Start poses: the object's scanned xy, plus `[+0.08, 0]`, `[-0.08, 0]`, `[0, +0.08]`. Keep z. Reject a pose whose xy falls outside the source counter's top rectangle (center ± half size, inset by 0.05 m).
- Lighting: `dim`, `neutral`, `bright`.
- Wordings, in this order, with `{name}` replaced by the object label:

  1. `Pick up the {name} from the counter and place it in the sink.`
  2. `Put the {name} in the sink.`
  3. `Move the {name} off the counter into the sink.`
  4. `Take the {name} and drop it in the sink.`

  For `counter_to_counter`, replace the destination phrase with `onto the other counter` in all four. Exact strings for family `counter_to_counter`:

  1. `Pick up the {name} from the counter and place it onto the other counter.`
  2. `Put the {name} onto the other counter.`
  3. `Move the {name} off the counter onto the other counter.`
  4. `Take the {name} and drop it onto the other counter.`

An object is on a counter when its xy is inside that counter's top rectangle expanded by 0.0 m (no extra margin) and its z is within 0.25 m above the counter top. Counter top z is `position[2] + size[2] / 2`.

The proposer does not call a language model.

## Episode attempt

Internal, before export:

```json
{
  "attempt_index": 0,
  "success": true,
  "task": {},
  "fps": 20,
  "frames": [
    {
      "t": 0.0,
      "joint": [0, 0, 0, 0, 0, 0, 0],
      "gripper": 0.0,
      "eef": [1.0, 0.35, 1.15],
      "object_position": [1.0, 0.35, 0.97]
    }
  ]
}
```

`joint` length 7. `gripper` is 0 open and 1 closed. `eef` and `object_position` are meters, Z-up. These cartesian channels are how `sim_texture` draws. They are also written into a sidecar next to the meta files, `meta/arrival_cartesian.jsonl`, one JSON object per successful episode, with `episode_index` and `frames`; GR00T's loader ignores it.

### Scripted motion

80 frames, 20 fps, 4.0 s. Piecewise linear in cartesian space, 16 frames each:

1. Hover: eef from `[start.x, start.y, start.z + 0.18]` to `[start.x, start.y, start.z + 0.02]`. Gripper 0. Object stays at start.
2. Grasp: gripper linearly 0 → 1. Eef stays. Object stays.
3. Lift: eef and object z to `start.z + 0.20`. Gripper 1.
4. Carry: eef and object xy to the target point. z stays. Gripper 1.
5. Release: gripper 1 → 0 over the first 8 frames of this segment, then hold 0. Object stays at the release point. Eef rises 0.15 m.

Target point:

- `counter_to_sink`: sink `position` with z = sink top + 0.02. Sink top z = `position[2] + size[2] / 2`.
- `counter_to_counter`: `[counter_b.position[0], counter_b.position[1], counter_b top z + object.size[2] / 2]`.

Joints: store zeros in all 7 channels. Do not pretend they are Panda joint angles. `embodiment.json` records that joints are a placeholder.

Miss attempt (oracle drop): same trajectory but gripper stays 0 and the object never moves. `success` is false when, on the last frame, the horizontal distance from the object to the target point is greater than 0.05 m, or the gripper closed flag never exceeded 0.5. The miss rule is the distance test, not a special case in the oracle. The generator creates the miss by not moving the object.

Retarget: a seed is any task whose start pose is the scanned pose and whose lighting is `neutral` and whose wording is template 1. Other tasks copy that cartesian shape translated by `start_position - seed_start_position`. The target point is recomputed from the fixtures, not translated.

Sampling for a dataset of N successes: shuffle the cartesian product of the proposer's tasks with `random.Random(0)`, then cycle. Overlay the miss pattern on attempt indexes, not on task identity.

## LeRobot v2 / GR00T

Directory:

```
out/datasets/<room_id>/
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/modality.json
  meta/embodiment.json
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.ego_view/episode_000000.mp4
```

Episode indexes are 6 digits, starting at 0, with no gaps after drops. Dropped attempts do not occupy an episode index.

### modality.json

```json
{
  "state": {
    "single_arm": {"start": 0, "end": 7},
    "gripper": {"start": 7, "end": 8}
  },
  "action": {
    "single_arm": {"start": 0, "end": 7, "absolute": true},
    "gripper": {"start": 7, "end": 8, "absolute": true}
  },
  "video": {
    "ego_view": {"original_key": "observation.images.ego_view"}
  },
  "annotation": {
    "human.task_description": {}
  }
}
```

State and action are float32 length 8: 7 joints then gripper. Action at frame i equals the state at frame i (absolute joint targets). This matches the GR00T slicing rule in <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md>.

### embodiment.json

```json
{
  "embodiment": "robocasa_panda",
  "dof_arm": 7,
  "dof_gripper": 1,
  "control": "absolute_joint_position",
  "joints_are_placeholder": true,
  "cartesian_sidecar": "arrival_cartesian.jsonl",
  "note": "single_arm values are zeros until a physics generator fills them. Do not fine-tune on this dataset while joints_are_placeholder is true."
}
```

The sidecar path is relative to `meta/`.

### The sim dataset

`python -m arrival sim` writes the same layout to `out/datasets/<room_id>_sim/` with these differences, all of them visible in the meta files: `robot_type` and `embodiment` are `arrival_sim_panda`; `arrival_generator` is `mujoco_scripted_ik`, `arrival_physically_valid` is `true`, `arrival_appearance` is `mujoco_raster`, and `info.json` also carries `arrival_grasp` (`friction`), `arrival_success_check` (`physics`), `arrival_room_id` and `arrival_families`; `joints_are_placeholder` is `false` and the state holds the simulated joint positions, with the gripper channel the finger opening in metres; there are two videos, `observation.images.ego_view` and `observation.images.wrist`, at `--res` (default 128) square, both listed in `modality.json`; there is no cartesian sidecar; `episodes.jsonl` carries the per-episode `arrival_*` fields listed below.

### Parquet columns

| column | type |
|---|---|
| observation.state | list of 8 float32 |
| action | list of 8 float32 |
| timestamp | float32 seconds |
| annotation.human.task_description | int64 task index |
| task_index | int64 |
| episode_index | int64 |
| index | int64 global frame index |
| next.reward | float32, 0 except 1 on the last frame if the episode succeeded |
| next.done | bool, true only on the last frame |

One file per episode. `index` counts successful-episode frames only, starting at 0.

### tasks.jsonl and episodes.jsonl

```json
{"task_index": 0, "task": "Pick up the mug from the counter and place it in the sink."}
```

```json
{"episode_index": 0, "tasks": ["Pick up the mug from the counter and place it in the sink."], "length": 80}
```

Unique instruction strings get unique task indexes, assigned in first-seen order.

The `sim` dataset adds per-episode fields to `episodes.jsonl`: `arrival_family`, `arrival_lighting`, `arrival_people` (count), `arrival_success_detail`, `arrival_people_xy` (subsampled walker positions), `arrival_min_human_distance_m` (smallest horizontal clearance between a robot collision geom and a person over the episode, metres; `null` with no people) and `arrival_human_contact` (bool; always `false` in a written dataset because contact episodes are dropped before export).

### info.json

Include at least:

```json
{
  "codebase_version": "v2.1",
  "robot_type": "robocasa_panda",
  "fps": 20,
  "total_episodes": 1000,
  "total_frames": 80000,
  "total_tasks": 0,
  "arrival_generator": "scripted_retarget",
  "arrival_physically_valid": false,
  "arrival_successes": 1000,
  "arrival_attempts": 1111,
  "arrival_video_policy": "sparse",
  "arrival_appearance": "sim_texture",
  "features": {
    "observation.state": {"dtype": "float32", "shape": [8]},
    "action": {"dtype": "float32", "shape": [8]},
    "observation.images.ego_view": {"dtype": "video", "shape": [64, 64, 3]}
  }
}
```

`total_tasks` is the number of lines in `tasks.jsonl`. Fill it. Do not leave the example's 0 unless there really are zero tasks.

`features` may include the other scalar columns. Video shape is `[64, 64, 3]`, RGB.

### Video

64×64 RGB, 20 fps, 80 frames, H264 mp4 if `imageio-ffmpeg` can write it. Pixel rule for `sim_texture`:

- Background RGB from lighting: dim `(28, 32, 38)`, neutral `(70, 78, 86)`, bright `(210, 206, 196)`.
- Counter slab: a rectangle covering the lower 20 rows, color `(150, 140, 130)`.
- Object: a filled circle, radius 4 px, color `(180, 90, 70)`, at pixel `x = 32 + (object.x - room_center_x) * 8`, `y = 40 - (object.z - 0.9) * 40`. Clamp to the frame.
- End effector: a 3×3 square, color `(240, 240, 240)`, same projection using `eef`.

`room_center_x` is 2.0 for the hero kitchen. The projection is only a schematic. Tests check that frame 0 and frame 40 are not identical for a successful episode, and that a dim episode's corner pixel equals `(28, 32, 38)`.

## Job

`out/jobs/<id>/status.json`

```json
{
  "id": "job_01",
  "status": "queued",
  "created_unix": 0,
  "filename": "scan.json",
  "bytes": 0
}
```

`id` is `job_` plus 8 lowercase hex chars from `uuid.uuid4().hex[:8]`.

The worker (`python -m arrival work`) keeps those fields and adds these when it claims the job:

```json
{
  "status": "running",
  "stage": "sim",
  "started_at": 1792189828,
  "finished_at": null,
  "error": null,
  "room_id": "hero_kitchen_job_01587ed1",
  "twin_dir": "out/twins/hero_kitchen_job_01587ed1",
  "dataset_dir": "out/datasets/hero_kitchen_job_01587ed1_sim",
  "zip": "out/jobs/job_01587ed1/dataset.zip",
  "attempts": 2,
  "successes": 2
}
```

`status` is one of `queued`, `running`, `done`, `failed`. `stage` is the last stage entered: `normalize`, `twin`, `objects`, `sim`, `validate`, `zip`. `started_at` and `finished_at` are unix seconds like `created_unix`. `error` is null unless `status` is `failed`, then `"<ExceptionType>: <message>"`. `room_id` is the upload filename stem reduced to `[a-z0-9_]` plus `_<id>`. Paths are as the worker saw them (relative to its cwd when `--out` is relative). Fields not reached yet are null.

## Report

`out/reports/<room_id>.json`

```json
{
  "room_id": "hero_kitchen",
  "attempts": 1111,
  "successes": 1000,
  "dropped": 111,
  "seconds": 0.0,
  "demos_per_hour": 0.0,
  "tasks": ["counter_to_sink", "counter_to_counter"],
  "dollars": {
    "local_cpu": {"usd": 0, "measured": true},
    "token_factory": {"usd": null, "measured": false},
    "serverless_jobs": {"usd": null, "measured": false},
    "cosmos": {"usd": null, "measured": false},
    "object_storage": {"usd": null, "measured": false},
    "endpoint": {"usd": null, "measured": false}
  },
  "teleop_baseline": null,
  "usefulness": null,
  "realism": null
}
```

`demos_per_hour = successes / seconds * 3600`. `seconds` is wall time of the generate-and-export call.

The `sim` report (`out/reports/<room_id>_sim.json`) adds `human_contacts` (total and per family): attempts dropped because a robot geom reached a person. Those attempts count in `attempts` and `dropped`, never in `successes`.

## Eval

`out/eval/<room_id>_<policy>.json`, written by `python -m arrival eval`.

```json
{
  "room_id": "hero_kitchen",
  "policy": "scripted",
  "episodes_per_family": 4,
  "seed": 0,
  "people": 2,
  "holdout": false,
  "resolution": 128,
  "families": {
    "counter_to_sink": {
      "episodes": 4,
      "successes": 4,
      "success_rate": 1.0,
      "wilson_95": [0.5101, 1.0],
      "mean_seconds": 8.8,
      "mean_wall_seconds": 1.7,
      "human_contacts": 0,
      "min_human_distance_m": 0.222,
      "ik_error_max": 0.0123
    }
  },
  "totals": {
    "episodes": 4,
    "successes": 4,
    "success_rate": 1.0,
    "wilson_95": [0.5101, 1.0],
    "human_contacts": 0,
    "wall_seconds": 9.2
  }
}
```

`holdout: true` means every task used wording template 4 and the start offset `[0, +0.08]` where the family has them; `sim` datasets never contain those (the sim applies half the y offset, so the held-out start is +0.04 m in y). `success_rate = successes / episodes`, where an episode with a human contact is a failure whatever the physics check said. `wilson_95` is the Wilson score interval at z = 1.96. `mean_seconds` is sim time, frames / 20. `min_human_distance_m` is the minimum over episodes (`null` with `people` 0); a negative value is the depth of the worst overlap. `ik_error_max` is `null` for policies without an IK residual.

## Validator

`python -m arrival validate <dataset_dir>` exits 0 when:

- The four meta files plus `embodiment.json` exist.
- `modality.json` slices match the contract.
- Parquet columns exist and `observation.state` length is 8.
- Episode count equals `total_episodes`.
- Sum of episode lengths equals `total_frames`.
- If `arrival_video_policy` is `sparse`, at least 8 mp4s exist for a 1,000-episode dataset, and at least 1 mp4 exists for a dataset with fewer than 125 episodes.
- `arrival_physically_valid` is a bool.
- Exits 1, with one `INVALID: ...` line per problem on stderr, when a required file is missing or a slice is wrong.

It does not download GR00T.
