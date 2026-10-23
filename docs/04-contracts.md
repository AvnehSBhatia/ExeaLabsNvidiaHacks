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

Walls: treat each wall as a Y-up box whose center converts like an object. Project the box onto the floor and store `start` and `end` as the two endpoints of the long horizontal edge. `height` is `dimensions[1]`. `thickness` is the smaller of `dimensions[0]` and `dimensions[2]`. A wall whose `curve` has `startAngle`, `endAngle` and a positive `radius` is replaced by those chords: equal steps of 15° along the arc (at least two segments), in the wall's local XZ plane about `curve.center` when present (else the transform origin).

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

## RGB-D scan frames

`arrival/scan/rgbd.py` (CAP-2a) defines and validates a bundle of RGB-D frames. The iOS app records it during a RoomPlan scan (`FrameRecorder.swift`, CAP-2; compiled and its writer round-trip tested on a Mac, never run on a phone), and tests use a synthetic fixture. `POST /jobs` accepts a bundle as a zip in the `frames` field, validates it and stores it next to the job. The worker compares held-out frames with the compiled twin before adding synthetic objects (docs/REFERENCE.md, Uploaded-frame comparison).

Layout: `frames_manifest.json` plus the files it names, all relative paths inside the bundle folder (no absolute paths, no `..`).

```json
{"schema_version": 1, "room_id": "kitchen_01",
 "axes": {"camera": "arkit_x_right_y_up_z_backward", "room": "z_up_metres"},
 "arkit_world_to_room": [16 numbers, 4x4 row-major],
 "rgb_size": [w, h], "depth_size": [w, h], "depth_unit": "m",
 "frames": [{"id": "f000", "timestamp": 1.0, "rgb": "rgb/f000.png", "depth": "depth/f000.npy",
             "confidence": "conf/f000.png",
             "intrinsics": {"fx": 0, "fy": 0, "cx": 0, "cy": 0},
             "camera_to_room": [16 numbers, 4x4 row-major], "split": "train"}]}
```

- Axes: an ARKit camera is x right, y up, z backward (it looks down -z). The room frame is Z-up metres, the twin's frame. `arkit_world_to_room` records the transform from ARKit's Y-up world to the room frame; `camera_to_room` is already composed with it and maps camera points straight to the room.
- Pixel (u, v) is column, row, v down. A pixel at depth d is the camera point `((u-cx)/fx*d, -(v-cy)/fy*d, -d)`. Intrinsics are for the RGB image; depth intrinsics are scaled by `depth_size / rgb_size`.
- `depth_unit` is `m` (float32 `.npy`, metres) or `mm` (16-bit PNG, millimetres). 0 or non-finite is no reading. `confidence` is optional.
- `split` is `train` or `heldout` per frame, so an evaluation can use views the model never saw. An optional top-level `splits: {"train": [ids], "heldout": [ids]}` must agree with it; a frame in both is rejected.
- `load_manifest` rejects, each with its own `ManifestError.code`: `schema_version`, `axis_convention`, `depth_unit` (also a file extension that does not match), `non_monotonic_timestamps`, `duplicate_frame_id`, `nonfinite_transform`, `non_orthonormal_rotation` (tolerance 1e-3, or a reflection), `bad_matrix`, `bad_size`, `intrinsics`, `split`, `bad_path`, `missing_file`, `frame_in_both_splits`, plus structural codes `missing_manifest`, `bad_json`, `room_id`, `frames`, `frame_id`, `timestamp`; `load_depth` rejects `depth_shape`.
- Consent and sanitising: capture only rooms whose occupant agreed in writing that frames may be stored and shared. Before a bundle leaves the capturer's device, look through every RGB frame for people, screens, documents, photos and mail, and drop or blur those frames. Never commit real imagery; tests use `write_synthetic_bundle` (a flat floor and a wall, flat-colour images).

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

Object physics come from `arrival/catalog/objects.py`: shape, nominal size, mass at that size (scaled by volume for any other size), friction and colour. `PICKABLE` lists the ids the scripted expert holds by the body (mug, cup, sponge, apple, bottle, book, remote), `RIM_HELD` (bowl, plate) are hollow and pinched by the rim nearest the robot, `HANDLE_HELD` (pan) is pinched across the handle at its root, `THIN_HELD` (spatula) is pinched across its width with the pad tips 3 mm over the top it lies on; `GRASPABLE` is their union and what the pick families draw from. `WIPE_TOOLS` (sponge, spatula) are what `wipe_counter` drags. The closed laptop is a scene object only (it is wider than the gripper opening both ways). `room_kind` reads a twin's scanned categories (kitchen, bedroom, living room or office) and `ROOM_OBJECTS` lists what `place_default_objects` puts on a table or counter for each: the kitchen's mug, bowl, plate, cup and apple, an office's laptop, book and cup, a living room's remote, book and cup, a bedroom's book and cup; outside a kitchen, on a table deeper than 0.8 m, they lie within 0.40 m of a long edge. The plate is a deep plate, 3 cm tall with an 18 mm rim, because the closed pads leave a 12 mm gap and a thinner rim slips out. `place_default_objects` draws each default object's size within 8% of the nominal size from its seed.

Moving parts in the sim (`arrival.sim.scene.part_names(twin)`): each is a joint, with a site `<joint>_handle` on its handle bar, in the parent fixture's frame.

| fixture type | joints |
|---|---|
| counter | `<id>_drawer0`, `<id>_drawer1` (slide, 0 to 0.4 m), `<id>_door` (hinge, 0 to 1.9 rad), `<id>_upper_door<k>` (hinge; one wall-cabinet door per 0.6 m of counter run, none over tables, islands or counters under 0.8 m) |
| sink inside a counter | `<sink_id>_faucet` (hinge, 0 to 1.2 rad; the lever up is water on) |
| stove | `<id>_oven_door` (hinge about the bottom edge, 0 to 1.5 rad; closed is held by gravity) |
| fridge_bottom_freezer | `<id>_door` (hinge, 0 to 2.0 rad). Optional fixture field `hinge`: `"left"` (default, and what a RoomPlan scan always gets) or `"right"`, which mirrors the door in the fixture's x (hinge on the right edge, free edge on the left, opening the same way: q > 0 is open). Left output is byte-identical to before the field |
| dishwasher | `<id>_door` (hinge about the bottom edge, 0 to 1.5 rad), `<id>_rack` (slide, 0 to 0.45 m: the upper rack rolls out over the open door; the carcass is hollow so it can); floor-standing; a counter run whose end overlaps it is trimmed back, like a fridge or stove (an appliance wholly inside a run is not cut out) |
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

Wall `pos` is the segment midpoint at half height, so the wall box is centred on the scanned line (it is not shifted toward the interior by half the thickness): `[ (start.x+end.x)/2, (start.y+end.y)/2, height/2 ]`. Wall `size` is `[length, thickness, height]` for walls whose long axis is X, and `[thickness, length, height]` for walls whose long axis is Y. Floor `size` is `[width, depth, 0.02]` from the wall bounding box, `pos` at the bbox center with z `0`. When the twin has `floors` with corners, the bounding box of those corners is used instead.

Counter synthesis (only when the scan has a sink or stove whose centre is more than 0.3 m from every counter footprint and whose own footprint is more than 0.05 m from every counter, so a stove built in between two counters gets none): emit a counter of size `[max(fixture.size[0] + 1.2, 1.8), 0.65, 0.92]`, centered on the fixture's x,y, with z `0.46`. The authored sample already has counters, so this branch is covered by a unit test with a sink-only bundle, not by the hero scan.

Do not emit backing walls.

The twin copies `doors`, `windows`, `openings`, `floors`, `story` and `story_count` from the bundle when present, unchanged, and a fixture flagged as an island carries `island: true`; that flag is also written on its `layout.yaml` row. Doors, windows and openings are not written to `layout.yaml`; the sim draws them, as below.

In the sim (`arrival.sim.scene.portals_xml`) each door, window and opening is a set of visual-only geoms on the world body (`contype="0" conaffinity="0" group="2"`), so it shows in renders and never touches anything. They are named `door<k>_<part>`, `window<k>_<part>` and `opening<k>_<part>`, with `k` the entry's index in the twin's list. The wall is the one `wall_id` names, else the wall whose segment passes nearest the centre, within 0.25 m (exports from before iOS 17 have no `parentIdentifier`); with no such wall the entry is not drawn. The scanned rectangle (`size[0]` wide and `size[2]` tall around `position`) is put on the wall's centre line and clipped to its length and height; a door or opening whose bottom is within 5 cm of the floor starts on it, and one narrower or lower than 15 cm after that is not drawn. A 5 cm frame sits inside the rectangle: `jamb0` (at the end nearer the wall's `start`), `jamb1` and `head`, plus a `sill` under a window and under an opening that starts above the floor. Inside the frame a closed door (`is_open` false or absent) has a `leaf` and a `knob` (a bar through the wall 7 cm in from `jamb1`, 1.0 m up or at half the leaf's height if that is lower), an open door or an opening has a dark `recess` (RoomPlan gives no hinge side, so an open leaf is not drawn), and a window has a `pane` and a centre `mullion`. Every box runs through the wall, so it shows on both faces, and stands proud of each face of the drawn wall (at least 5 cm thick): 2 mm for a recess, 4 mm for a pane, 6 mm for a leaf, 1.5 cm for the frame, 3 cm for a window sill; the knob reaches 6 cm. The wall itself stays solid. A twin with no doors, windows or openings builds the same MJCF as before.

Each object also has `label`, `provenance` (`detected`, `authored` or `default_synthetic`; docs/REFERENCE.md, Provenance) and, when the pipeline put it there, `placed_by` (`place_default_objects` or `detect_objects`). A detected object has `detection`: `{"confidence", "frames": [frame ids], "evidence": [{"frame", "box": [x0, y0, x1, y1] in RGB pixels, "confidence"}]}`. A twin whose objects the worker filled has `object_detection`: `{"state": "detected" | "none_found" | "not_run", "reason", "frames_read", "frames_unread", "reply_sources": {source: frames}, "model", "rejected": [{"frame", "label", "box", "confidence", "reason"}]}`. `reply_sources` says where each frame's reply came from (`token_factory`, or `hand_written` in test fixtures) or why the frame was not read (`not_configured`, `unreachable`, `call_failed`, `unusable_reply`). A rejection `reason` is `unsupported_label`, `low_confidence`, `bad_box`, `no_depth`, `no_support` (no counter or table under the box or the merged object), `nothing_above_top`, `too_wide`, `not_standing` (the points do not rise from the top as the object would), `flat_surface` (a wall or backsplash seen face on), `off_top` (the object's footprint would hang off the top), `against_wall`, `on_fixture` (it would stand in another fixture, or in a sink or the strip behind it where the faucet stands; the last three also apply to a merged object's averaged position) or `same_object` (a weaker box on an object the same frame already found).

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
- Lighting: `dim`, `neutral`, `bright`. In the sim the label sets the light level (0.35, 0.7, 1.0 of nominal) and the task `seed` jitters it within 15% plus colour temperature, key-light position and which side the window light comes from, so the three labels stay ordered on frame brightness for every seed.
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

The `sim` dataset (`<room_id>_sim`, written by `python -m arrival sim`) has a third slice: `"base": {"start": 8, "end": 11}` in `state` and the same with `"absolute": true` in `action`, so its state and action are length 11. The slice is the base x, y (meters) and yaw (radians) in the room frame; the action is the commanded base pose. With the default fixed base the three values hold the parked pose for the whole episode; with `--base mobile` they move at up to 0.3 m/s and 0.5 rad/s. One schema for both, so one GR00T modality config lists all three slices. `embodiment.json` carries `"dof_base": 3`, `"base": "planar"`, `"base_frame": "room"` and `"base_actuated": true|false`.

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

The `sim` dataset adds per-episode fields to `episodes.jsonl`: `arrival_family`, `arrival_lighting`, `arrival_people` (count), `arrival_success_detail`, `arrival_people_xy` (subsampled walker positions), `arrival_min_human_distance_m` (smallest horizontal clearance between a robot collision geom and a person over the episode, metres; `null` with no people) and `arrival_human_contact` (bool; always `false` in a written dataset because contact episodes are dropped before export). With `--planner vlm`, `arrival_planned_by` says what actually planned the episode: `token_factory`, `cache` or `rule_based` (the fallback with no key or a failed call); `planner` in the provenance and report is only what was asked for.

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
  "successes": 2,
  "job_attempts": 1
}
```

`status` is one of `queued`, `running`, `done`, `failed`. `stage` is the last stage entered: `normalize`, `twin`, `realism`, `objects`, `sim`, `validate`, `zip`. `started_at` and `finished_at` are unix seconds like `created_unix`. `error` is null unless `status` is `failed`, then `"<ExceptionType>: <message>"`. `room_id` is the upload filename stem reduced to `[a-z0-9_]` plus `_<id>`. Paths are as the worker saw them (relative to its cwd when `--out` is relative). Fields not reached yet are null.

`job_attempts` is how many times a worker has started this job; it is incremented when the job is claimed and is absent until then. A failed attempt that may pass is written back as `queued` (never `failed`) with `error` null, and the attempt's error is appended to `retry_errors` (a list of `"<ExceptionType>: <message>"`, oldest first). After the last allowed retry, `status` is `failed`, `error` is that last attempt, and `job_attempts` is the number of starts. `ARRIVAL_JOB_RETRIES` (or `arrival work --retries`) is how many times a failed attempt is queued again; default 2, `0` means never. A bad upload (`ValueError`, including unreadable JSON and `ScanBundleError`), a request over a limit, `unsupported_layout`, a dataset with nothing to ship, or the wall-clock timeout is not retried.

The job status also has `realism`, initially null, then `{state, metrics, consumed, preview, reason}`. Scored results add `frame_count` and `association`. `metrics` is null unless the camera-offset control passes; otherwise it contains `depth_l1_m`, `rgb_mae`, `rgb_one_minus_ssim`, `n_depth`, `n_rgb` and `frames`. These fields describe the uploaded-frame comparison only, not the generated dataset. Raw reports, the comparison PNG, a pre-placement twin and the applied camera transform stay under the job's `realism/` directory. The public report route returns only the status projection. Normalized RoomPlan bundles carry `arkit_world_to_room` as a row-major 4x4 matrix, used as `bundle_world_to_room @ inverse(manifest_world_to_room)` before scoring.

The job status also has `objects`, null until the `objects` stage, then `{"counts": {"detected", "authored", "default_synthetic"}, "detection", "reason", "reply_sources", "rejected"}`. `detection` is the twin's `object_detection.state`, or `not_run` with the reason `the scan lists its objects`; `rejected` is the number of rejected boxes, which are listed in `twin.json`.

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

The `sim` report (`out/reports/<room_id>_sim.json`) adds `human_contacts` (total and per family): attempts dropped because a robot geom reached a person. Those attempts count in `attempts` and `dropped`, never in `successes`. It also adds `fridge_visits` (attempts in which a walker opened the fridge) and `door_contacts` (attempts in which a robot geom touched the door while a person held it open; those are counted in `human_contacts` too). The dataset's per-episode fields do not change.

## Eval

`out/eval/<run_id>/report.json`, written by `python -m arrival eval`. Each invocation creates a new directory. The summary fields below are retained; the report also includes `run_id`, `checkpoint_id`, `checkpoint_id_source`, `manifest_id`, `twin_sha256`, `manifest_path`, `episodes_path`, `runtime`, `base` and `omitted_families`.

`manifest.json` has `schema_version: 1`, the full `twin`, its `twin_sha256`, ordered `tasks`, `omitted_families` and `settings` (families, episodes_per_family, seed, people, holdout, resolution, base, fps, max_steps). `manifest_id` is the SHA-256 of sorted compact UTF-8 JSON with that ID omitted; nonfinite values are refused. The twin hash uses the same encoding. Replay validates and copies the manifest before any rollout.

`episodes.jsonl` has one row per attempted trial: `run_id`, `checkpoint_id`, `manifest_id`, `trial_index`, `task_id`, `family`, `physics_success`, `success`, `seconds`, `wall_seconds`, `safety`, `ik_error`, `detail`, `error`. Physics success is null on an exception; scored success is false. Simulation time and safety fields are null on exceptions. Family and total `observed_episodes` counts identify attempts with measurements; safety aggregates and mean simulation time use only those attempts and are null when none were observed. Success rates still count all attempts, including errors. Completed reports aggregate these rows. A run interrupted before completion keeps its manifest and recorded attempts without a summary report.

```json
{
  "room_id": "hero_kitchen",
  "policy": "scripted",
  "episodes_per_family": 4,
  "seed": 0,
  "people": 2,
  "holdout": false,
  "resolution": 128,
  "base": "fixed",
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

`holdout: true` means every task used wording template 4 and the start offset `[0, +0.08]` where the family has them; `sim` datasets never contain those (the sim applies half the y offset, so the held-out start is +0.04 m in y). `success_rate = successes / episodes`, where an episode with a human contact is a failure whatever the physics check said. `wilson_95` is the Wilson score interval at z = 1.96. `mean_seconds` is sim time, frames / 20. `min_human_distance_m` is the minimum over episodes (`null` with `people` 0); a negative value is the depth of the worst overlap. `ik_error_max` is `null` for policies without an IK residual. `base` is `fixed` or `mobile` (`--base`), and the `sim` report carries the same key.

## Validator

`python -m arrival validate <dataset_dir>` exits 0 when:

- The four meta files plus `embodiment.json` exist.
- `modality.json` slices match the contract.
- Parquet columns exist and `observation.state` length matches `info.features` (8 for `demo`, 11 for `sim`).
- Episode count equals `total_episodes`.
- Sum of episode lengths equals `total_frames`.
- If `arrival_video_policy` is `sparse`, at least 8 mp4s exist for a 1,000-episode dataset, and at least 1 mp4 exists for a dataset with fewer than 125 episodes.
- `arrival_physically_valid` is a bool.
- Exits 1, with one `INVALID: ...` line per problem on stderr, when a required file is missing or a slice is wrong.

It does not download GR00T.
