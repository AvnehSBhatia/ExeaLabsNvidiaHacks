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

## Synchronized ego and exocentric capture

The existing MuJoCo pipeline can export three views of each observation: robot-base `ego_view`, hand-mounted
`wrist`, and the fixed room camera `overview` (exocentric). This is one physics rollout, not independently
recorded episodes: every camera is rendered from the same simulation state before the action is applied.
Frame `k` in all three MP4s corresponds to parquet row `k`, timestamp `k / 20` seconds within that episode.
The added camera does not change the expert, success checks, actions, or the two original camera streams.

```bash
python -m arrival sim --room hero_kitchen --families open_drawer --episodes 2 --people 1 --res 128 --seed 3 --exocentric --save-preview --out out/multiview
python -m arrival validate out/multiview/datasets/hero_kitchen_sim
```

`--exocentric` is opt-in; without it the original two-camera dataset is written. `--save-preview` also writes
`out/multiview/previews/hero_kitchen_sim_ego_exo.mp4` and a middle-frame `.png`: labeled side-by-side views with
one shared frame number and timestamp. These pairs use the exact observation frames supplied to the dataset
writer; labels are added only to the human preview. The existing overview movie and contact sheet remain available.
As with existing previews, the first task must reach rollout for a preview to be written; it may show a failed attempt.

The external video is `videos/chunk-000/observation.images.overview/episode_000000.mp4` inside the dataset.
It has the same dimensions, episode length, frame rate and indexing as ego and wrist; `meta/info.json` and
`meta/modality.json` declare all three. `arrival_camera_capture` records the camera roles and `pre_action` capture
phase. The writer rejects a missing, short, wrong-size or non-uint8 camera array before writing episode files.
The collection report and provenance record the chosen cameras. `--resume` refuses a different camera set,
including enabling the external view on a previous two-camera run. Use the same flag for every shard;
`merge-shards` retains all three cameras and rejects mixed schemas.

For a fixed-base GR00T experiment, use [arrival_panda_multiview_config.py](../configs/gr00t/arrival_panda_multiview_config.py)
instead of the default config. It requests all three cameras at observation index zero and keeps the existing
arm/gripper state and action semantics. The default config continues to use ego and wrist, so it can still read
these datasets while ignoring the extra view. The current policy-server wire format also continues to send ego
and wrist only; this contribution does not serve a three-camera model.

This is physics-simulated, scripted-expert data, not a trained policy or real robot demonstration. The overview
uses the existing room-wide camera; framing and occlusion depend on the room. No three-view GR00T training,
policy-quality improvement, GPU run or cloud cost has been measured. Test and preview evidence is in
[tests/test_multiview.py](../tests/test_multiview.py) and [the paired example](media/ego-exo.png).

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
python -m arrival serve --host 0.0.0.0   # reachable from the iOS capture app on the same Wi-Fi; prints the LAN URL
python -m arrival serve --host 0.0.0.0 --public --behind-proxy --max-upload-mb 100 --rate-limit 5 --max-queued 20   # hosted, behind a TLS proxy
```

Check the machine before a first run (offline, exit 1 only when something required is missing):

```bash
python -m arrival doctor            # python, core packages, mujoco, ffmpeg, optional keys, disk
python -m arrival doctor --json     # the same checks as JSON
```

`doctor` reports Python (3.11 or newer), numpy, pyarrow, pyyaml, pillow, imageio and imageio-ffmpeg (a missing one fails), MuJoCo (a warning when absent: `sim`, `eval` and `work` need it, `demo` and `serve` do not), an ffmpeg binary that runs (the one imageio-ffmpeg ships, else `PATH`), free disk where `--out` will be written (warning under 2 GiB, failure under 0.25 GiB or when not writable), and whether `TOKEN_FACTORY_API_KEY`, `TAVILY_API_KEY`, `GR00T_POLICY_URL` and `COSMOS_URL` are set, in the environment or in a `.env` in the cwd or repo root. A key shows only as `set in the environment`, `set in .env` or `not set`; its value is never printed, logged or sent anywhere, and a missing optional key is never a failure.

| Route | What it does |
|---|---|
| `GET /` | Lists rooms under `scans/` with `geometry_source`, task counts per family, dataset path, and download link. Has an upload form |
| `GET /rooms/<id>` | Fixtures, the two task families, `arrival_physically_valid`, and the run report |
| `GET /download/<id>` | Zip of `out/datasets/<id>` |
| `POST /jobs` | Multipart upload: file field `scan` (a `.json` RoomPlan export or `scan.json`) and an optional `frames` (a `.zip` RGB-D bundle, see below). Body max 200 MB, larger gets 413 (`ARRIVAL_MAX_UPLOAD_MB` or `--max-upload-mb` lowers it). With `ARRIVAL_MAX_QUEUE` queued or running jobs already (default 20) it gets 429 before the body is read. Writes `out/jobs/<id>/` and returns `202 {"status":"queued",...}`. Nothing is compiled or generated in the request; `python -m arrival work` drains the queue |
| `GET /jobs/<id>` | The job's `status.json` (without the owner hash; in `--public` mode also without server paths) |
| `DELETE /jobs/<id>` | Only for the browser that uploaded the job (the `arrival_client` cookie hash must match the job's `owner`; any other caller, no cookie, or an unknown id gets the same 404). A finished job (`done`, `failed`, `cancelled`) is deleted with its twin, dataset and report: 200 `{"result":"deleted"}`. A queued job is cancelled: 200 `{"result":"cancelled"}`. A running one is asked to stop: 202 `{"result":"cancelling"}`. The jobs table shows Cancel/Delete buttons on the caller's own rows |
| `GET /jobs/<id>/download` | `out/jobs/<id>/dataset.zip` once the job is `done`, else 404 with the current status |
| `GET /previews/<file>` | A png, mp4 or gif from `out/previews/` (what `sim --save-preview` writes), with `Range` support so browsers can seek video |

The index page shows a card per room (stats and the latest preview), a drag-and-drop upload with a progress bar, and every job under `out/jobs/` with its status, stage bar, room id, episode count and a download link; running jobs refresh live. The server binds `127.0.0.1` by default and has no authentication. `--host 0.0.0.0` opens it to the local network for the capture app (`ios/README.md`); do that only on a network you trust. Uploads are streamed to a temp file and downloads are streamed too; a body shorter than `Content-Length`, a missing closing boundary, an empty file, or a part with no filename gets 400. If `out/datasets/<room>_sim` exists, the room page links it and shows its `arrival_generator`, `arrival_physically_valid`, episode count and families.

### Web hardening (L4)

Job ids are `job_` plus 32 random hex characters (128 bits), so an id is a capability and cannot be enumerated; every route that takes an id checks it against a strict pattern and answers 404 otherwise (no path is built from anything else). What an upload must pass, in order, each with its own status:

| Check | Answer |
|---|---|
| more than `--rate-limit` uploads in a minute from one client (default 10, `0` = off) | 429 with `Retry-After` |
| `Content-Length` above `--max-upload-mb` (default 200) | 413 |
| not `multipart/form-data`, or no or an overlong boundary | 400 |
| `--max-queued` jobs (default 50, `0` = no cap) already queued or running | 503 with `Retry-After` |
| free disk under twice the body plus 100 MB | 507 |
| 4 uploads or dataset zips already in flight | 503 with `Retry-After` |
| body not finished within 10 minutes, or shorter than `Content-Length` | 408 / 400 |
| `scan` missing, empty, not named `*.json`, or not starting with `{` | 400 / 415 |
| `frames` not a `*.zip` starting with the zip magic, or failing the bundle checks below | 415 / 400 / 413 / 422 |

The stored name is a sanitised copy of the client's filename (basename only, `A-Za-z0-9._() -`, at most 80 characters); the bytes are always written to `upload.bin`, so a client name never becomes a path. The scan is sniffed, not parsed: a `.json` file that starts with `{` but is not a valid scan is accepted here and fails at the worker's `normalize` stage.

Every response carries `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy`, `Cross-Origin-Opener-Policy` and `Cross-Origin-Resource-Policy`, `Cache-Control: no-store` (previews: 30 s private), and a `Content-Security-Policy` that allows the page's one inline script and one inline stylesheet by sha256 hash, no `unsafe-inline`, no external origins. Errors (404, 405, 413, 429, 500, a malformed request line) are short generic pages or `{"error": ...}` JSON, never a traceback or a Python version; the `Server` header is just `Arrival`. The server writes no access log; protocol errors and crashes go to stderr with environment secrets, bearer tokens and `api_key=` pairs masked (`arrival.web.hardening.redact`).

`--public` is for a hosted page: `GET /` lists only the jobs the visitor's own browser uploaded (an `arrival_client` cookie holds a random token, only its hash is stored in the job), and `GET /jobs/<id>` drops `twin_dir`, `dataset_dir` and `zip`. Without it (the default on localhost) the page lists every job, as before. `--behind-proxy` trusts `X-Forwarded-Proto` (adds `Strict-Transport-Security` and marks the cookie `Secure` when it says `https`) and the last entry of `X-Forwarded-For` (the rate-limit key); never set it when clients can reach the port directly, because they could then forge both. Unchanged limits: there is still no login and no TLS in this server, a job id (or a room id, for `/download/<room>`) is the only secret, `/rooms/*` and `/previews/*` are readable by anyone who can reach the port, and one process serves everything. Put TLS and authentication in front before exposing it to the internet.

### RGB-D frames in an upload (L14)

Send the CAP-2a bundle (docs/04-contracts.md) as a second multipart field `frames`: a `.zip` with `frames_manifest.json` at its root and the files it names (`.png`, `.jpg`, `.npy`; no other types). The server unpacks it into a scratch folder (rejecting `..`, absolute or backslash paths, symlinks, encrypted or duplicate entries, more than 4000 entries, a member over 64 MiB or a bundle over 1 GiB unpacked, measured while streaming rather than from the zip header), runs `arrival.scan.rgbd.load_manifest`, and opens every RGB image, depth array (`.npy` without pickles, or 16-bit PNG) and confidence image to check that it is readable and matches `rgb_size` / `depth_size`. A bundle that fails returns 422 with the loader's `code` (for example `non_monotonic_timestamps`, `missing_file`, `depth_shape`, `bad_size`, `bad_image`) and nothing is queued; a bad archive returns 400. A good one is moved to `out/jobs/<id>/frames/` and the status gains `"frames": {"count", "train", "heldout", "depth_unit", "rgb_size", "depth_size", "stored": true, "consumed": false}`. The worker compares held-out frames against the compiled twin before adding default objects. The `scan` still carries the geometry; appearance reconstruction is separate work, and the train frames are read for objects (Objects from frames, below). Frames under `out/` are gitignored; never commit private frames.

From a RoomPlan export instead of the authored scan (`scans/<room>/roomplan.json`):

```bash
python -m arrival demo --room hero_kitchen --from-roomplan --episodes 4
```

The normalizer unwraps Apple's keyed enums (`{"sink": {}}` -> `sink`, `washerDryer` -> `washer_dryer`), drops `low`-confidence objects, puts the floor at z = 0 (from `floors[]`, else the lowest wall bottom), rotates the room so its longest wall runs along +X with the wall bounding box at (0, 0), and snaps near-90° yaws. RoomPlan captures no small objects, so a mug, a bowl, a plate, a cup and an apple are placed (seeded, each at a seeded size within 8% of its catalog size; one that does not fit the counter is skipped) on the counter nearest a sink; nothing is borrowed from the authored scan. A scan with no sink, stove or counter-like fixture yields no tasks and the command says so. `--scans` defaults to `./scans`, falling back to the repo's `scans/` when run from elsewhere.

Doors, windows and openings are kept in the bundle and twin with the id of their wall (the sim draws them on it, visual only), `polygonCorners` floors set the floor extent, a wall that carries `curve` (`startAngle`, `endAngle`, `radius`) becomes straight chord segments every 15° (at least two), and a merged multi-story export yields one story (`roomplan.load(path, room, story=n)`; default the story with the most objects). To eyeball every room as a top-down plan:

```bash
python scripts/render_rooms.py                              # out/renders/<room>_top.png and <room>_roomplan_top.png
python scripts/render_rooms.py tests/fixtures/roomplan/amal.json
```

### Uploaded-frame comparison

After processing, the jobs table links to `/jobs/<id>/realism`, with a comparison image and depth/RGB errors. `/jobs/<id>/realism/report` returns the safe result fields; `/jobs/<id>/realism/preview` serves the image. In public mode these routes require the upload owner's cookie. They never serve raw capture files or local artifact paths.

The worker uses the held-out-frame driver from L16, with its confidence mask, bounded frame sampling and camera-offset control. It saves the twin before synthetic objects are placed. For raw RoomPlan uploads, the normalizer records its ARKit-world transform and camera poses are mapped from the manifest's room frame into that normalized frame. For normalized scan bundles without that transform, poses must already use the scan's coordinates. A named scan and frames must have the same source room ID; raw RoomPlan captures are associated by being uploaded together. Neither check proves calibration or room identity.

`realism.state` is `measured`, `invalid`, `not_measured` or `error`, with `metrics: null` unless the depth control passes. A missing bundle, train-only bundle, mismatched room or blocked claim remains unmeasured. A comparison error is recorded locally and does not discard a valid simulation dataset. `frames.consumed` becomes true when scoring finishes, including an invalid control result; `used_for` is then `realism`. A retry clears these fields and hides old previews until fresh results exist.

This is a schematic room comparison. Depth error measures geometry, while RGB error also measures lighting and surface differences. Use empty-room captures with centered intrinsics and square pixels supported by the renderer. Synthetic fixtures exercise upload, alignment, masks, scoring and display; they do not establish real-room realism.

### Objects from frames (C4)

When the scan lists no objects and the upload has RGB-D frames, the worker's `objects` stage runs `arrival.twin.detect.detect_objects` before it falls back to default objects. It reads up to 12 frames of the `train` split (held-out frames stay for the comparison above), asks Nemotron Omni (`NEMOTRON_OMNI_MODEL`, through the Token Factory client the planner uses) for the catalog objects in each frame as a label, a pixel box and a confidence, and back-projects each box with the frame's depth, intrinsics and pose (`rgbd.unproject`, mapped into the twin the way the comparison maps it).

For each counter or table, the box's points over its footprint at object height, less those within 3 cm of a wall's face in the twin and those on the back 2.5 cm of a counter or RoomPlan storage worktop where its backsplash stands (not a table or an island), are split into groups connected on a 2 cm grid. A group counts only if it could be the object standing on that top: no wider than the catalog object plus 4 cm, its lowest point near the top, some of the object's height, and depth in x and y, which a wall seen face on lacks. The group whose centre projects nearest the box centre is the object. Its x and y are moved away from the camera by pi/4 of the object's radius because only the near side of a solid object is seen (a bowl or plate is not moved), and z puts the catalog-size object on the top. Its footprint must lie over the top, clear of walls, and clear of other fixtures, a sink and the 10 cm behind it where the faucet stands included. Two boxes of one label in one frame that land on one spot are one object. A hit joins the nearest object of its label within 0.10 m seen in other frames, never one from its own frame, and the merged position must pass the footprint check again, since the mean of two clear spots can land on a sink's corner. Each object is labelled `detected`, with its confidence, frames and boxes under `detection`. Every box that is not placed goes into `twin.json` `object_detection.rejected` with its reason (docs/04-contracts.md, Twin).

Without `TOKEN_FACTORY_API_KEY`, or when the endpoint is unreachable or a reply is unusable, the frame is not read: `object_detection.state` is `not_run` with the reason, nothing is guessed, and `place_default_objects` fills the counter as before, labelled `default_synthetic`. A run that read frames and found nothing is `none_found` and falls back the same way. The job status `objects` block carries the per-label counts and the detection state and reason, and the jobs table shows them in a chip (`objects: 1 detected`, or `objects: 5 default_synthetic (invented, not scanned)`).

Replies are cached by a hash of the image (`DetectionClient(cache_path)`); the worker keeps no cache. The tests use `tests/fixtures/detection/`: three 128 x 96 renders of hero_kitchen's twin with one mug and the same three views without it, plus `replies.json`, which is hand-written, not recorded. The model has never been called for this feature. On the mug fixture the merged mug lands 1.31 cm from its true centre in x and y and 0 cm off in z (the tests allow 3 cm and 1 cm), the same with the boxes padded by 0, 2, 3, 4, 8, 20 or 40 px. In rendered probes, 609 empty-room boxes over 29 regions (faucet, sink, cooktop, fronts, wall and fridge corners, a corner table) place nothing; 96 of 112 counter runs over 8 spots and 7 labels land within 3 cm, and 30 of 40 table runs within 1.8 cm (the others are explained in docs/BUILD-LOG.md).

A follow-up added tests for a mug, a bottle and a bowl 1 and 3 cm in front of a backsplash, the east wall, their corner and a storage worktop's backsplash (all within 3 cm), and for an empty storage backsplash (nothing placed). The sweep over all three kitchens gives the same result before and after it: 1239 empty-room runs place nothing, 291 of 294 counter runs and 30 of 40 table runs land within 3 cm. The wall strip is measured from the twin's wall, which the scene draws at least 5 cm thick around the same centre line, so against a rendered wall it reaches 1.5 cm in front of the face, and `against_wall` lets an object overlap the drawn wall by up to 2.5 cm.

Limits, measured in those probes: two mugs touching along the view give one object and miss the other; two same-label objects 9 cm apart that no frame shows together merge; nothing is placed over a sink or within 10 cm behind it, where the faucet stands; a wrong label of similar shape is placed at the wrong height (a bottle boxed as a mug sits 6 cm low). These numbers measure back-projection and merging on clean rendered depth with boxes taken from the renders, not the model's boxes or real ARKit depth.

## One-command demo

```bash
make demo                              # same as: python scripts/demo_all.py --quick --out out
python scripts/demo_all.py --quick     # 2 episodes, 1 person, 64 px; --full is 8, 2, 128
python scripts/demo_all.py --room hero_kitchen --out out/demo --serve
python scripts/demo_all.py --quick --timeout 300     # abort with exit 1 after 300 s (default 900, 0 = none)
```

Runs scan and compile, twin layout PNG, sim dataset, validate, scripted-expert eval and contact sheet, prints PASS or FAIL per stage, and writes `<out>/demo_summary.md`, `demo_summary.json` and a self-contained `<out>/index.html` (stage table plus the contact sheet inlined; open it from disk, no server). See `docs/ref/DEM-4.md`.

It needs no network and no keys. `TOKEN_FACTORY_*`, `TAVILY_*`, `GR00T_POLICY_URL` and `COSMOS_URL` are hidden from the run, and every connection to a non-loopback address is refused for its duration. Before running it picks a MuJoCo GL backend by rendering a 16 px frame in a child process, trying the platform default first, then `egl`, `osmesa` and `glfw` on Linux (`glfw` on macOS, `glfw` then `egl` on Windows); an exported `MUJOCO_GL` is respected and tried alone. Exit codes: 0 all stages passed, 1 a stage failed or `--timeout` expired, 2 the machine cannot run it (mujoco not installed: `pip install -e ".[sim]"`, or no working OpenGL backend, with one line saying what was tried). Measured on an Apple Silicon laptop: about 20 s and 0.5 GB peak for `--quick`; the install itself needs a package index once.

## Repo checks (CI)

```bash
make lint            # ruff: syntax errors and undefined names only
make docs-check      # scripts/check_docs.py and scripts/check_links.py
make package-smoke   # python scripts/smoke_package.py [--offline] [--keep DIR] [--build-python PY]
```

`scripts/check_links.py [--root DIR]` checks every inline Markdown link and image in every tracked Markdown file: a relative target must exist and a `#fragment` must match a heading in the file it points at. External links are not fetched. `scripts/smoke_package.py` builds the sdist and wheel with setuptools, installs the wheel in a fresh venv (`--offline` skips the venv and borrows this interpreter's packages), and from an empty directory imports the installed copy, runs `--help`, runs `demo` against the checkout's `scans/` with `--scans`, and validates the result. The wheel carries only the `arrival` package, so an installed copy needs `--scans`; the last check confirms the failure without it is one clear line. CI (`.github/workflows/ci.yml`) runs tests, the demo and validate on ubuntu, macOS and Windows with Python 3.11 and 3.12 (ubuntu gates; the other two are reported only), the three checks above, a Docker build, and uploads `demo_summary.md`, `demo_summary.json` and `index.html` from each matrix cell as a build artifact. Dependabot (`.github/dependabot.yml`) opens weekly update PRs for Actions, pip and Docker. See `CONTRIBUTING.md`.
To land a new real scan (CAP-3a), `scripts/intake_scan.py` creates `scans/<room_id>/`, strips EXIF from the photos, writes `notes.md` and `measurements.json`, runs the geometry checks and prints GO or NO-GO (exit 0, 1, or 2 when an input is refused):

```bash
python scripts/intake_scan.py kitchen_01 --roomplan roomplan.json --photos a.jpg b.jpg --location-type kitchen \
    --phone "iPhone 15 Pro" --ios 18.1 --lighting daylight --clutter low \
    --measure counter_length=3.20,counter_height=0.92,sink_width=0.55   # also --no-photos, --scans DIR, --out DIR
```

`arrival.scan.intake.validate_scan_dir(path)` lists what is still missing. Details in `docs/ref/CAP-3a.md`.
To check the twin against tape measurements, put a `measurements.json` beside a room's `scan.json` (format and extraction rules in `docs/ref/TWN-1a.md`) and run `python scripts/scan_accuracy.py [scans/<room> ...]`: it prints the signed error per dimension and exits 1 when any is over `tolerance_m` (default 0.05 m); rooms without the file are skipped.

## Static export of the page

```bash
python -m arrival export-site                              # out/static_site/index.html, no server needed
python -m arrival export-site --out out --dest out/static_site --zips
```

Crawls the real page (`arrival serve`'s handler on an ephemeral 127.0.0.1 port) and writes plain files: `index.html`, `rooms/<id>/index.html`, `previews/` (the png, mp4 and gif files the pages reference), `data/<room>.json` plus `data/index.json` (the numbers the page shows; anything missing on disk is `null`) and `.nojekyll`. Links are rewritten to relative file links so the folder works from `file://` and under a GitHub Pages project prefix. The upload form, the jobs table and the page script are replaced by a note, because no backend takes a scan. Dataset zips are included only with `--zips`; otherwise the download buttons are removed. The export is link-checked (every relative link and asset must resolve inside the folder) and nothing is written if a link is broken. Rooms with no dataset under `--out` still export and say "not generated"; run `python -m arrival demo` or `sim` first. It does not touch the repo-root `site/` directory. Not done: a Pages workflow, and CORS headers on the API for a hosted live-upload page.

## Multi-room summary

```bash
python -m arrival summary                                     # hero_kitchen + the two real RoomPlan fixtures
python -m arrival summary --rooms hero_kitchen,tests/fixtures/roomplan/amal.json --episodes 2 --out out/summary
```

Runs normalize, compile, tasks, sim and validate (structural, then per episode) on each room with tiny settings (default 2 episode attempts, 64 px, 1 person; work files under `out/summary`, not `out/`) and writes `summary.md` and `summary.json` there. A failing stage records its reason and the later stages read `-` (not run); a room with no reachable task reads `skipped`. `--rooms` takes scan names under `--scans` or paths to a RoomPlan or `scan.json` file. The tape-error column is filled only where a `measurements.json` exists (hero_kitchen's is authored from its own scan and is labelled so), and realism is always "not measured": it needs held-out RGB-D frames that no room here has. RoomPlan exports hold no small objects, so their objects are `default_synthetic` (invented) and the table says so.

## Simulation (MuJoCo)

`python -m arrival sim --episodes N` builds a MuJoCo scene from the compiled twin and records episodes to `out/datasets/<room>_sim` (LeRobot v2 layout). Needs the `sim` extra. The code lives in `arrival/sim/`:

- **Articulated kitchen** from the twin: counters with drawers (slide joints), cabinet and wall-cabinet doors (hinge joints), a sink with a faucet lever, a stove with a bottom-hinged oven door, and fridge, dishwasher and microwave doors. `arrival.sim.scene.part_names(twin)` lists every joint with its handle site and an open position. Objects collide with the moving parts, so a mug rests on a drawer tray or an open oven door.
- **Arm**: a procedurally built 7-DoF Panda-like arm driven through IK, so joint states in the dataset are the simulated joint positions, plus a parallel gripper that holds objects by friction contact. `--arm menagerie` (on `sim` and `eval`) swaps in the MuJoCo Menagerie Franka Panda (Apache-2.0): run `python scripts/fetch_menagerie.py` once to put its 34 MB of meshes in the git-ignored folder arrival/assets/menagerie, which is gitignored and never part of a commit or a CI run; without the folder the command exits 2 and prints that line, and the Menagerie tests skip. Same joint, site and camera names, same position actuators, and it works with `--base mobile`; the real hand opens 8 cm instead of 10 cm and its pads are smaller, so a few plans carry per-arm numbers (`robot.HAND` / `menagerie.HAND`, `preflight.ARM_STANDOFF_TRIM`). `embodiment.json` and the reports record `arm`; `close_door` on the hero kitchen's east counter (`counter_b`, 5 cm from the wall) is the one family that fails with it, because the real hand body meets the wall (see the L25 paragraph in `docs/BUILD-LOG.md`).
- **Base**: fixed per episode by default. `--base mobile` (on `sim` and `eval`; scripted policy only) puts the pedestal on two slide joints and a hinge with position actuators, axes aligned to the room whatever the parked yaw. The expert drives it at up to 0.3 m/s and 0.5 rad/s, only with the gripper closed on an object, and waits while a person is within 0.7 m in front of it. Either way the base pose (x, y, yaw in the room frame) is `state[8:11]` and `action[8:11]`, and `embodiment.json` says whether it was actuated (`base_actuated`). Only `counter_to_counter` drives; every other family parks once. Why 11 floats and what it means for training: [State and action width](#state-and-action-width-8-vs-11-floats).
- **Lighting and materials**: a key spot with shadows, a cool fill and a warm window light, scaled by the task's `lighting` label and jittered (level within 15%, colour temperature, key position, window side) by the task seed, so every episode in a label looks a little different and the labels never overlap in brightness. Surfaces use MuJoCo builtin textures tinted by the geom colour (tile floor, wood fronts, stone tops, backsplash boards, matte walls), so the scene still loads from a string with no asset files.
- **People**: kinematic proportioned bodies (seeded height and colours) with a leg and arm swing, walking seeded loops through the room: a free point, a lane that crosses behind the robot base, a 2 to 4 s stop at the counter, back out, every leg routed round the fixture footprints. The loop is laid out in the frame of the fixture the task works at, so it holds on any wall, and a room with no free strip in front of that fixture (the office scan) gets people on random free points and no counter stop (`sim` logs that per episode); the walk into a stop slows over its last 0.6 m so the swinging toe never reaches the counter. They move and occlude; they are not simulated humans and they do not collide. `arrival/sim/safety.py` measures the horizontal clearance between every robot collision geom and each person every control step; an episode whose minimum reaches 0 is a human contact and is dropped from the dataset even when the task succeeded. The value is written per episode as `arrival_min_human_distance_m` and `arrival_human_contact` in `meta/episodes.jsonl`.
- **Cameras**: ego and wrist views rendered with MuJoCo's offscreen renderer. `--res` (default 128) must be a multiple of 16; the writer rejects anything else before writing a file.
- **Policies** behind one interface: a scripted expert (default), a VLM planner via Nebius Token Factory (`--planner vlm`; needs `TOKEN_FACTORY_API_KEY`, falls back to the rule parser without one, and caches responses in `arrival/models/nemotron_cache.json`), and a VLA client for a GR00T policy server (placeholder; no server is shipped or run here).
- **Success is checked in physics** (object resting in the target region, drawer/door joint past threshold), not asserted by the script.

Run it:

```bash
python -m pip install -e ".[dev,sim]"
python -m arrival sim --episodes 8 --people 2 --save-preview   # contact sheet + mp4 in out/previews/
python -m arrival sim --base mobile --families counter_to_counter --episodes 4   # the pedestal drives (MAN-3)
python scripts/fetch_menagerie.py && python -m arrival sim --arm menagerie --episodes 4   # the real Franka asset (L25)
python -m arrival sim --episodes 1000 --shard 0/8 --resume     # block 0 of 8 -> out/shards/0of8/ (see Sharding below)
python -m arrival merge-shards out --out out/merged            # join the shards, then validate the result
python -m arrival validate out/datasets/hero_kitchen_sim
python scripts/check_dataset_loader.py out/datasets/hero_kitchen_sim [--json]
```

### Per-episode quality check

`validate` says whether the folder is a dataset; `--episodes` says whether each episode is worth training on, and `--drop` writes a cleaned copy:

```bash
python -m arrival validate out/datasets/hero_kitchen_sim --episodes            # exit 1 if any episode is flagged
python -m arrival validate out/datasets/hero_kitchen_sim --episodes --frames --json out/quality.json
python -m arrival validate out/datasets/hero_kitchen_sim --drop out/datasets/hero_kitchen_sim_clean
```

Flags, with the thresholds in `arrival/export/quality.py` (all echoed in the `--json` report): `unreadable` (parquet missing or the wrong row count), `nonfinite` (NaN or inf in state or action), `zero_motion` (state moves less than 1e-3 over the episode), `stuck_arm` (the state is frozen for more than 40 consecutive steps while the command is more than 0.15 rad away), `joint_at_limit` (within 3% of a joint's range for more than 40 consecutive steps; `arrival_sim_panda` datasets only, since other datasets carry placeholder joints), `length_outlier` (more than 2.5x off the median of its task family, families of 4 or more), and with `--frames` `blank_video` (more than half the frames of a camera are near-uniform). Not checked, because the dataset does not record it: carried-object tilt, and what the wrist camera actually sees (only uniformity is measured). `--drop` re-indexes the kept episodes 0..K-1, recomputes `episodes_stats.jsonl` and `stats.json`, renames the videos, writes `meta/dropped_episodes.json` (old index to new index, and the reasons), refuses an output folder that already exists or sits inside the source, and re-runs the structural check on the result. `arrival_attempts` is left as collected. `--drop` implies `--episodes`; with either, a structural error no longer stops the run, so a dataset with a NaN episode still gets its per-episode report.

`scripts/check_dataset_loader.py` is an offline stand-in for the GR00T loader (no `gr00t` import, details in `docs/ref/DAT-2a.md`). It exits 0 on pass and 1 on any failure, and refuses the placeholder `demo` dataset.

Status as of Oct 17 (scripted expert, friction grasps, physics-checked; 16/16 over these seven families with `--families all --episodes 16 --people 2 --seed 1` on a fresh clone):

| Family | Success | Note |
|---|---|---|
| `counter_to_sink` | 100% | mug, cup, apple and bottle by the body; bowl and plate by the rim; pan by its handle; spatula pinched across its width with the pad tips 3 mm over the counter (L26) |
| `open_drawer` | 100% | both counters |
| `open_door` | 100% | both cabinet doors, 8/8 over two seeds; the door ends at about 70° (threshold 60°). A run whose hinge end is drawn at least 5 mm into a wall, with the bar ending the pull within 0.06 m of it (the hero kitchen's `counter_b`), is pulled from 0.12 m further out, which keeps the wrist off that wall |
| `counter_to_counter` | 100% with `--base mobile`, skipped on the fixed base | pick point and target about 2 m apart: the base parks a step toward the target, grasps, tucks the object in front of itself above the counter top, drives along the counter front and places; 6/6 with two people (seed 1, 0 contacts, closest 0.11 m); also runs on the u_kitchen_42 run that faces -y |
| `close_drawer` | 100% | starts open 0.30 m, pushed shut on the handle bar; success is the slide joint under 0.03 m |
| `close_door` | 100% | starts open 70°; the closed fingers push the handle bar along its arc from the side to 40°, then the hand goes back to its start pose and pushes the door shut from in front; where the run ends flush against a wall and the open door's handle starts within 0.13 m of it, the base stands 6 cm nearer the hinge so the wrist clears that wall; success is the hinge under 5° |
| `put_in_drawer` | 100% | drawer starts open 0.14 m, mug set on the exposed part of the tray; success is the mug resting inside the tray |
| `wipe_counter` | 100% | sponge or spatula pinched and dragged four strokes (the sponge pressed 4 mm into the top, the spatula held at the pinch height); success needs 0.4 m travelled pressed to the top and the tool back on it |
| `open_fridge` | 100% | 32/32 over two seeds, with and without people; the fingers close round the vertical bar and pull the door to 40°, then the closed fingertips push the inside face to 80° (threshold 60°). Not a default family |
| `open_oven` | 100% | hero kitchen, 4/4 per seed pair; the hand comes down on the bar pitched 1.35 rad below horizontal and keeps that orientation while it pulls the door to 0.5 rad (the round bar turns between the pads), then lets go and gravity takes it to its 1.5 rad stop (threshold 1.0 rad). The base stands in front of the bar, 6 cm beyond where the open door's top edge lands |

| `open_oven` | 100% | hero kitchen, 4/4 per seed pair; the hand points into the panel, the expert pulls the door to 0.5 rad and lets go, gravity takes it to its 1.5 rad stop (threshold 1.0 rad). The base stands 0.25 m along the run from the bar on whichever side passes the stand check (+x first, then -x, else straight in front), still in front of the door, and 6 cm beyond where the open door's top edge lands; straight in front of the bar, joint 5 sat on its stop for 71 steps |
| `open_dishwasher` | 100% | only in rooms with a dishwasher (the hero scan has none); same plan as the oven with a tilted hand picked by IK residual |
| `turn_on_faucet` | 100% | the closed fingers push the lever tip up from below (lever to about 1.1 rad, threshold 0.6 rad), from 34 cm in front of the counter run the sink is set into |
| `load_dishwasher` | 100% | only in rooms with a dishwasher and a pickable object on the adjoining counter within reach of a stand spot beside it; the door starts flat and the upper rack rolled out 0.45 m; success is the object resting in the rack tray |
| `unload_dishwasher` | 100% | same layout, the object starts in the rack and ends on the counter where it started |
| `move_on_table` | 100% on the office scan | only in rooms with a scanned table: the robot stands at the table edge nearer the object (within 0.44 m of it), outside furniture and walls, and moves the object 0.35 m along the table (toward the middle, else the other way when that spot is taken by another object); success is the object resting on the top within 6 cm of the target. Cup, book and remote each pass on the office table; the laptop is scenery. Kitchens have no table, so it is skipped there |

Defaults are `counter_to_sink,open_drawer,move_on_table`. `--families all` runs every family in `arrival.sim.tasks.SIM_FAMILIES` (pick and place, drawer and door open and close, put in drawer, wipe, open fridge, move on table, open oven, open dishwasher, turn on faucet, load and unload the dishwasher); `counter_to_counter` is skipped with a log line on the fixed base and runs with `--base mobile` (MAN-3), and any failed attempt is dropped and counted. With `--base mobile` the pre-flight also checks the drive: the place point against the drive spot's reach and the whole straight line between the two stand spots against walls and fixtures (`drive path`), and `make_tasks(mobile=True)` lets the target be any counter of the same run, not only one within 1 m of the object. Fixtures may stand on any wall: the base parks in the fixture's own frame, and before a task is built `arrival/sim/preflight.py` checks that its stand spot is inside the room, 0.15 m from every wall face and fixture, and, for the families whose hand targets come from the twin (pick and place, move on table, put in drawer, wipe, load and unload the dishwasher), that the arm reaches the expert's own waypoints from it: `arrival/sim/reach.py` solves `ik.solve` on an arm-only model of the chosen `--arm` (joint limits, no collisions) and allows 2.5 cm of position error where the hand touches and 5 cm where it only hovers (1 cm for `put_in_drawer`). The stand rules in `base_pose` stay the first choice; only when the rule's spot fails does the planner try a fallback grid of spots up to 0.2 m along the fixture's run and from 0.06 m closer to 0.12 m further out from its face, nearest first, and the variation keeps the first that passes at every start offset as `base_shift` (logged as `stands ... from the usual spot`, counted as `shifted` in the pre-flight dict). The stand check also keeps the base 0.13 m clear of a bottom-hinged door lying open or a pulled-out drawer the task uses; hinged doors' swing is not checked. Two stand rules keep the arm off its joint stops and off a wall: `open_oven` stands 0.25 m along the run from the bar (`OVEN_ALONG`, the +x side first, then -x, then straight in front, the first spot that passes the stand check), and `open_door` stands 0.12 m further out when the run's hinge end is drawn at least 5 mm into a wall and the bar ends the pull within 0.06 m of it, and only when that further spot passes the stand check (`OPEN_DOOR_WALL_OUT`). Door, drawer, fridge, flap-door and faucet families get the stand checks only. A task that passes can still miss in physics. A rejected variation is logged with its reason and counted as `skipped_unreachable`; a task whose env build or rollout raises is logged (`SKIP at build|rollout: ...`), counted as `skipped_error`, and writes no episode. Skipped tasks are not attempts: `attempts` counts tasks that ran to the physics check and `dropped` those that failed it. The report (`out/reports/<room>_sim.json`) and `meta/info.json` carry `skipped`, `skipped_unreachable`, `skipped_error` and `skip_reasons` (at most 50; `arrival_` prefixed in `info.json`), `sim` prints them to stderr, and the worker status has `skipped`; when no task at all can run, the job fails with `error_kind: unsupported_layout` and the reasons in `error_reasons`, not a traceback. `--families` accepts every name in `SIM_FAMILIES` (`arrival/sim/tasks.py`) even though its `--help` text lists only four. `--planner vlm` sends each instruction through the Nemotron planner (POL-1), `--instructions nemotron --split train|eval` draws generated wordings with a held-out set (POL-2). `--tavily` adds appliance research to the twin and the planner prompt (see [Tavily research](#tavily-research)). The VLA client has never run against a real GR00T server; `python -m arrival.policy_server.mock_server --port 8000` is an offline stand-in that proves the plumbing only (see [ref/POL-3.md](ref/POL-3.md)).

### Sharding and resume

`--episodes N --shard i/n` (i counts from 0) runs block `[i*N//n, (i+1)*N//n)` of the N tasks `make_tasks` draws. Every shard builds the whole task list from the same `--seed`, so the n shards together run exactly the tasks one process would run, each with the seed and `task_id` (`sim_<family>_<global index>`) it has there, and `--episodes` is the total, not per shard. Output goes to `out/shards/<i>of<n>/` (`datasets/<room>_sim`, `reports/`, `progress/`), and `meta/info.json` carries `arrival_shard` (index, count, task range and the settings that must agree). `python -m arrival merge-shards <folder> [--out out]` takes the shard folders (or `out`, or `out/shards`, which hold them), refuses a missing, duplicated, unfinished or mismatched shard (different seed, families, res, base and so on), then writes `<out>/datasets/<room>_sim` in shard order with episode, task and frame indices and the stats rebuilt, copies the mp4s, writes a summed `reports/<room>_sim.json` (`seconds` is compute time summed over shards, `seconds_longest_shard` the wall time if they ran in parallel) and runs the validator (exit 1 if it fails, 2 for an unusable shard set). Pre-flight rejections are counted once, not once per shard. On a laptop, 2 shards of 2 episodes merge to the same episodes, frames, tasks, task ids and stats as one run of 4 (`tests/test_shards.py`); this was not run across machines.

Every `sim` run writes a journal, `<out>/progress/<room>_sim.jsonl`, one line per finished task, appended after that task's episode files. `--resume` reads it, skips the finished tasks, deletes the files of a task a kill interrupted, and continues, so a killed shard ends with the dataset an uninterrupted run would have written (checked in the same test file). A journal from different settings (room, families, episodes, seed, people, policy, planner, split, base, res, shard) is refused with the differing names rather than mixed in; without `--resume` the old run is overwritten. Datasets now put episode `i` in `chunk-{i // 1000:03d}` like the validator expects, so a merged run of more than 1,000 episodes is a multi-chunk dataset (before, the sim writer put every episode in `chunk-000` and `validate` rejected more than 1,000).

### Fan-out to Nebius (C1)

`python -m arrival fanout [--backend local|nebius] --run-id r1 --shards 4 --episodes 32 [sim settings] [--out out]` splits a sim run into `--shard i/n` jobs (above), runs every shard that has no done-marker, submits a failed shard again up to 3 times, and merges the shards once every marker exists (`arrival/jobs/fanout.py`). A marker (`<run>/done/<i>of<n>.json`) is written only after the shard's files are in storage, so a rerun with the same `--run-id` skips finished shards; a run id first used with other settings is refused. `--backend local`, the default, runs shards as subprocesses and keeps storage in `out/fanout_store`.

Each shard also gets a status record, `<run>/status/<i>of<n>.json`, in the same storage as `<run>/run.json` (`arrival/jobs/fanout_status.py`). It is rewritten when an attempt starts, changes state or ends, and a rerun appends to it. Per attempt it holds `job_id`, `states` (each state change with its unix time), `submitted_at`, `finished_at`, `wall_seconds` (the orchestrator's clock from submit until the marker is written or the failure is seen, so provisioning and the copy are included), `resources` (platform, preset, disk and job timeout on Nebius; `null` for a local subprocess), `outcome` (`running`, `succeeded`, `failed`, or `abandoned` for an attempt an earlier process left open), and for a failed attempt `failed_stage` (`submit`, `run`, `fetch` or `marker`) and `reason` (the job's state and message, or the error, redacted). Nothing is priced from it.

`--backend nebius` (`arrival/jobs/nebius.py`) runs each shard as a Nebius Serverless AI Job and keeps the run in one Object Storage bucket. It needs no SDK: the job calls are the REST API on `https://api.nebius.cloud` with a bearer token (`POST /ai/v1/jobs`, `GET /ai/v1/jobs/{id}`, `POST /ai/v1/jobs/cancel`), and the bucket calls are path-style S3 (`PUT`, `GET`, `HEAD`, `ListObjectsV2`) on `https://storage.<region>.nebius.cloud`, signed with Signature Version 4. Each job runs the arrival image with `python -m arrival.jobs.nebius run-shard --dest <run>/attempts/<i>of<n>/<tag> -- <sim args>`, which simulates the shard on the job's own disk and uploads the shard folder under that staging prefix. The orchestrator then copies the staged files to `<run>/shards/<i>of<n>/` and writes the marker last, so a failed or half-uploaded attempt never reaches the shard prefix. Every attempt is a fresh machine, so a retried shard starts over instead of resuming its progress journal. Job states map to running (`STATE_UNSPECIFIED`, `PROVISIONING`, `STARTING`, `IMAGE_PULLING`, `RUNNING`), succeeded (`COMPLETED`) and failed (`FAILED`, `CANCELLED`, `ERROR`, `CANCELLING`, `DELETING`); any other state stops the run with an error that names it. When the command ends (finished, failed or Ctrl+C) it cancels the jobs it started that are still open; jobs left by a killed process have to be cancelled by hand. A job that has not finished `NEBIUS_MAX_JOB_SECONDS` (default 7200) after it was created, or has reported no state for `NEBIUS_MAX_NO_STATE_SECONDS` (default 1800) in a row, is cancelled and counts as a failed attempt, so it is retried up to the attempt limit. The job spec already asks the service to stop the container at 1 h; the extra hour is for provisioning and the image pull, which that timeout may not count, so the limit only fires on a job the service failed to end, and a value not above the job timeout is refused. Defaults: platform `cpu-d3`, preset `4vcpu-16gb`, a 64 GiB network SSD, a 1 h job timeout.

Setup, once: build the repo `Dockerfile` and push it where the project can pull it (`NEBIUS_JOB_IMAGE`); create a bucket (`NEBIUS_BUCKET`) and an access key pair for a service account that can write to it (`NEBIUS_ACCESS_KEY_ID`, `NEBIUS_SECRET_ACCESS_KEY`); put the same pair in a MysteryBox secret with the payload keys `NEBIUS_ACCESS_KEY_ID` and `NEBIUS_SECRET_ACCESS_KEY` (`NEBIUS_JOB_S3_SECRET_ID`), so the key never appears in a job spec; set `NEBIUS_PROJECT_ID` and an access token from `nebius iam get-access-token` (`NEBIUS_IAM_TOKEN`). The token lasts 12 h, which caps how long one run can poll. A missing setting stops the command before any request, naming the variables and never their values.

It is tested only against in-process fakes (`tests/fake_nebius.py`, `tests/test_nebius.py`): a job API that checks the token and the documented request fields and runs each job's command as a local subprocess, and an S3 server that verifies every request's signature and pages listings 3 keys at a time. The signer matches two cases of the AWS signing test suite. With one injected failed shard, and after a kill and a rerun, the Nebius backend merges to the same episode rows and frames as an uninterrupted local fan-out, with no duplicate episode ids. A job stuck in `RUNNING` and a job that never reports a state are each cancelled and retried, and the shard fails after the attempt limit with the stage and reason in its record (on a fake clock, so the default limits are what is tested). A run killed with a shard open keeps that attempt in the record, and the rerun marks it `abandoned` and appends the next. Neither limit has been checked against the real service's timing. It has never run against Nebius: no job has been created and no bucket touched, so job name rules, quotas, the image pull, the enum spelling on the wire, MysteryBox lookup and how the service splits `args` are unchecked. Docs relied on, read 2026-10-21: [job create](https://docs.nebius.com/rest-api/ai/v1/jobs/create), [job get and states](https://docs.nebius.com/rest-api/ai/v1/jobs/get), [job cancel](https://docs.nebius.com/rest-api/ai/v1/jobs/cancel), [managing jobs](https://docs.nebius.com/serverless/jobs/manage), [`nebius ai job create`](https://docs.nebius.com/cli/reference/ai/job/create), [automatic restarts](https://docs.nebius.com/serverless/jobs/auto-restarts), [REST authentication](https://docs.nebius.com/rest-api/authentication), [platforms and presets](https://docs.nebius.com/compute/virtual-machines/types), [Object Storage URLs](https://docs.nebius.com/object-storage/overview), [access keys](https://docs.nebius.com/iam/service-accounts/access-keys), [S3 compatibility](https://docs.nebius.com/object-storage/interfaces/s3-api-compatibility), and the [AWS signing test suite](https://github.com/awslabs/aws-c-auth/tree/main/tests/aws-signing-test-suite/v4).

### State and action width: 8 vs 11 floats

The decision (L13, issue #18): **every `sim` dataset has 11-float state and action rows**, `[0:7]` arm joints, `[7:8]` gripper, `[8:11]` base x, y (m) and yaw (rad) in the room frame; the action's base part is the commanded pose. The fixed base writes its parked pose there on every frame (constant) and `embodiment.json` has `base_actuated: false`. The `demo` placeholder dataset stays 8 floats (the loader check refuses it anyway), and `sim` datasets written before this change are 8 floats and would have to be regenerated to be trained together with new ones.

Why 11 for both modes and not 8 for the fixed base: one dataset layout and one `modality.json` shape for fixed and mobile data, so a fixed-base and a mobile run can be mixed or compared without a converter. What it costs: three columns that carry no information on the fixed base. What that means for training: the 8-float view is not lost. `arrival_panda_config.py` lists only `single_arm` and `gripper`, so a fixed-base fine-tune reads `[0:8]` of the 11-float rows and never sees the constant base columns; only `arrival_panda_mobile_config.py` adds `base`. Nothing here has been run through Isaac-GR00T's own loader (G1), so whether it accepts a modality config that skips part of a row is untested. If it does not, the fallback is an 8-float writer for the fixed base: `STATE_DIM` and the base slice in `arrival/sim/export.py`, plus the width the expert pads to in `run_episode`.

`python scripts/check_dataset_loader.py <dataset>` understands the base slice: it requires `[8, 11)` in both state and action with 11-float rows and a boolean `embodiment.base_actuated`, errors when a base that is declared fixed moves (`base_fixed_moves`) or a mobile one moves faster than 0.3 m/s or 0.5 rad/s (`base_speed`, a teleport), and warns once, as `fixed_base_constant`, that a parked base's columns are constant and the arm-only config is the one to train with.

### Closed-loop eval

```bash
python -m arrival eval --policy scripted --families counter_to_sink,open_drawer --episodes 4 --people 2
python -m arrival eval --holdout            # wording template 4 and the [0, +0.08] start offset (docs/07-eval.md)
python -m arrival eval --policy vla --checkpoint-id base-gr00t  # needs GR00T_POLICY_URL; exits 2 with a message when unset or unreachable
```

`eval` runs `--episodes` episodes per family through the same `SimEnv` and `run_episode` as `sim`, writes no dataset, prints a table, and writes `out/eval/<run_id>/report.json` (shape in [04-contracts.md](04-contracts.md)). Per family: episodes, successes, success rate with a Wilson 95% interval, mean sim seconds (frames / 20), mean wall seconds, human contacts, the smallest robot-human distance seen, the largest IK residual, and arm-fixture contacts (`arm_contact_episodes`, `arm_contact_max_depth_m`). A human contact counts as a failure; an arm-fixture contact is counted, not scored. Every family ends through the same `finish_task` as `sim`, so the close and put families are covered, and an episode that raises is recorded as a failure instead of stopping the run. The eval is sim-in-the-twin only; nothing here scores a real robot.

`python -m arrival.generate.retarget` records one scripted-expert seed and replays it for new object start poses (`--attempts`, `--family counter_to_sink|counter_to_counter`, `--res`, `--seed`, `--baseline` to time the scripted expert on the same count). Successes (physics oracle) are written to `out/datasets/<room>_seed_replay/` with `arrival_generator = "seed_replay_retarget"` and a seed reference; unreachable or failed replays are dropped. Details in [ref/DAT-4a.md](ref/DAT-4a.md).

What this is not: MuJoCo frames are not a Gaussian splat of the room, episodes are not MimicGen output, and no policy was trained on them. Read `arrival_generator` and `arrival_physically_valid` in the sim dataset's `meta/info.json` before using it.

### Realism: twin render vs real frames

```bash
python -m arrival realism scans/my_kitchen            # a folder with frames_manifest.json + roomplan.json (or scan.json)
python -m arrival realism my_kitchen --scans scans --max-frames 4 --max-side 128
python -m arrival realism scans/my_kitchen --twin out/twins/my_kitchen/twin.json --min-confidence 0
```

`realism` scores the twin's MuJoCo render against the scan's **held-out** RGB-D frames and writes `out/eval/<room>_realism.json` plus a side-by-side PNG (`<room>_realism.png`: reference RGB, twin RGB, reference depth, twin depth, one row per frame, `--png-frames` rows; the per-frame PNGs sit next to it). It is a driver over `arrival/eval/realism.py`; the frames come from `frames_manifest.json` ([ref/CAP-2a.md](ref/CAP-2a.md)), in ARKit camera axes, which the command converts to the renderer's before scoring. Only frames the manifest marks `heldout` are scored (at most `--max-frames`, evenly spaced); the JSON lists the ids it scored and how many `train` frames it left out, and a manifest with no held-out frame is refused. A pixel counts for depth only if its reference depth is finite and above 0 and its ARKit confidence is at least `--min-confidence` (default 2, high; frames without a confidence map keep every non-zero depth). The twin render hides the robot and people and the recorder films the empty room, so there is no robot mask. Frames larger than `--max-side` (default 192) are downscaled, depth by nearest neighbour.

The JSON also carries a misaligned-pose control (same frames, camera moved 0.15 m and turned 5 degrees) and an `alignment_check`. If depth error does not rise under the control, the poses or the twin's frame are wrong: the report says `valid: false` with the reason, and the command exits 1 so the number is not quoted. A scan without `frames_manifest.json` (RoomPlan only, or four photos) is refused with exit 1 and nothing is written; a four-photo bundle that carries frames reports `realism: null` with the reason from `blocks_claims`. Exit 2 means MuJoCo is not installed. The `scope` field repeats what the number is: the twin is a schematic untextured box render, so `depth_l1_m` measures geometry and the RGB metrics mostly measure missing texture and lighting. Nothing in this repository has run it on a real scan yet: the iOS RGB-D recorder (P1) has not been field-tested, and the tests use frames rendered from the twin itself.

### Provenance

`twin.json` carries a `provenance` block: `input.file_sha256` (the scan file as received: `scan.json`, the RoomPlan export, or the upload's `upload.bin`), `input.bundle_sha256` (the normalized bundle), `code.sha` and `code.dirty` (the git checkout the code ran from; `ARRIVAL_GIT_SHA` overrides it for an image with no `.git`, and both are `null` when neither exists), `generators` (the geometry source, the twin compiler, and how many objects carry each label below) and `seeds` (the default-object placement seed). Each twin object has a `provenance` label: `authored` (a person wrote it into `scan.json`), `default_synthetic` (`place_default_objects` put a catalog object on a counter because RoomPlan sees no small objects) or `detected` (`arrival.twin.detect` found it in the upload's RGB-D frames; see Objects from frames). A twin the worker filled also has `object_detection`, which says whether detection ran and why the defaults were used when they were. Every dataset's `meta/info.json` has `arrival_provenance`: the twin's input hash, generators and seeds plus this run's code sha, seed, generator and options. These fields say where data came from; they do not change `arrival_physically_valid` or any other claim.

## Upload worker

`python -m arrival work` runs in a second terminal next to `serve`. It polls `out/jobs/*/status.json` every `--poll` seconds (default 2), takes the oldest `queued` job, marks it `running`, and runs these stages, rewriting `status.json` at each one: `normalize` (the upload is a RoomPlan CapturedRoom export if it has `walls`/`objects` with `transform`, or a `scan.json` if it has `schema_version`; anything else fails here), `twin` (`compile_twin`), `realism` (optional held-out RGB-D comparison), `objects` (when the scan has no objects, as RoomPlan exports never do: the objects detected in the upload's RGB-D frames if any, else `place_default_objects` with the objects that suit the kind of room: an office gets a laptop, a book and a cup; then `out/twins/<room>/`), `sim` (`arrival.sim.collect.collect`, same options as `sim`: `--episodes 8`, `--families counter_to_sink,open_drawer,move_on_table`, `--people 1`, `--res 128`, `--seed 0`), `validate`, and `zip` (`out/jobs/<id>/dataset.zip` holding `<room>_sim/`). The room id is the upload's filename stem in `[a-z0-9_]` plus the job id, so two uploads of `kitchen.json` never share a twin or dataset folder. A job with zero successful episodes or a failing validator ends `failed`. A transient failure (disk, memory, an unexpected error in the twin, renderer or sim, a child that died) is queued again up to `--retries` times (env `ARRIVAL_JOB_RETRIES`, default 2; `0` = never); a bad upload, a request over a limit, a room the sim cannot use or the wall-clock timeout is not. `status.json` counts starts in `job_attempts` and keeps earlier errors in `retry_errors`; `error` stays null while the job will run again. Permanent failures are written to `status.json` (`error`) and `error.txt` (traceback); the worker then moves to the next job. A job that raises mid-stage is recorded as `failed` with an `interrupted:` error. `status.json` is written atomically, so polls never see a torn file. `--once` processes at most one job and exits. Needs the `sim` extra; without MuJoCo every job fails at `sim`.

Limits, concurrency, retention and metering (`arrival/jobs/limits.py`, `worker.py`, `retention.py`):

- **Limits.** Each job runs in a child process (`python -m arrival.jobs.child`) that the worker kills after `--max-job-seconds` (env `ARRIVAL_MAX_JOB_SECONDS`, default 1800; `0` runs jobs in-process with no timeout). A killed job ends `failed` with `TimeoutError: job exceeded the N s wall-clock limit during stage 'sim'`; a child that dies on its own ends `failed` with `WorkerProcessError`. `--max-episodes` (env `ARRIVAL_MAX_EPISODES`, default 100) and `--max-people` (env `ARRIVAL_MAX_PEOPLE`, default 4) are ceilings on the worker's own `--episodes` and `--people`: above them `arrival work` exits 2 before touching a job. An `upload.bin` larger than the upload limit fails at `normalize`. `ARRIVAL_MAX_QUEUE` (default 20, `0` = unlimited) caps queued plus running jobs; the count is not atomic with the write, so simultaneous uploads can overshoot it by a few. There is no memory cap: a job that exhausts RAM is killed by the OS and recorded as `WorkerProcessError`.
- **Several workers.** Any number of `python -m arrival work` processes may share one `out/`. A worker claims a job by creating `out/jobs/<id>/claim.lock` with `O_CREAT|O_EXCL`, touches that file every 10 s while the job runs (the heartbeat) and deletes it at the end. A `running` job whose heartbeat is older than 120 s, or whose lock was written by a dead process on the same host, goes back to `queued` (`requeues` counts it; after 2 requeues it fails with `WorkerLost`); every worker checks for this at start and then periodically, so a restart no longer waits on or requires stopping other workers. Claiming relies on `O_EXCL` working on the filesystem under `out/`: do not share `out/` over a network filesystem that does not guarantee it. Cancel is `arrival.jobs.worker.cancel_job(out/jobs, id)`: a queued job becomes `cancelled` at once, a running one is killed (child process) or stops at its next stage boundary; `DELETE /jobs/<id>` calls it for the job's owner. While the `sim` stage runs, `status.json` carries `progress` (`{"done", "total"}`, one per finished task line from `collect`, total = requested episodes, so skipped tasks count as done) and the jobs table shows `sim: 3/8 episodes`; there is no finer progress inside one episode. `ARRIVAL_MAX_JOB_MB` (default unset = off) limits the resident memory of the job's child process and everything it starts: the worker reads it every 0.5 s (`/proc` on Linux, `ps` elsewhere) and kills the job past the limit, recording `MemoryLimitError: the job used N MB resident, over the M MB limit (ARRIVAL_MAX_JOB_MB), during stage '...'`. Like a timeout it is not retried. It needs the child process, so it has no effect with `max_job_seconds` 0. A 2-episode hero_kitchen job peaks at about 365 MB (docs/ref/MEMCAP.md).
- **Retention.** `python -m arrival gc --max-age-hours N [--dry-run] [--out out]` deletes finished (`done`, `failed`, `cancelled`) jobs whose `finished_at` is older than N hours, with `out/jobs/<id>/`, `out/twins/<room>/`, `out/datasets/<room>_sim/`, `out/reports/<room>_sim.json` and the sim progress journal `out/progress/<room>_sim.jsonl`, only for a `room` that is the job's own (`..._<job id>`); queued and running jobs and hand-made rooms are never touched. `ARRIVAL_RETENTION_HOURS` (or `arrival work --retention-hours`) makes a running worker do the same sweep every 5 minutes; unset, nothing is ever deleted. A file that cannot be removed leaves the job listed with its `status.json`: `gc` logs it and retries on the next sweep, and `DELETE` answers 500. `DELETE /jobs/<id>` calls `delete_job` for a finished job, and the page shows a retention note (the `ARRIVAL_RETENTION_HOURS` value in the server's environment, or "kept until the uploader deletes them").
- **Retry.** A failed attempt is queued again up to `--retries` times (env `ARRIVAL_JOB_RETRIES`, default 2; `0` = never), written while the worker still holds `claim.lock`. `job_attempts` counts starts; `retry_errors` holds the error of each attempt that was retried. A bad upload, a request over a limit, `unsupported_layout`, an unusable dataset or the wall-clock timeout is not retried. A retry writes `not_before` and waits `ARRIVAL_JOB_RETRY_BACKOFF` seconds (default 5, doubling per attempt, capped at 300) before `claim_next` will take it.
- **Metering.** Every attempt that reaches the end of `process` gets a `usage` block in `status.json` and one line in `out/usage/jobs.jsonl` (the ledger, kept after `gc`): `wall_seconds`, `stage_seconds`, `episodes_attempted`, `episodes_succeeded`, `frames` (from the dataset's `total_frames`), `upload_bytes`, `zip_bytes`, `dataset_bytes`, `measured_on: local_cpu`, plus `job_attempts` so a retried job's lines stay distinct. A value that was not produced is `null`, never 0. For a timeout or a dead child the worker fills `wall_seconds` itself and the stage in flight has no split. `python -m arrival.eval.cost` reads the usage block and the ledger into the job rows (`frames`, wall seconds) and a `metering` totals object; nothing is priced from it, since the run is on a local CPU and no Nebius product was used.

```bash
python -m arrival work                   # until Ctrl+C
python -m arrival work --once --episodes 2
```

Measured on a laptop CPU: a `scan.json` upload with `--episodes 2` took 5.9 s end to end, 2/2 episodes, 322 frames, zip 77,624 bytes. Several workers can share one local `out/` (see above). A failed job is retried up to 2 times by default; there is no memory cap, and nothing runs on Nebius (OPS-2).

### Smoke test for a running server

`scripts/smoke_hosted.py URL` checks a running web server, local or hosted, with the standard library only:

```bash
python scripts/smoke_hosted.py http://127.0.0.1:8000
python scripts/smoke_hosted.py http://127.0.0.1:8000 --wait 600 --json   # also wait for `work` to finish the job
```

It needs the index page to load; the hero `roomplan.json` uploaded as `scan` to be queued (202) with a readable status; uploads without a `scan` field, with a non-multipart body or with an empty file to be refused with a 4xx; and crafted job and download paths (`../`, encoded `%2F`) to give 400, 403 or 404 without file contents. `/healthz` (the server answers `{"status": "ok"}`) is reported, and required with `--require-healthz`. With `--wait N` it polls the job until the worker marks it `done` and then downloads `dataset.zip`. Exit 0 when every required check passes, 1 otherwise. The good upload is a real job (`smoke_<unix time>.json`), so run it only against a server whose queue you may write to. Checked against `serve` plus `work` on a laptop: every check passed and the job's 637 KB `dataset.zip` downloaded.

## Tavily research

`python -m arrival research` looks up appliance specs (door and handle mechanisms, clearances, safety guidance) and caches the answers in `arrival/models/tavily_cache.json`, so repeat runs need no network. With no `TAVILY_API_KEY`, or when the call fails, it serves the cache and then a small built-in knowledge table; the source of each answer (`tavily_api`, `cache` or `fallback`) is printed.

```bash
python -m arrival research --room hero_kitchen             # appliances in the room's twin -> out/research/hero_kitchen_research.json
python -m arrival research --query "dishwasher door latch" # one direct query -> out/research/query_research.json
python -m arrival sim --tavily --episodes 2                # attach the research to the twin and the VLM planner prompt
```

Only appliance fixtures present in the twin are looked up (the sink counts; counters and walls are skipped). The research shapes the planner prompt and tags tasks `tavily_enriched`; it does not change the physics.

## Output layout

`out/` is gitignored. Nothing under it is committed.

```
out/
  twins/hero_kitchen/       twin.json, layout.yaml (RoboCasa subset), tasks.json
  datasets/hero_kitchen/    GR00T LeRobot v2
    meta/info.json          arrival_generator, arrival_physically_valid, arrival_successes, ...
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/modality.json      single_arm [0,7], gripper [7,8], base [8,11] (sim datasets), ego_view, task description
    meta/embodiment.json
    meta/episodes_stats.jsonl, meta/stats.json
    meta/arrival_cartesian.jsonl   cartesian sidecar (eef/object traces)
    data/chunk-XXX/episode_XXXXXX.parquet   (1,000 episodes per chunk)
    videos/chunk-XXX/observation.images.ego_view/episode_XXXXXX.mp4
  datasets/hero_kitchen_sim/ MuJoCo sim dataset (same layout, ego_view + wrist videos)
  shards/<i>of<n>/          `sim --shard i/n`: datasets/, reports/, progress/ of one shard; `merge-shards` joins them
  progress/<room>_sim.jsonl per-task journal that `sim --resume` continues from
  previews/                 sim contact sheets and overview mp4s (--save-preview)
  research/                 `research` output (<room>_research.json, query_research.json)
  reports/hero_kitchen.json attempts, successes, dropped, seconds, demos/hour, dollars
  eval/<run_id>/report.json        closed-loop eval summary and checkpoint identity
  eval/<run_id>/manifest.json      frozen twin, settings and ordered trials
  eval/<run_id>/episodes.jsonl     per-attempt outcomes, safety and errors
  eval/<room>_realism.json, .png    `realism`: twin render vs held-out frames, misaligned-pose control
  jobs/<job_id>/            upload.bin, status.json (with a `usage` block when finished), claim.lock while a worker runs it, frames/ (optional RGB-D upload); after `work`: dataset.zip (done) or error.txt (failed)
  usage/jobs.jsonl          one metering line per finished attempt (kept by `arrival gc`)
  twins/<stem>_<job_id>/, datasets/<stem>_<job_id>_sim/   what `work` compiled and simulated for an upload
```

The demo dataset's embodiment label is `robocasa_panda` (7 joints + 1 gripper, placeholder joints) and its two task families are `counter_to_sink` and `counter_to_counter`; the sim dataset is labelled `arrival_sim_panda` and lists its families in `arrival_families` (see the sim dataset notes in [04-contracts.md](04-contracts.md)). Both vary by object, start pose, lighting, and instruction wording.

## Credentials

The demo uses no credentials and makes no network calls, and neither do the tests. Every variable below is read through `arrival/config.py` (`SETTINGS` lists each one, whether it is a secret, and which module uses it); no other module touches the environment, and `tests/test_config.py` fails if one starts to:

| Variable | Purpose |
|---|---|
| `TOKEN_FACTORY_API_KEY` | Bearer token |
| `TOKEN_FACTORY_BASE_URL` | Defaults to `https://api.tokenfactory.nebius.com/v1`. Check it against the live docs |
| `NEMOTRON_OMNI_MODEL`, `NEMOTRON_ULTRA_MODEL`, `NEMOTRON_SUPER_MODEL`, `NEMOTRON_LIGHTNING_MODEL` | Optional model ids |
| `TAVILY_API_KEY`, `TAVILY_BASE_URL` | Optional, for `research` and `sim --tavily`; read only in `arrival.models.tavily`. Base URL defaults to `https://api.tavily.com/search` |
| `GR00T_POLICY_URL` | Address of a GR00T policy server, for `--policy vla` |
| `COSMOS_URL` | Cosmos endpoint for `python -m arrival.render.variation --backend cosmos` |
| `NEBIUS_IAM_TOKEN`, `NEBIUS_PROJECT_ID`, `NEBIUS_API_URL` | Nebius REST API for `arrival fanout --backend nebius`: an access token from `nebius iam get-access-token` (secret, valid 12 h), the project the jobs run in, and the API base (defaults to `https://api.nebius.cloud`) |
| `NEBIUS_JOB_IMAGE`, `NEBIUS_JOB_PLATFORM`, `NEBIUS_JOB_PRESET`, `NEBIUS_JOB_SUBNET_ID`, `NEBIUS_JOB_S3_SECRET_ID` | The shard jobs: the arrival image, platform and preset (default `cpu-d3`, `4vcpu-16gb`), an optional subnet, and the MysteryBox secret that hands the bucket key to each job |
| `NEBIUS_BUCKET`, `NEBIUS_ACCESS_KEY_ID`, `NEBIUS_SECRET_ACCESS_KEY`, `NEBIUS_REGION`, `NEBIUS_S3_ENDPOINT` | Object Storage: the bucket, an access key pair (both secret), the region (default `eu-north1`) and the endpoint (default `https://storage.<region>.nebius.cloud`) |
| `ARRIVAL_DOTENV` | `0` turns the `.env` lookup off; the test suite sets it so a developer's real keys never reach a test |

Export them in your shell, or copy `.env.example` to an untracked `.env` (gitignored) in the repo root or cwd. `arrival.config.get` reads `.env` (cwd first, then the repo root) only for a variable that is unset, and never overrides one already set. `arrival.config.status()` reports which variables are set without their values, and `arrival.config.redact()` (also a logging filter, `install_log_redaction`) scrubs set secrets and common token shapes (Bearer tokens, `nvapi-`/`tvly-` keys, `user:pass@` in URLs, `?token=` parameters) from text before it is printed or stored; the `--policy vla` connection error goes through it. Never commit keys. Account links and what to request are in [docs/08-access.md](08-access.md).

## Cost

`python -m arrival.eval.cost [--out out] [--usage usage.json] [--pricing configs/pricing.example.yaml] [--baseline teleop_baseline.json] [--json]` turns `out/reports`, `out/eval`, finished job statuses and the worker's usage ledger `out/usage/jobs.jsonl` into demos per hour, seconds per successful demo and drop rate, with dollar rows that stay null unless a usage file gives units, and a generated-vs-teleop comparison only when a valid teleop baseline exists (`docs/ref/OPS-3a.md`). The pricing rates in `configs/pricing.example.yaml` are assumed, not invoiced.

### Teleop baseline

`scripts/teleop_baseline.py` times a person teleoperating the same task in the same twin, and writes the `teleop_baseline.json` that `arrival.eval.cost` reads (`arrival/sim/teleop.py`):

```bash
python scripts/teleop_baseline.py --operator-usd-per-hour 40 --successes 20 --record   # plain python, also on macOS
python scripts/teleop_baseline.py --operator-usd-per-hour 40 --successes 20 --record     # Linux, Windows
```

After that, the cost module above prints the generated-vs-teleop comparison.

A window opens over the robot's shoulder; type into it (the task and the keys are shown in its corners; drag to orbit, right-drag to pan, scroll to zoom). `w`/`s` move the gripper toward or away from the counter, `a`/`d` left and right, `r`/`f` up and down (1 cm a press, 4 cm with a capital), `q`/`e` turn it, space opens or closes it, Enter judges the attempt, `x` gives it up, Esc quits. Each key moves an end-effector target that goes through the scripted expert's IK at 20 Hz, so the arm, the physics and the success oracle are the ones `sim` uses; on Enter the arm holds still for a second and then the oracle decides, and a human contact fails the attempt. Operator time runs from the moment an attempt's scene is ready to its judgement, so failed and abandoned attempts count. `cost_basis_usd` is operator hours times `--operator-usd-per-hour` plus `--setup-usd`, nothing else; `--assumption` adds lines to `setup_assumptions`. Every attempt is logged to `out/reports/<room>_teleop_attempts.jsonl`. No baseline file is written until at least one attempt succeeds. `--record` writes the successful demos to `out/datasets/<room>_teleop` (`arrival_generator: teleop_keyboard`, oracle-checked), which can serve as DAT-4 seeds. One window serves every attempt and keeps your camera between them; `--no-viewer` reads keys from the terminal with no window. The default task is `counter_to_sink` with no people, as the OPS-3 baseline asks; tested headless with a scripted key stream (`tests/test_teleop.py`) and once by a person (2 attempts, 1 bowl in the sink, 117 operator seconds); a credible baseline still needs ~20 successes.

None of the Nebius products below were called. Token Factory calls from `--planner vlm` and `--instructions nemotron` are not metered here either. The only measured number is the local generation run, which cost $0 on a laptop CPU. That says nothing yet about how generated data compares with teleoperation, because the comparison has not been measured.

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
  --dataset-path out/datasets/hero_kitchen_sim \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path configs/gr00t/arrival_panda_config.py \
  --num-gpus 1 \
  --output-dir out/checkpoints/gr00t \
  --max-steps 2000 \
  --global-batch-size 32
```

The modality config is `configs/gr00t/arrival_panda_config.py` (arm trained as relative joint deltas, gripper absolute, 16-step action horizon, base left out). For a dataset written with `--base mobile` use `configs/gr00t/arrival_panda_mobile_config.py` instead, which adds the `base` key as an absolute commanded pose; only one of the two can be loaded per run. `tests/test_gr00t_config.py` checks both configs' keys against the `meta/modality.json` that `arrival sim` writes (fixed and mobile), with GR00T stubbed out. Not yet checked: that GR00T's own loader reads the dataset (needs the Isaac-GR00T install and a GPU box) and that its stats files compute without NaNs.

### Appearance variation (offline)

`python -m arrival.render.variation SRC_DATASET NEW_COPY_DIR [--backend local_stub|cosmos] [--seed N] [--fraction 0.1]` copies a sim dataset and restyles about 10% of its frames; the source is untouched. Writes `meta/arrival_variation.jsonl`, `meta/arrival_variation_summary.json`, `variation_videos/`, and sets `arrival_appearance_variation` in the copy's `info.json` to `cosmos` (only after a real Cosmos success), `local_stub` or `none`. `cosmos` reads `COSMOS_URL`; without it every selected frame is reported as skipped and stays original. The summary reports backend call seconds and added stage wall seconds; cost stays null without endpoint billing data. See [ref/SIM-7a.md](ref/SIM-7a.md).

## Known gaps

Not built yet, and not claimed anywhere in the outputs:

- Field-tested iPhone capture. The app exists (`ios/`, CAP-1) and uploads to `POST /jobs`, but no real kitchen scanned with it is checked in yet. Uploads are processed only by a local `python -m arrival work` process.
- MimicGen generation (`arrival/generate/mimicgen.py` is a stub).
- Gaussian splat training or compositing (`arrival/render/splat.py` is a stub).
- Cosmos restyle of frames.
- A real Nebius run of the fan-out. `arrival fanout --backend nebius` submits, polls and stores shards through the documented Nebius APIs, but it has only run against in-process fakes; no Nebius job has been created.
- GR00T fine-tune (command above, not run). `python -m arrival eval` scores policies in the sim; it has only been run with the scripted expert.
- The mobile base (`--base mobile`) drives only for `counter_to_counter`, in a straight line along one run; the door and drawer families still park once (so `open_door` still stops at about 63-66° and `put_in_drawer` keeps its 0.6 m rule), and a target on another wall needs a turn and a route that no family plans yet. `--base mobile --policy vla` is refused because the policy server's wire format carries no base pose (G2).
- RoboCasa-native layout conventions: `layout.yaml` is a subset emitter (world-frame `pos`, `z_rot` on oblique walls), not loadable by RoboCasa as-is.

