# Phase 1 local roll tracking

This folder contains the single-camera Phase 1 tracker. It tracks only class `1`
(`roll`) and adds application-level global IDs and lifecycle state above the
local tracker. It deliberately does not implement cross-camera Re-ID, bin
selection, OCR, relay control, or ply tracking.

## Setup

Activate the Python environment that contains the V5 Ultralytics installation,
then install the tracking dependencies:

```bash
python -m pip install -r requirements.txt
```

Put the rear-camera videos at `videos/cam2.mp4` and `videos/cam3.mp4`.

## First 1,000-frame Cam2 test

```bash
python track_rolls.py \
  --model "../runs/obb/v5_fresh/weights/best.pt" \
  --source "videos/cam2.mp4" \
  --output "outputs/cam2_tracking_tuned.mp4" \
  --tracker configs/bytetrack_roll.yaml \
  --conf 0.12 \
  --iou 0.50 \
  --imgsz 640 \
  --codec avc1 \
  --motion-window 8 \
  --motion-threshold 20 \
  --active-hold-frames 15 \
  --max-missed-frames 90 \
  --max-frames 1000 \
  --debug
```

Add `--device mps` to request Apple MPS explicitly. Omit it to let
Ultralytics select the device. Once the preview looks correct, omit
`--max-frames 1000` to process the full video.

Cam3 uses its own calibrated zone file; never reuse the Cam2 coordinates:

```bash
python track_rolls.py \
  --model "../runs/obb/v5_fresh/weights/best.pt" \
  --source "videos/cam1.mp4" \
  --output "outputs/cam3_tracking_destination_fixed.mp4" \
  --tracker configs/botsort_roll.yaml \
  --conf 0.12 \
  --iou 0.50 \
  --imgsz 640 \
  --codec avc1 \
  --device mps \
  --zones configs/cam3_zones.json \
  --destination-proximity-px 60 \
  --draw-zones \
  --max-frames 1000 \
  --debug
```

The output logs are derived from the video name. For example,
`cam2_tracking.mp4` produces `cam2_tracking.csv`, `cam2_events.csv`, and
`cam2_tracking_summary.txt`.
Existing outputs are protected; pass `--overwrite` deliberately to replace them.
The default `avc1` codec writes H.264 MP4 files that play in QuickTime on macOS.

ByteTrack may occasionally leave a new or unmatched detection without a track ID
for a frame. The video shows these transient boxes as `UNTRACKED`; they are not
added to the tracking CSV until ByteTrack assigns an ID, and processing continues.

## Controlled BoT-SORT comparison

The custom BoT-SORT configuration uses the supported Ultralytics 8.4.x fields
and keeps its default `with_reid: False`; it does not download or use an external
ReID model. Use the same detector and inference settings as the ByteTrack test:

```bash
python track_rolls.py \
  --model "../runs/obb/v5_fresh/weights/best.pt" \
  --source "videos/cam2.mp4" \
  --output "outputs/cam2_tracking_botsort.mp4" \
  --tracker "configs/botsort_roll.yaml" \
  --conf 0.12 \
  --iou 0.50 \
  --imgsz 640 \
  --codec avc1 \
  --motion-window 8 \
  --motion-threshold 20 \
  --active-hold-frames 15 \
  --max-missed-frames 90 \
  --max-frames 1000 \
  --debug
```

This produces `cam2_tracking_botsort.mp4`, `cam2_tracking_botsort.csv`,
`cam2_events_botsort.csv`, and `cam2_tracking_botsort_summary.txt` without
overwriting the ByteTrack outputs.

## Global roll identity layer

`track_rolls.py` keeps the selected local tracker ID truthful and treats the
application-level global ID as a **transport journey ID**. Static detections do
not own global IDs. They remain `UNASSIGNED_STATIC` and are drawn as
`ROLL | Local N | STATIC | Global ---`. Endpoint activation remains the fast
path: activation at A creates `A_TO_B`, while activation at B creates `B_TO_A`.
If activation occurs away from both endpoints, the local enters a bounded
PENDING decision phase and can be promoted by sustained coherent motion.

Unassigned locals use three explicit states: `UNASSIGNED_STATIC`,
`PENDING_PICKUP`, and `ASSIGNED_TO_GLOBAL`. PENDING promotion requires five
ACTIVE observations, at least 20 pixels of net displacement, and a net/path
motion-consistency ratio of at least 0.6. Direction uses the latest 10 trajectory
points relative to the A–B axis and endpoint distance changes. PENDING is capped
at 30 frames. Static locals never enter the global candidate pool. Precommit
centroid history initializes velocity for later tracker-fragment reassociation.

A committed global roll becomes `OCCLUDED` when its local track disappears,
remains eligible for geometry-only reassociation, and becomes `LOST` only after
`--global-occlusion-frames`. Invisible rolls retain identity state but receive
no synthetic detection box.

Reassociation uses the minimum of last-centroid and velocity-predicted distance,
gap duration, velocity prediction quality, and rotation-independent OBB area.
Short gaps use a velocity-aware spatial allowance before any new global ID is
created. Size contributes a gradual log-space score and is only a hard rejection
outside broad emergency limits. Rolls that were ACTIVE immediately before an
occlusion also get a separate long-gap window. That path combines position,
time, pre-occlusion movement direction, size, and an ACTIVE prior; direction is
soft evidence rather than a hard gate. If more than one
qualifying local/global pairing exists, the manager logs
`AMBIGUOUS_REASSOCIATION` and creates a separate global ID rather than guessing.
No appearance embeddings or cross-camera association are used.

## Offline synchronized cross-camera debug video

`make_cross_camera_debug_video.py` combines Cam0 and Cam1 on one normalized
timeline. It is a read-only visual validation tool: it does not run or modify
YOLO, BoT-SORT, camera-global identity, or cross-camera matching.

Normalized time is `video timestamp + camera offset`. Calibrate the offsets
from one obvious physical event visible in both videos:

```bash
python estimate_camera_offset.py --cam0-time 12.30 --cam1-time 17.80
```

The example prints `--cam0-offset 0 --cam1-offset -5.5`, because Cam1's event
at 17.80 seconds then normalizes to Cam0's 12.30 seconds. Generate a short
side-by-side inspection window with:

```bash
python make_cross_camera_debug_video.py \
  --cam0-video videos/cam0.mp4 \
  --cam1-video videos/cam1.mp4 \
  --cam0-csv outputs/cam0_actual_bin_entry.csv \
  --cam1-csv outputs/cam1_actual_bin_entry.csv \
  --cam0-offset 0 \
  --cam1-offset -5.5 \
  --start-time 15 \
  --end-time 30 \
  --output outputs/cross_camera_debug.mp4
```

Add `--cross-camera-csv outputs/cross_camera_matches.csv` when that file
exists. Match-centric CSV columns understood by the visualizer are:
`factory_roll_id`, `cam0_global_id`, `cam1_global_id`, optional
`cam0_exit_timestamp`, `cam1_entry_timestamp`, `handoff_gap_seconds`, and
`match_score`. It also accepts an event-oriented CSV. Factory IDs are shown
only when explicitly present in this file; they are never inferred.

The default output covers the union of the normalized camera timelines. A
camera is shown as `NO FRAME` outside its own time range. Use `--overwrite` to
replace an existing output intentionally.

## Bin-focused single-camera output

For a clean single-camera review containing only physical-bin geometry and a
summary TXT (no CSV logs), use `--bin-focused --draw-bins --summary-only` and
do not pass `--draw-zones`. In this mode the video and summary report bin
membership from OBB-to-bin polygon overlap; zone A/B direction and destination
diagnostics are hidden.

### Production and debug overlays

The default video overlay is production-oriented. It shows only the current
Global/Factory Roll ID, expected bin, actual bin once known, and the operator
status (`TRACKING`, `PLACEMENT PENDING`, `CORRECT`, `WRONG BIN`, or
`UNRESOLVED`). A wrong placement is highlighted as a clear red warning. Bin and
entry polygons remain available through their existing drawing options.

Add `--debug-overlay` to restore Local IDs, lifecycle/candidate states,
trajectories, scores, reasons, occluded/sealed globals, reassociation details,
and internal counters on the rendered video. This flag changes only the video
display; CSV, log, and summary detail is unchanged.

### Lifecycle zones and terminal completion

`configs/cam2_zones.json` contains real endpoint polygons from the existing
Cam0 setup. The former pickup polygon is `zone_a`; the former handoff polygon is
`zone_b`; their coordinates were not changed. Legacy configs using
`pickup_zone` and `handoff_zone` are normalized at load time. The polygons are
visibly drawn during validation and remain operator-editable. Each needs at
least three `[x, y]` points.

To replace the polygons by clicking on a representative frame:

```bash
python configure_zones.py \
  --source videos/cam0.mp4 \
  --frame 520 \
  --output configs/cam2_zones.json \
  --overwrite
```

The helper collects `zone_a`, `tracking_zone`, then `zone_b`. Left
click adds a point, `U` undoes, `R` resets the current polygon, and `N` or Enter
finishes it. Escape cancels without writing.

For Cam3, the helper defaults to the real `cam1.mp4` input and a separate
`configs/cam3_zones.json` output:

```bash
python configure_zones.py
```

It opens frame 260 and collects `zone_a`, `tracking_zone`, and `zone_b` in that
order. On Cam3, Zone A is the left physical bin and Zone B is the right physical
bin; neither name permanently means source or destination. The origin is the bin
where a journey starts and the opposite bin becomes its destination. After all
three polygons are entered, the
helper displays a combined preview and writes nothing until `S` or Enter is
pressed. It never overwrites `cam2_zones.json`; use `--overwrite` only when
deliberately recalibrating an existing Cam3 file.

At startup the tracker prints the loaded path, every polygon, validation errors
or out-of-frame warnings, and `Lifecycle enabled: YES/NO`. Invalid or incomplete
zones visibly disable lifecycle behavior rather than failing silently.

A static local is tracked without a global ID. When it changes from STATIC to
ACTIVE within `--endpoint-margin-px` of A or B, a directional transport ID is
created and the opposite endpoint becomes its destination. After it enters that
destination, any of these conditions makes it terminal
`COMPLETED`:

- it disappears;
- it leaves the configured tracking polygon; or
- it remains in the destination polygon for `--handoff-hold-frames` consecutive
  frames.

For Cam3 B-to-A journeys, `--destination-proximity-px 60` provides a
conservative exit fallback. The centroid must first move consistently toward
Zone A and end within the configured distance, and the roll must subsequently
disappear. Proximity by itself never completes a visible journey. The overlay
shows destination distance and `DESTINATION APPROACH` when this fallback is
armed. Use `--debug-verbose` only when per-frame distance output is needed.

`COMPLETED` and `LOST` identities are never reassociation candidates. Only
`OCCLUDED` identities can be reconnected to a new local tracker ID.

```bash
python track_rolls.py \
  --model "../runs/obb/v5_fresh/weights/best.pt" \
  --source "videos/cam0.mp4" \
  --output "outputs/cam2_tracking_pending_promotion.mp4" \
  --tracker "configs/botsort_roll.yaml" \
  --conf 0.12 \
  --iou 0.5 \
  --imgsz 640 \
  --device mps \
  --zones-config "configs/cam2_zones.json" \
  --endpoint-margin-px 50 \
  --journey-activation-frames 5 \
  --journey-min-displacement 20 \
  --journey-motion-consistency 0.6 \
  --journey-direction-window 10 \
  --pending-max-frames 30 \
  --draw-zones \
  --max-frames 1000 \
  --debug
```

`--reassoc-max-distance` is a fraction of the frame diagonal. Short gaps up to
30 missing frames start with a 150-pixel allowance that expands with recent speed
and gap duration, capped at 300 pixels. Area ratios 0.25–4.0 define the soft score
scale; only ratios outside 0.10–8.0 are emergency rejections. Motion state does
not change an already assigned identity; the STATIC-to-ACTIVE transition is only
the commit point for a previously unassigned pickup. For gaps longer than the
short-gap window, only a roll that was ACTIVE immediately before disappearing
can use the expanded ACTIVE window and distance. Multiple plausible pairings
remain ambiguous and are never guessed.

Every tracked detection row includes `assignment_state`; `global_id` is blank
for static/unassigned rows. The local event CSV records `LOCAL_UNASSIGNED`,
`PENDING_STARTED`, `PENDING_PROMOTED`, `PENDING_CANCELLED`, and
`PENDING_DIRECTION_AMBIGUOUS` transitions plus their motion/direction evidence.
A separate global event
CSV records `GLOBAL_CREATED`, `GLOBAL_VISIBLE`, `GLOBAL_OCCLUDED`,
`GLOBAL_REASSOCIATED`, `GLOBAL_PICKED_UP`, `GLOBAL_ENTERED_DESTINATION`,
`GLOBAL_COMPLETED`, `GLOBAL_LOST`, and `AMBIGUOUS_REASSOCIATION`. Global events
include `origin_zone`, `destination_zone`, and `journey_direction`.

## Global physical-bin geometry

`configs/bins.json` defines the global physical IDs `BIN_1` through `BIN_4`.
Each camera stores its own polygon observation under the same physical ID. For
example, `cameras.cam0.BIN_3` and `cameras.cam1.BIN_3` are two views of the one
physical `BIN_3`; camera-local names are never generated. The initial config is
intentionally empty because final coordinates must be selected from the videos.

Configure Cam0 from a representative frame:

```bash
python configure_bins.py \
  --source videos/cam0.mp4 \
  --camera cam0 \
  --config configs/bins.json \
  --frame 520
```

Configure Cam1 without removing Cam0's polygons:

```bash
python configure_bins.py \
  --source videos/cam1.mp4 \
  --camera cam1 \
  --config configs/bins.json \
  --frame 260
```

Press `1` through `4` to select the global bin ID, left-click polygon points,
and press Enter to commit that polygon. `U` undoes a point, `R` resets the
selected polygon, and `D` deletes it. Press `S` for the combined preview and
`Y` to write the config. Existing configs are backed up automatically.

The reusable `bin_geometry.py` functions provide conservative geometry only:
`point_in_bin(...)` returns a unique physical bin ID or `None`, while
`obb_bin_overlap(...)` returns overlap ratio, pixel intersection, OBB area, and
centroid membership for every configured bin. They do not trigger PASS/FAIL or
change any roll lifecycle state.

Add the visual-only overlay to a normal tracking command with:

```bash
--bins-config configs/bins.json --camera-id cam0 --draw-bins
```

Only polygons configured for that camera are drawn. The overlay is not consumed
by YOLO, BoT-SORT, reassociation, PENDING promotion, or lifecycle decisions.

### Actual physical-bin entry

When `--camera-id` is supplied, visible transported Global Rolls are evaluated
against that camera's physical-bin polygons. `bin_entry.py` calculates each
bin's intersection with the four-corner roll OBB and divides the intersection
area by the OBB area. The highest overlap becomes a candidate only at or above
`--bin-overlap-threshold` (default `0.35`). It must remain the unambiguous best
candidate for `--bin-confirm-frames` consecutive visible observations (default
`5`). If two qualifying overlaps differ by less than
`--bin-ambiguity-margin` (default `0.10`), evidence pauses instead of guessing.

Starting-bin occupancy is excluded: confirmation is armed only after a journey
clears the physical bin in which it began. Confirmed assignments are stored on
the Global Roll, survive local-ID changes, and remain locked. The debug-only
`--allow-bin-reassignment` option requires `--debug`. This layer reports actual
entry only; it does not implement expected-bin comparison, PASS/FAIL, alarms,
or ESP32 behavior.

Cam0 1,000-frame validation:

```bash
python track_rolls.py \
  --model ../runs/obb/v5_fresh/weights/best.pt \
  --source videos/cam0.mp4 \
  --output outputs/cam0_actual_bin_entry.mp4 \
  --tracker configs/botsort_roll.yaml \
  --conf 0.12 --iou 0.5 --imgsz 640 --device mps \
  --zones configs/cam2_zones.json \
  --bins-config configs/bins.json --camera-id cam0 \
  --draw-zones --draw-bins \
  --bin-overlap-threshold 0.35 \
  --bin-confirm-frames 5 \
  --bin-ambiguity-margin 0.10 \
  --max-frames 1000
```

For Cam1, use `videos/cam1.mp4`, `configs/cam3_zones.json`, camera ID `cam1`,
and a distinct output filename. Each run writes a dedicated
`*_bin_events.csv` containing candidate, reset, ambiguity, and confirmation
events. The regular tracking CSV also contains candidate and confirmed-bin
fields, and the text summary reports aggregate and per-journey assignments.

## Optional ROI

Adjust the example pixel coordinates in `configs/roi.json` to the camera frame,
then add:

```bash
--roi-config configs/roi.json
```

Without this argument, the full frame is tracked.

## Tests

```bash
python -m unittest -v test_track_rolls.py test_global_identity.py
```
