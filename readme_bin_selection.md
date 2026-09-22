RTracker&OCR - Ordered result queue + ESP32 bin lights: decisions (queue_v1 -> queue_v2)
==========================================================================================

1. PROBLEM
- Pi 5 (2 GB RAM, Hailo-8L) running roll/ply/range detector v2 + PP-OCRv5 mobile OCR confirms results too slowly.
- A roll's result could come out after the next roll was already in frame, out of order, or be lost.
- Goal: results in strict roll arrival order, even if delayed. The overall process may stretch, but the sequence stays correct.

2. FILES (naming rule: "v5" = PP-OCRv5 mobile model, not script version 5; new logic gets a feature suffix)
- rtracker_ocr_paddle_v5_queue_v1.py : first queue logic (ply-only lookup, one row per ply). Kept unchanged.
- rtracker_ocr_paddle_v5_queue_v2.py : current version with the bin-selection rules below.
- Not modified: ocr.py, detector.py, rtracker_ocr.py, ocr_paddle_v5.py, rtracker_ocr_paddle_v5.py,
  ply_bin_lights.py, bin_light_controller.py, ESP32 sketch.
- Outputs: output/annotated_queue_v2.mp4, output/readings_queue_v2.csv (v1: *_queue_v1.*), so v5 outputs are never overwritten.

3. QUEUE ARCHITECTURE
- Main loop (15 fps): detection + roll tracking only, never waits for OCR.
- Each roll gets a job when text is first seen on it; seq number = arrival order (rolls pass one at a time).
- Snapshots: straightened ply/range crops (not full frames) are added to the roll's job every --snap-every frames
  (default 2); at most --snapshots crops (default 6) are kept waiting per field; the blurriest is dropped first;
  the sharpest is read first.
- One OCR worker reads jobs strictly in seq order. A roll's result is never overtaken by the next roll.
- A roll's crops are kept after it leaves the frame, so a slow read never loses the roll.
- When the source ends, rolls still in view are closed and the queue finishes in order.
- The video overlay shows "#track seq N Ply .. | range .. -> bin" and the number of rolls waiting in the queue.

4. CONFIRMATION / TIMING
- --votes (default 3): matching OCR reads needed to confirm the ply (and the range).
- OCR read counts only if its confidence is >= 0.5 (MIN_SCORE).
- --read-timeout: optional, off by default. If given, counted from when OCR starts on that roll (reaching
  the head of the queue), not from when it was first seen, so waiting in the queue doesn't use it up.
  Without it, a roll is UNREADABLE only after it has left the frame (1 s without detection) and all its crops are read.
- The timeout is not based on master.csv size (a ply lookup is instant); confirmation is by --votes.

5. MASTER SHEET
- Must be passed on every run: --master <path> (required, e.g. data/master.csv). No default.
- Columns used: ply_no (--key-col), no_of_ply = bin (--bin-col), start, end. Other columns ignored.
- Ply matching is loose: "080", "80", "80.0" are the same ply.
- Re-read automatically when the file changes; if it can't be read, the last good copy is kept.
- The sheet is used as is (e.g. ply 205 really is 1-DW / 1-UW only).

6. BIN SELECTION (per confirmed ply)
- Bin priority order: 1-DW, 1-UW, 2-DW, 2-UW (= ESP32 relays 1, 2, 3, 4).
- Ranges of the same ply within 0.1 on both start and end (--range-tol) count as the SAME range
  (e.g. ply 205: 2.75-19.2 and 2.8-19.2 -> same range -> case 2).
- Case 1 - ply has one row: that bin. Range not checked. Lit as soon as the ply is confirmed.
- Case 2 - ply has several rows, all the same range: first unused bin in priority order. Range not checked.
  Lit as soon as the ply is confirmed. Example: ply 81 -> 1st roll 1-DW, 2nd roll 1-UW.
- Case 3 - ply has rows with different ranges: waits for the roll's range reading, then:
  - Uses the confirmed range, or the BEST reading so far (most reads) if not confirmed.
  - Closeness = number of differing digits (edit distance), decimal points ignored (OCR often drops them),
    start and end compared separately and added. Numbers are normalised first (78.90 -> 789, 45.0 -> 45).
  - Match only if the closest range differs by at most 2 digits in total (--range-max-diff 2; test value,
    to be adjusted if tests aren't satisfactory).
  - Tie (two ranges equally close): the one whose first free bin comes first in the priority order.
  - Within the chosen range: first unused bin in priority order.
  - Example: ply 93 has 28.4-48.5 (1-DW) and 50.6-70.4 (1-DW, 1-UW). Read "59.6-704" -> 1 digit off
    50.6-70.4 -> 1-DW, next such roll -> 1-UW.
  - Example: read "59.6-789" vs 50.6-78.9 = 1 digit, vs 34.5-67.8 = 5 digits -> 50.6-78.9.
  - Case 3 rolls light later than cases 1/2 because they wait for the range.

7. USED BINS / DUPLICATES
- Every sheet row (ply + range + bin) can be used once per set of rolls.
- A roll that needs an already used row -> "DUPLICATE" in the terminal, no light (applies to cases 1, 2 and 3).
- Memory is erased only by typing: reset + Enter in the terminal (works with --no-show on the Pi).
  Not reset automatically. Restarting the script also starts empty. Used rows are kept when master.csv is reloaded.
- Known risk: if the tracker loses a roll and picks it up again, the same roll is counted twice -> it takes the next
  bin (cases 2/3) or shows DUPLICATE.

8. OUTCOMES (terminal line per roll, in seq order; CSV row per roll)
- Lit: "[seq N] roll #id  ply P -> bin 1-DW  BIN 1 | range R (sheet S, case C, k of n for this range, x digits off) | lit t s after first seen"
- Ply not in sheet: "ply P not in master.csv" -> terminal only, no light.
- UNREADABLE (no confirmed ply): terminal only (with best guess for debugging), no light, not in the ESP32 sequence.
- Case 3, range not readable: "range NOT READABLE -- cannot choose between ..." -> no light.
  (Not reported as "not in csv".)
- Case 3, range matches no row: "range R does not match any row in master.csv (closest S, x digits off)" -> no light.
- Duplicate: "DUPLICATE -- bins ... for range S already used" -> no light.
- Range is always read and printed per roll next to the sheet's range, as reference. Each roll's range is
  independent (nothing carried to the next roll). Range-mismatch alert = future work, not implemented.
- CSV columns: timestamp, seq, track_id, status, case, ply, range_read, sheet_range, digits_off, bin, relay,
  sheet_row, seconds_after_first_seen. Status = picked | duplicate | not_in_master | range_unreadable |
  range_no_match | unreadable.

9. LIGHTS (ESP32, protocol unchanged: "BIN 0..4\n", replies OK)
- A lit bin stays on until the next roll lights a bin.
- Next roll to the same bin: switched off for --blink-gap (0.3 s) then on again (re-highlight, not continuous).
- Outcomes with no light leave the previous bin lit (known: the operator could misplace an unreadable roll).
- --esp32-port omitted = dry run (terminal only). --baud default 115200.
- (Light behaviour may change in later versions.)

10. KNOWN LIMITATIONS
- Queue order doesn't make OCR faster. If rolls arrive faster than OCR reads them, the lights fall further
  behind (watch the queue count on screen).
- A roll still in view without a confirmed ply (and no --read-timeout) holds up the rolls behind it.
- Assumes one roll at a time in arrival order.
- The 2-digit rule may confuse labels printed with 3 decimals (e.g. 65.433 read as 65.4 = 2 digits off).
- queue_v1/v2 import the camera/tracking helpers from rtracker_ocr_paddle_v5.py; editing that file affects them.
- Tested on the laptop only (video files, .pt model, dry run); not yet on the Pi + Hailo + ESP32.

11. COMMANDS
- Pi:  python3 rtracker_ocr_paddle_v5_queue_v2.py --master data/master.csv --source /dev/video0 --model models/rolls.hef --esp32-port /dev/ttyUSB0
- PC:  python rtracker_ocr_paddle_v5_queue_v2.py --master data/master.csv --source videos/cam0_onsite.mp4 --model models/rolls_v2.pt
- Options: --votes 3  --read-timeout <s>  --range-max-diff 2  --range-tol 0.1  --snapshots 6  --snap-every 2
  --blink-gap 0.3  --roi x,y,w,h  --exposure <100us units>  --no-show
- On the Pi, rolls.hef is the v1 detector; detector v2 (rolls_v2.pt) still needs compiling to rolls_v2.hef.
