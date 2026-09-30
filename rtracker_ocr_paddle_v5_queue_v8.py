"""Roll tracking + OCR (PP-OCRv5 mobile) with an ordered result queue that drives the ESP32 bin lights -- queue v8.
New in v8: works with a detector that has ONE text class (both text boxes of a roll labelled "ply"), next to the
old ply / range / roll detectors. Uses detector_v2.py, which maps the model's classes by name:
  single text class  every text box is queued as "text"; its way up comes from reading it both ways (the ply-above-
                     range layout needs to know which box is which); after the read, the TEXT decides the field:
                     '.' or number-number -> range ('50.8-78.3', '31.3.37.1', '44-46'), digits with or without
                     U/D W/B -> ply ('1511', '7841-DW', '85W'); anything else is only logged (no vote).
                     From there on the votes, confirmations and decision rules are exactly those of queue v7.
  ply / range model  unchanged from v7 (fields and way up from the detector classes).
  --classes          class names of a .hef in training order when it is neither ply,range,roll nor ply,roll
  --merge-text       treat ply and range boxes as one text class even if the model has both classes
  ocr_reads.csv      field column: "text:ply" / "text:range" / "text:-" for single-class boxes
Everything below is queue v7's description, unchanged:
Same reading as rtracker_ocr_paddle_v5_queue_v6.py (free reading of every box, fine-tuned model, no threshold,
read ahead + --ocr-workers), with its own rules for WHEN and HOW a roll is decided:
  early (green)   as soon as a ply value is CONFIRMED (--votes matching reads, or fast confirm) and its digits are
                  in master.csv, the roll is decided at once: bin picked and lit, box turns green on the overlay.
                  Case 3 also needs a confirmed range read within --range-max-diff of one of the ply's sheet
                  ranges (the closest one is used). Several such plys: the one confirmed last.
                  Once decided, no more crops of the roll are queued.
  otherwise       (confirmed but not in the sheet, e.g. '7' while '72' is being written; or nothing confirmed yet)
                  reading goes on over every frame, also after the roll has left, and the roll is decided with the
                  rules below once it has left (--close-after frames unseen) or --read-timeout has run out AND all its
                  crops are read:
  ply (end)       among the ply reads whose digits are in master.csv, the MOST RECENT read (latest video frame)
                  wins, even if read only once. None in the sheet: the latest confirmed value -> not in master.
                  Nothing confirmed either: UNREADABLE.
  range (end)     among ALL range reads of the roll, the one closest to one of the ply's sheet ranges wins (fewest
                  differing digits, at most --range-max-diff), even if read only once; tie -> the most reads.
Rolls are still decided, lit and reported strictly in seq order: a roll confirmed early waits for the rolls before
it to be decided first.
  clean crops     crops are cut from an untouched copy of the frame (no drawn boxes / ROI border in them).
Kept from v6: whole dictionary, no character restriction, no format check, no dot-to-dash fix, the whole read is
the value and only the sheet lookup uses its digits ('85W' -> '85'), every non-empty read is a vote, both ways up
read and logged when the layout does not give the way up, 'auto' boxes (text-only mode) only logged, every read
crop saved.
How the chosen ply picks its bin in master.csv
(a ply may have several rows, one per bin):

  case 1  ply has one row                 -> that bin
  case 2  all its rows have the same range -> first unused bin in the order Before-DW, Before-UW, After-DW, After-UW (relay 1..4),
          (ranges within --range-tol count as the same, e.g. 2.75-19.2 and 2.8-19.2)
  case 3  its rows have different ranges   -> the range read closest to one of its sheet ranges (at most
          --range-max-diff differing digits), then the first unused bin in that range
Every bin row is used once: a roll that would need an already used row is a DUPLICATE (terminal only, no light).
Type `reset` + Enter in the terminal to clear the used rows for the next set of rolls.
Not in master, range not readable, range matching no row, UNREADABLE: terminal only, lights unchanged.
The lit bin stays on until the next roll lights one; the same bin again is switched off and on.

Pi 5 (USB camera):  python3 rtracker_ocr_paddle_v5_queue_v8.py --master data/master.csv --source /dev/video0 --model models/rolls.hef --esp32-port /dev/ttyUSB0
PC (video file):    python rtracker_ocr_paddle_v5_queue_v8.py --master data/master.csv --source videos/cam0_onsite.mp4 --model models/rolls_v2.pt --roi 240,150,840,220
(omit --esp32-port for a dry run: results in the terminal only)

Test log: every run writes its own folder (never overwritten), test_logs/<source>__<date_time>/:
  run_info.json   source, date, script, all arguments, machine
  detections.csv  one row per detected box per frame (class, confidence, box, roll track id, detect ms)
  ocr_reads.csv   one row per OCR read (roll, field, orientation, read used + score, the other way up + score,
                  which way was used (0/180), counted as a vote, OCR ms, worker, crop image)
  crops/          every crop that was read, as PNG
  results.csv     one row per roll, same columns as --csv
  annotated.mp4   the annotated video (unless --out is given)
  summary.txt     totals and average speeds, written at the end (also on q / Ctrl-C)
--no-log turns it off (no crops saved; annotated video then goes to output/annotated_queue_v8.mp4).
"""
import argparse
import atexit
import csv
import json
import os
import platform
import re
import statistics
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from bin_light_controller import BinLightController
from detector_v2 import load_detector, rect_bounds, rect_points, PLY, RANGE, ROLL, TEXT
from ocr_paddle_v5 import align_rect, crop_rect, text_rotation
from ply_bin_lights import norm_ply
from rtracker_ocr_paddle_v5 import FPS, MAX_MISSED, iou, open_source, frames_15fps


class FreeReader:
    """PP-OCRv5 recognizer read with its whole dictionary: greedy CTC, no character restriction, no format check,
    no dot-to-dash fix (as logic_test_ocr_reading.py). Same preprocessing as ocr_paddle_v5.TextReader.
    One ONNX Runtime session, safe to run from several OCR worker threads at once."""

    def __init__(self, path, threads=1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        chars = self.sess.get_modelmeta().custom_metadata_map["character"].splitlines()
        n_out = self.sess.get_outputs()[0].shape[-1]
        n_out = n_out if isinstance(n_out, int) else len(chars) + 1
        self.chars = [""] + chars + [" "] * max(0, n_out - 1 - len(chars))  # 0 = CTC blank, extra slot = space

    def _read(self, crop):
        h, w = crop.shape[:2]
        new_w = min(320, max(16, int(48 * w / h)))
        img = cv2.resize(crop, (new_w, 48)).astype(np.float32)
        img = (img[:, :, ::-1] / 255.0 - 0.5) / 0.5  # BGR->RGB, normalize to [-1, 1]
        x = np.zeros((1, 3, 48, 320), np.float32)
        x[0, :, :, :new_w] = img.transpose(2, 0, 1)
        probs = self.sess.run(None, {self.inp: x})[0][0]
        best, conf = probs.argmax(1), probs.max(1)
        text, scores, prev = "", [], 0
        for k, p in zip(best, conf):
            if k != prev and k != 0:
                text += self.chars[k] if k < len(self.chars) else "?"
                scores.append(p)
            prev = k
        return text.strip(), float(np.mean(scores)) if scores else 0.0

    def read(self, crop, rotation=None):
        """-> (text, score, flipped_text, flipped_score, used). rotation 0/180 from the layout: one read, flipped
        values None. rotation None: read upright and turned 180 deg; used = 0 or 180, the higher score."""
        if crop.size == 0:
            return "", 0.0, None, None, rotation or 0
        if rotation is not None:
            text, score = self._read(crop if rotation == 0 else cv2.rotate(crop, cv2.ROTATE_180))
            return text, score, None, None, rotation
        text, score = self._read(crop)
        text2, score2 = self._read(cv2.rotate(crop, cv2.ROTATE_180))
        return text, score, text2, score2, 180 if score2 > score else 0


def ply_key(ply):
    """master.csv lookup key of a whole ply read: its digits ('85W' -> '85', '784-DW' -> '784'); '' = none."""
    return re.sub(r"\D", "", ply or "")


def text_field(text):
    """Field of a read from a single-class text box -> 'range', 'ply' or None (only logged, no vote).
    '.' or number-number -> range ('50.8-78.3', '31.3.37.1', '44-46'); digits, optionally with U/D W/B before or
    after them -> ply ('1511', '7841-DW', '85W')."""
    t = (text or "").strip().upper()
    if not re.search(r"\d", t):
        return None
    if "." in t or re.fullmatch(r"\d+-\d+", t):
        return "range"
    if re.fullmatch(r"(?:[UD]?[WB]-?)?\d+(?:-?[UD]?[WB])?", t):
        return "ply"
    return None


class Track:
    next_id = 1

    def __init__(self, rect):
        self.id, Track.next_id = Track.next_id, Track.next_id + 1
        self.rect, self.missed, self.job = rect, 0, None  # rotated rect (cx, cy, w, h, theta)


class RollJob:
    """One roll in the result queue. The main loop adds crops, the OCR workers take them (both under the lock)."""

    def __init__(self, seq, track_id):
        self.seq, self.track_id = seq, track_id
        self.first_seen = time.time()
        self.started = None     # when this roll became the next one to decide (--read-timeout counts from here)
        self.ocr_started = None  # first OCR read on this roll (may be before it reaches the head)
        self.crops = {"ply": deque(), "range": deque(), "text": deque(), "auto": deque()}  # waiting: (crop, rotation, frame)
        self.votes = {"ply": Counter(), "range": Counter()}  # every non-empty read, by text
        self.sure = {"ply": Counter(), "range": Counter()}  # votes scoring >= --fast-score
        self.last_frame = {"ply": {}, "range": {}}   # text -> latest video frame it was read on
        self.first_read = {"ply": {}, "range": {}}   # text -> seconds after first seen when first read
        self.confirmed_at = {"ply": {}, "range": {}}  # text -> time it reached --votes / fast confirm
        self.reads = 0          # OCR attempts on this roll
        self.busy = 0           # OCR reads in progress on this roll
        self.closed = False     # roll left the frame, no more crops coming
        self.ply = None         # chosen ply, the whole read (e.g. '72'), set when decided
        self.rule = ""          # how the ply was chosen
        self.rng = None         # chosen range read (case 3), set when decided
        self.result = None      # BinPicker.pick() result, or {"status": "unreadable"}
        self.decided_after = 0.0
        self.done = False       # printed; no more crops wanted

    def add(self, field, crop, rotation, frame=None, timing=None):
        """Every crop is kept (no cap, nothing dropped). timing: detection conf / times for the test log."""
        self.crops[field].append((crop, rotation, frame, timing or {}))

    def take(self, fields):
        """Oldest waiting crop of the first field that has one -> (field, crop, rotation, frame, timing), or None."""
        for f in fields:
            if self.crops[f]:
                return (f,) + self.crops[f].popleft()
        return None


def top(votes):
    return votes.most_common(1)[0] if votes else (None, 0)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def digits(value):
    """'78.90' -> '789', '45' -> '45': the digits of the number as printed, dots ignored (OCR often drops them)."""
    s = str(value).strip()
    try:
        s = f"{float(s):.3f}".rstrip("0").rstrip(".")
    except ValueError:
        pass
    return s.replace(".", "")


def range_diff(read, start, end):
    """Digits that differ between an OCR range 'a-b' and a sheet range, start and end compared separately."""
    a, _, b = read.partition("-")
    return edit_distance(digits(a), digits(start)) + edit_distance(digits(b), digits(end))


def same_range(r1, r2, tol):
    try:
        return abs(float(r1[0]) - float(r2[0])) <= tol + 1e-9 and abs(float(r1[1]) - float(r2[1])) <= tol + 1e-9
    except ValueError:
        return (r1[0], r1[1]) == (r2[0], r2[1])


class BinPicker:
    """master.csv -> ply -> range groups -> bin rows (Before-DW, Before-UW, After-DW, After-UW order). Remembers the rows already
    used until reset(); re-reads the file when it changes (used rows are kept, last good copy kept on error)."""

    def __init__(self, path, key_col, bin_col, bin_map, max_diff=2, tol=0.1):
        self.path, self.key_col, self.bin_col, self.bin_map = Path(path), key_col, bin_col, bin_map
        self.max_diff, self.tol = max_diff, tol
        self.mtime, self.plys, self.used, self.lock = None, {}, set(), threading.Lock()
        self._load()
        if not self.plys:
            sys.exit(f"no plys loaded from {self.path}")

    def _load(self):
        try:
            mtime = self.path.stat().st_mtime
            if mtime == self.mtime:
                return
            plys, seen, n = {}, Counter(), 0
            with open(self.path, newline="", encoding="utf-8-sig", errors="replace") as f:
                for line, r in enumerate(csv.DictReader(f), start=2):  # line = row number in the sheet
                    ply, bin_ = norm_ply(r[self.key_col] or ""), (r[self.bin_col] or "").strip()
                    if not ply or not bin_:
                        continue
                    try:
                        relay = int(float(bin_))  # plain relay number
                    except ValueError:
                        relay = self.bin_map.get(bin_)  # bin label such as Before-DW
                    start, end = (r.get("start") or "").strip(), (r.get("end") or "").strip()
                    key = (ply, start, end, bin_)
                    seen[key] += 1
                    row = {"line": line, "bin": bin_, "relay": relay, "start": start, "end": end,
                           "key": key + (seen[key],)}  # identity that survives a reload
                    groups = plys.setdefault(ply, [])
                    for g in groups:
                        if same_range((g["start"], g["end"]), (start, end), self.tol):
                            g["rows"].append(row)
                            break
                    else:
                        groups.append({"start": start, "end": end, "rows": [row]})
                    n += 1
        except (OSError, KeyError, csv.Error) as e:
            print(f"{self.path}: cannot read ({e!r}) -- keeping the previous copy")
            return
        for groups in plys.values():
            for g in groups:
                g["rows"].sort(key=lambda r: (r["relay"] if r["relay"] is not None else 99, r["line"]))
        self.plys, self.mtime = plys, mtime
        print(f"{self.path}: {n} rows, {len(plys)} plys loaded")

    def reset(self):
        with self.lock:
            n = len(self.used)
            self.used.clear()
        return n

    def needs_range(self, ply):
        with self.lock:
            self._load()
            return len(self.plys.get(norm_ply(ply), [])) > 1

    def in_sheet(self, ply):
        """Is this ply (lookup key) in the sheet? No file reload: cheap enough for every frame."""
        with self.lock:
            return bool(ply) and norm_ply(ply) in self.plys

    def ranges(self, ply):
        """[(start, end)] of the ply's range groups."""
        with self.lock:
            self._load()
            return [(g["start"], g["end"]) for g in self.plys.get(norm_ply(ply), [])]

    def _free(self, g):
        return [r for r in g["rows"] if r["key"] not in self.used]

    def pick(self, ply, rng=None):
        """-> dict with status: picked | duplicate | not_in_master | range_unreadable | range_no_match.
        rng (OCR range text) is only used when the ply has more than one range (case 3)."""
        with self.lock:
            self._load()
            groups = self.plys.get(norm_ply(ply))
            if groups is None:
                return {"status": "not_in_master"}
            diff = None
            if len(groups) == 1:
                g = groups[0]
                case = 1 if len(g["rows"]) == 1 else 2
            else:
                case = 3
                if not rng:
                    return {"status": "range_unreadable", "case": 3, "groups": groups}
                scored = [(range_diff(rng, x["start"], x["end"]), i) for i, x in enumerate(groups)]
                diff = min(scored)[0]
                tied = [groups[i] for d, i in scored if d == diff]
                if diff > self.max_diff:
                    return {"status": "range_no_match", "case": 3, "group": tied[0], "diff": diff, "groups": groups}
                # tie: the range whose first free bin comes first in the Before-DW, Before-UW, After-DW, After-UW order
                with_free = [x for x in tied if self._free(x)]
                g = min(with_free, key=lambda x: self.rank(self._free(x)[0])) if with_free else tied[0]
            free = self._free(g)
            if not free:
                return {"status": "duplicate", "case": case, "group": g, "diff": diff}
            row = free[0]
            self.used.add(row["key"])
            return {"status": "picked", "case": case, "group": g, "row": row, "diff": diff,
                    "nth": g["rows"].index(row) + 1}

    @staticmethod
    def rank(row):
        return row["relay"] if row["relay"] is not None else 99, row["line"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="ply -> bin sheet for this set of rolls, e.g. data/master.csv")
    ap.add_argument("--source", default="/dev/video0", help="camera index, /dev/videoN or video file")
    ap.add_argument("--model", default="models/rolls.hef", help=".hef on the Pi, .pt on a PC")
    ap.add_argument("--rec-model", default="models/v5/ppocr_rec_fine_tuned.onnx",
                    help="PP-OCRv5 text recognizer (stock: models/v5/ppocr_rec.onnx)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--exposure", type=int, default=None, help="manual exposure (100 us units), omit for auto")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--classes", default=None,
                    help="detector class names in training order, e.g. ply,roll (only needed for a .hef that is "
                         "neither ply,range,roll nor ply,roll)")
    ap.add_argument("--merge-text", action="store_true",
                    help="ply and range boxes are one text class (the reading decides the field), even if the "
                         "model has both classes")
    ap.add_argument("--roi", default=None, help="static ROI 'x,y,w,h' in source pixels; only this region is detected")
    ap.add_argument("--votes", type=int, default=3,
                    help="matching OCR reads needed to confirm a value (any non-empty read is a vote)")
    ap.add_argument("--fast-votes", type=int, default=2,
                    help="matching reads that each scored >= --fast-score also confirm a value (--votes: off)")
    ap.add_argument("--fast-score", type=float, default=0.95)
    ap.add_argument("--ocr-workers", type=int, default=2, help="OCR threads reading the queue")
    ap.add_argument("--ocr-threads", type=int, default=1, help="CPU threads per OCR read (ONNX Runtime)")
    ap.add_argument("--close-after", type=int, default=5,
                    help="frames a roll must be unseen before its job counts as having left the frame")
    ap.add_argument("--read-timeout", type=float, default=None,
                    help="seconds of OCR on a roll (from when it is the next roll to decide) before it is "
                         "UNREADABLE; omit to wait until the roll has left the frame and all its crops are read")
    ap.add_argument("--range-max-diff", type=int, default=5,
                    help="case 3: most differing digits (start + end) for a range to match a sheet row")
    ap.add_argument("--range-tol", type=float, default=0.1,
                    help="sheet ranges of one ply within this on start and end count as the same range (case 2)")
    ap.add_argument("--key-col", default="ply_no")
    ap.add_argument("--bin-col", default="no_of_ply")
    ap.add_argument("--esp32-port", default=None, help="ESP32 serial port, e.g. /dev/ttyUSB0 (omit = dry run)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--blink-gap", type=float, default=0.3, help="seconds off before re-lighting the same bin")
    ap.add_argument("--out", default=None, help="annotated video (default: annotated.mp4 in the run's test log "
                                                 "folder, or output/annotated_queue_v8.mp4 with --no-log)")
    ap.add_argument("--csv", default="output/readings_queue_v8.csv")
    ap.add_argument("--log-dir", default="test_logs", help="each run gets its own folder in here")
    ap.add_argument("--no-log", action="store_true", help="no test log folder for this run")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    for stream in (sys.stdout, sys.stderr):  # free reads may hold characters the console cannot show
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    run_dir = None
    if not args.no_log:  # test_logs/<source>__<date_time>/, a new folder for every run
        src = Path(args.source).stem if os.path.isfile(args.source) else "camera_" + Path(args.source).name
        run_dir = Path(args.log_dir) / f"{src}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        n = 2
        while run_dir.exists():
            run_dir = run_dir.with_name(run_dir.name.split("__run")[0] + f"__run{n}")
            n += 1
        run_dir.mkdir(parents=True)
        (run_dir / "crops").mkdir()
    if args.out is None:
        args.out = str(run_dir / "annotated.mp4") if run_dir else "output/annotated_queue_v8.mp4"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    lights = BinLightController(port=args.esp32_port, baud=args.baud)
    picker = BinPicker(args.master, args.key_col, args.bin_col, lights.bin_map, args.range_max_diff, args.range_tol)
    lights.open()
    detector = load_detector(args.model, args.conf, args.classes.split(",") if args.classes else None,
                             args.merge_text)
    single_text = TEXT in getattr(detector, "ids", [])
    print(f"detector: {args.model}  classes {getattr(detector, 'names', '?')}"
          + ("  -> single text class: the reading decides ply / range" if single_text else ""))
    reader = FreeReader(args.rec_model, threads=args.ocr_threads)  # one session, shared by the OCR workers
    print(f"recognizer: {args.rec_model}  ({args.ocr_workers} OCR workers x {args.ocr_threads} threads)")
    cap, is_file = open_source(args.source, args.width, args.height, args.exposure)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source}")

    new_csv = not os.path.exists(args.csv)
    csv_file = open(args.csv, "a", newline="", encoding="utf-8")  # free reads may hold any character
    log = csv.writer(csv_file)
    result_cols = ["timestamp", "seq", "track_id", "status", "case", "ply", "ply_lookup", "ply_rule", "ply_reads",
                   "range_read", "range_reads", "sheet_range", "digits_off", "bin", "relay", "sheet_row",
                   "seconds_after_first_seen", "queue_wait_s", "ply_first_read_s"]
    if new_csv:
        log.writerow(result_cols)

    # --- test log of this run ---
    t_start = time.time()
    stats = {"frames": 0, "det_ms": [], "classes": Counter(), "ocr_ms": [], "votes": 0, "results": Counter(),
             "lines": [], "fast": 0, "queue_wait": [], "lit": [], "crops": 0}
    det_log = ocr_log = res_log = None
    log_files = []
    if run_dir:
        info = {"source": args.source, "is_file": is_file, "started": datetime.now().isoformat(timespec="seconds"),
                "script": Path(__file__).name, "args": vars(args), "machine": platform.node(),
                "platform": platform.platform(), "processor": platform.processor(),
                "source_fps": cap.get(cv2.CAP_PROP_FPS), "source_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "processing_fps": FPS}
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        files = {name: open(run_dir / name, "w", newline="", encoding="utf-8-sig")  # -sig: Excel shows any symbol
                 for name in ("detections.csv", "ocr_reads.csv", "results.csv")}
        log_files = list(files.values())
        det_log, ocr_log, res_log = (csv.writer(files[n]) for n in ("detections.csv", "ocr_reads.csv", "results.csv"))
        det_log.writerow(["frame", "video_time_s", "elapsed_s", "class", "conf", "cx", "cy", "w", "h", "theta",
                          "roll_track_id", "detect_ms"])
        ocr_log.writerow(["elapsed_s", "seq", "roll_track_id", "crop_frame", "field", "orientation", "text", "score",
                          "text_other_way", "score_other_way", "used", "counted_as_vote", "ocr_ms", "worker", "crop",
                          # timing of this box: detection confidence, detection ms of its frame, crops waiting in
                          # the whole queue when it joined (itself included), ms it waited before an OCR worker
                          # took it, and total ms from the start of its frame's detection to the end of its read
                          "det_conf", "detect_ms", "queue_len_at_join", "queue_ms", "total_ms"])
        res_log.writerow(result_cols)
        print(f"test log: {run_dir}")

    def finish_log():
        """summary.txt + close the log files; runs once, at the end or on q / Ctrl-C."""
        if not run_dir or not log_files or log_files[0].closed:
            return
        for f in log_files:
            f.close()
        ms = lambda xs: (f"mean {statistics.mean(xs):.1f} ms, median {statistics.median(xs):.1f} ms"
                         if xs else "none")
        elapsed = time.time() - t_start
        det = stats["det_ms"][1:]  # first frame = model warm-up, reported on its own
        warm = f"  (first frame {stats['det_ms'][0]:.0f} ms warm-up, left out)" if stats["det_ms"] else ""
        lines = [f"source      {args.source}",
                 f"run folder  {run_dir}",
                 f"model       {args.model}   recognizer {args.rec_model}   roi {args.roi or 'none'}   "
                 f"conf {args.conf}   votes {args.votes} (any non-empty read, no score threshold)",
                 f"classes     {getattr(detector, 'names', '?')}"
                 + ("  (single text class: field from the reading)" if single_text else ""),
                 "reading     whole dictionary, no character restriction, no format check, no dot-to-dash fix; "
                 "every box on every frame, oldest crop first, never stopped by a confirmation; clean crops",
                 "decision    early (green) when a confirmed ply is in the sheet (case 3: + a confirmed range within "
                 "--range-max-diff); else after the roll has left (or --read-timeout) and all its crops are read: "
                 "ply = most recent read in the sheet, else latest confirmed; case 3 range = read closest to a "
                 "sheet range",
                 f"frames      {stats['frames']} at {FPS} fps in {elapsed:.1f} s "
                 f"({stats['frames'] / max(elapsed, 1e-6):.1f} fps processed)",
                 f"detection   {ms(det)}" + (f"  (~{1000 / statistics.mean(det):.0f} fps)" if det else "") + warm,
                 "detections  " + (", ".join(f"{k} {v}" for k, v in stats["classes"].most_common()) or "none"),
                 f"ocr         {len(stats['ocr_ms'])} reads, {stats['votes']} counted as votes, {ms(stats['ocr_ms'])}"
                 f", {stats['crops']} crops saved",
                 f"ocr workers {args.ocr_workers} x {args.ocr_threads} threads, busy "
                 f"{sum(stats['ocr_ms']) / 1000 / max(elapsed * args.ocr_workers, 1e-6):.0%} of the time   "
                 f"fast votes {args.fast_votes} at >= {args.fast_score} ({stats['fast']} values fast-confirmed)   "
                 f"close after {args.close_after} frames",
                 "queue wait  " + (f"mean {statistics.mean(stats['queue_wait']):.2f} s (first seen -> first OCR read)"
                                   if stats["queue_wait"] else "none")
                 + (f"   lit mean {statistics.mean(stats['lit']):.2f} s after first seen" if stats["lit"] else ""),
                 f"rolls       {sum(stats['results'].values())}: "
                 + (", ".join(f"{k} {v}" for k, v in stats["results"].most_common()) or "none"),
                 ""] + stats["lines"]
        (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
        print(f"test log written to {run_dir}")

    atexit.register(finish_log)

    jobs = deque()  # RollJobs in arrival order; jobs[0] is the next one to report
    cond = threading.Condition()
    text_only = getattr(detector, "text_only", False)
    last_relay = None

    def console():
        """`reset` + Enter: forget the used bins, the next roll starts a new set."""
        for line in sys.stdin:
            if line.strip().lower() == "reset":
                print(f"--- reset: {picker.reset()} used bin rows cleared, next roll starts a new set ---")

    if sys.stdin:
        threading.Thread(target=console, daemon=True).start()

    def light(relay):
        nonlocal last_relay
        if relay == last_relay:  # same bin as the previous roll: switch off first so it visibly re-lights
            lights.all_off()
            time.sleep(args.blink_gap)
        lights.set_relay(relay)
        last_relay = relay

    def reads_txt(counter):
        """Counter -> '7x6 72x1' (most read first)."""
        return " ".join(f"{t}x{n}" for t, n in counter.most_common())

    def current_ply(job):
        """(under the lock) Ply the roll would get now: the most recent in-sheet read, else the most read text."""
        in_sheet = [t for t in job.votes["ply"] if picker.in_sheet(ply_key(t))]
        if in_sheet:
            return max(in_sheet, key=lambda t: (job.last_frame["ply"][t], job.votes["ply"][t]))
        return top(job.votes["ply"])[0]

    def choose_ply(job):
        """(under the lock) -> (ply, rule). The most recent read whose digits are in the sheet; else the latest
        confirmed value (--votes or fast confirm); else (None, 'unreadable')."""
        in_sheet = [t for t in job.votes["ply"] if picker.in_sheet(ply_key(t))]
        if in_sheet:
            return max(in_sheet, key=lambda t: (job.last_frame["ply"][t], job.votes["ply"][t])), "latest_in_sheet"
        if job.confirmed_at["ply"]:
            return max(job.confirmed_at["ply"], key=job.confirmed_at["ply"].get), "confirmed_not_in_sheet"
        return None, "unreadable"

    def choose_range(job, ranges):
        """(under the lock) Case 3: the range read closest to one of the ply's sheet ranges (fewest differing
        digits), even if read once; tie -> most reads, then the latest. None if nothing was read."""
        reads = job.votes["range"]
        if not reads or not ranges:
            return None
        return min(reads, key=lambda t: (min(range_diff(t, s, e) for s, e in ranges), -reads[t],
                                         -job.last_frame["range"][t]))

    def early_plan(job):
        """(under the lock) Early decision while reading goes on: a CONFIRMED ply whose digits are in the sheet
        (the one confirmed last); case 3 also needs a confirmed range within --range-max-diff of one of its sheet
        ranges (the closest; tie -> most reads). -> {"ply", "rule", "rng"} or None (keep reading)."""
        plys = [t for t in job.confirmed_at["ply"] if picker.in_sheet(ply_key(t))]
        if not plys:
            return None
        ply = max(plys, key=job.confirmed_at["ply"].get)
        ranges = picker.ranges(ply_key(ply))
        rng = None
        if len(ranges) > 1:  # case 3
            near = {t: min(range_diff(t, s, e) for s, e in ranges) for t in job.confirmed_at["range"]}
            near = {t: d for t, d in near.items() if d <= args.range_max_diff}
            if not near:
                return None
            rng = min(near, key=lambda t: (near[t], -job.votes["range"][t]))
        return {"ply": ply, "rule": "confirmed_in_sheet", "rng": rng}

    def decide(job, plan=None):
        """Pick the bin and light it. plan: the early decision (early_plan); None: choose the ply (and case 3
        range) from all the roll's reads, after all its crops are read. Only the commit thread calls this,
        one roll at a time in seq order."""
        if plan:
            ply, rule, rng = plan["ply"], plan["rule"], plan["rng"]
        else:
            with cond:
                ply, rule = choose_ply(job)
            rng = None
        key = ply_key(ply)
        if ply is None:
            res = {"status": "unreadable"}
        elif not key:  # no digits in the ply read: nothing to look up
            res = {"status": "not_in_master"}
        else:
            ranges = picker.ranges(key)
            if len(ranges) > 1 and not plan:  # case 3
                with cond:
                    rng = choose_range(job, ranges)
            res = picker.pick(key, rng)
        with cond:
            job.ply, job.rule, job.rng = ply, rule, rng
            job.result = res
            job.decided_after = time.time() - job.first_seen
            cond.notify_all()
        if res["status"] == "picked" and res["row"]["relay"] is not None:
            light(res["row"]["relay"])

    def report(job):
        """One terminal line + CSV row per roll, in seq order."""
        res, head = job.result, f"[seq {job.seq}] roll #{job.track_id}"
        status, case, g = res["status"], res.get("case", ""), res.get("group")
        ply_reads, range_reads = reads_txt(job.votes["ply"]), reads_txt(job.votes["range"])
        rng = job.rng if job.rng is not None else top(job.votes["range"])[0]
        rng_txt = "?" if rng is None else f"{rng} (read {job.votes['range'][rng]}x)"
        sheet = f"{g['start']}-{g['end']}" if g else ""
        diff = res.get("diff")
        off = "" if diff is None else f", {diff} digit{'s' if diff != 1 else ''} off"
        row = res.get("row")
        why = f"  [ply reads: {ply_reads or 'none'}]"
        if status == "unreadable":
            print(f"{head}  UNREADABLE  (no ply read in {picker.path.name} and none confirmed, "
                  f"{job.reads} OCR reads){why}")
        elif status == "not_in_master":
            key = ply_key(job.ply)
            looked = f"looked up as {key}" if key else "no digits to look up"
            print(f"{head}  ply {job.ply} ({looked}) not in {picker.path.name}  |  range {rng_txt}{why}")
        elif status == "range_unreadable":
            options = " / ".join(f"{x['start']}-{x['end']}" for x in res["groups"])
            print(f"{head}  ply {job.ply}  range NOT READABLE -- cannot choose between {options}  -- no light{why}")
        elif status == "range_no_match":
            print(f"{head}  ply {job.ply}  range {rng_txt} does not match any row in {picker.path.name} "
                  f"(closest {sheet}{off}; range reads: {range_reads})  -- no light{why}")
        elif status == "duplicate":
            bins = ", ".join(r["bin"] for r in g["rows"])
            print(f"{head}  ply {job.ply}  DUPLICATE -- bin{'s' if len(g['rows']) > 1 else ''} {bins} for range "
                  f"{sheet} already used  |  range {rng_txt}  (case {case})  -- no light{why}")
        else:
            sent = f"BIN {row['relay']}" if row["relay"] is not None else f"no relay mapped for {row['bin']}"
            if row["relay"] is not None and not lights.enabled:
                sent += " (dry run)"
            print(f"{head}  ply {job.ply} -> bin {row['bin']}  {sent}  |  range {rng_txt}  (sheet {sheet}, case {case}, "
                  f"{res['nth']} of {len(g['rows'])} for this range{off})  |  lit {job.decided_after:.1f} s after "
                  f"first seen{why}")
        first = job.first_read["ply"].get(job.ply)
        result = [datetime.now().isoformat(timespec="seconds"), job.seq, job.track_id, status, case,
                  job.ply or "", ply_key(job.ply), job.rule, ply_reads, rng or "", range_reads, sheet,
                  "" if diff is None else diff,
                  row["bin"] if row else "", row["relay"] if row and row["relay"] is not None else "",
                  row["line"] if row else "", round(job.decided_after or time.time() - job.first_seen, 1),
                  "" if job.ocr_started is None else round(job.ocr_started - job.first_seen, 2),
                  "" if first is None else round(first, 2)]
        log.writerow(result)
        csv_file.flush()
        if res_log and not log_files[0].closed:
            res_log.writerow(result)
            log_files[2].flush()
        stats["results"][status] += 1
        if job.ocr_started is not None:
            stats["queue_wait"].append(job.ocr_started - job.first_seen)
        if row and row["relay"] is not None:
            stats["lit"].append(job.decided_after)
        stats["lines"].append(f"{head}  {status}  ply {job.ply or '?'} ({job.rule})  range {rng_txt}"
                              + (f"  -> bin {row['bin']}" if row else "")
                              + f"  ({job.reads} OCR reads; ply reads {ply_reads or 'none'}; "
                                f"range reads {range_reads or 'none'})")

    def timed_out(job):
        return args.read_timeout is not None and job.started is not None \
            and time.time() - job.started >= args.read_timeout

    def next_item():
        """Read ahead: the oldest waiting crop of the first undecided job in seq order (ply before range)."""
        for job in jobs:
            if job.result is None:
                item = job.take(("ply", "range", "text", "auto"))
                if item:
                    return job, item
        return None, None

    def ocr_worker(worker):
        while True:
            with cond:
                job, item = next_item()
                while item is None:
                    cond.wait(0.05)
                    job, item = next_item()
                job.busy += 1
                if job.ocr_started is None:
                    job.ocr_started = time.time()
                t_taken = time.perf_counter()

            field, crop, rotation, crop_frame, timing = item
            t0 = time.perf_counter()
            text, score, text2, score2, used = reader.read(crop, rotation)
            ocr_ms = (time.perf_counter() - t0) * 1000
            if rotation is None and used == 180:  # way up unknown: the higher-scoring way is used
                text, score, text2, score2 = text2, score2, text, score
            # every non-empty read is a vote (no score threshold); "auto" boxes are only logged;
            # single-class "text" boxes vote for the field their reading shows
            voted = text_field(text) if field == "text" else field
            vote = bool(text) and voted in ("ply", "range")
            crop_name = ""
            if run_dir:
                with cond:
                    stats["crops"] += 1
                    crop_name = f"crops/s{job.seq:03d}_f{crop_frame:05d}_{field}_{stats['crops']:06d}.png"
                # saved the way it was read when the way up is known, as cut out when both ways were read
                cv2.imwrite(str(run_dir / crop_name), crop if rotation in (None, 0) else cv2.rotate(crop, cv2.ROTATE_180))
            with cond:
                job.reads += 1
                job.busy -= 1
                stats["ocr_ms"].append(ocr_ms)
                stats["votes"] += vote
                if ocr_log and not log_files[0].closed:
                    # text/score = the read used (used = 0 or 180 deg); *_other_way = the other way up, only when
                    # the way up was unknown
                    ocr_log.writerow([f"{time.time() - t_start:.2f}", job.seq, job.track_id, crop_frame,
                                      field if field != "text" else f"text:{voted or '-'}",
                                      "layout" if rotation is not None else "unknown", text, f"{score:.3f}",
                                      "" if text2 is None else text2, "" if score2 is None else f"{score2:.3f}",
                                      used, int(vote), f"{ocr_ms:.1f}", worker, crop_name,
                                      f"{timing.get('conf', 0):.3f}", f"{timing.get('det_ms', 0):.1f}",
                                      timing.get("queue_len", ""),
                                      f"{(t_taken - timing['t_join']) * 1000:.1f}" if "t_join" in timing else "",
                                      f"{(time.perf_counter() - timing['t_det']) * 1000:.1f}" if "t_det" in timing else ""])
                    log_files[1].flush()
                if vote:
                    job.votes[voted][text] += 1
                    job.last_frame[voted][text] = max(crop_frame, job.last_frame[voted].get(text, 0))
                    job.first_read[voted].setdefault(text, time.time() - job.first_seen)
                    if score >= args.fast_score:
                        job.sure[voted][text] += 1
                    if text not in job.confirmed_at[voted]:
                        if job.votes[voted][text] >= args.votes:
                            job.confirmed_at[voted][text] = time.time()
                        elif job.sure[voted][text] >= args.fast_votes:
                            job.confirmed_at[voted][text] = time.time()
                            stats["fast"] += 1
                cond.notify_all()

    def ready_to_decide(job):
        """(under the lock) The roll has left the frame (or --read-timeout ran out) and all its crops are read."""
        return not job.busy and not any(job.crops.values()) and (job.closed or timed_out(job))

    def committer():
        """Decides, lights and reports rolls strictly in seq order."""
        while True:
            with cond:
                while True:
                    if jobs and jobs[0].result is not None and not jobs[0].busy:  # (reads in flight finish first)
                        job = jobs.popleft()
                        job.done = True
                        action = "report"
                        break
                    job = next((j for j in jobs if j.result is None), None)
                    if job is not None:
                        if job.started is None:
                            job.started = time.time()  # it is the next roll to decide: --read-timeout starts
                        plan = early_plan(job)  # confirmed + in the sheet: decide now, even while in view
                        if plan or ready_to_decide(job):
                            action = "decide"
                            break
                    cond.wait(0.05)
            if action == "report":
                report(job)
            else:
                decide(job, plan)

    for w in range(1, max(1, args.ocr_workers) + 1):
        threading.Thread(target=ocr_worker, args=(w,), daemon=True).start()
    threading.Thread(target=committer, daemon=True).start()

    tracks, writer, frame_no, t_fps, fps, seq = [], None, 0, time.time(), 0.0, 0
    for frame in frames_15fps(cap, is_file):
        frame_no += 1
        fh, fw = frame.shape[:2]
        clean = frame.copy()  # crops are cut from this, so nothing drawn on `frame` gets into them
        t0 = time.perf_counter()
        if roi:  # detect inside the ROI only (more pixels on the text), then shift back to frame coords
            rx, ry = max(0, roi[0]), max(0, roi[1])
            rw, rh = min(roi[2], fw - rx), min(roi[3], fh - ry)
            dets = [(c, s, (r[0] + rx, r[1] + ry, r[2], r[3], r[4]))
                    for c, s, r in detector.detect(frame[ry:ry + rh, rx:rx + rw])]
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 1)
        else:
            dets = detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        stats["frames"] += 1
        stats["det_ms"].append(det_ms)
        rolls = [d[2] for d in dets if d[0] == ROLL]
        texts = [d for d in dets if d[0] in (PLY, RANGE, TEXT)]

        # --- track rolls (greedy IoU matching on the rotated boxes' outer bounds) ---
        unmatched = list(range(len(rolls)))
        roll_track = {}  # index in rolls -> Track, for the test log
        for t in tracks:
            best = max(unmatched, key=lambda j: iou(rect_bounds(t.rect), rect_bounds(rolls[j])), default=None)
            if best is not None and iou(rect_bounds(t.rect), rect_bounds(rolls[best])) > 0.3:
                t.rect, t.missed = rolls[best], 0
                unmatched.remove(best)
                roll_track[best] = t
            else:
                t.missed += 1
        gone = [t for t in tracks if t.missed >= min(args.close_after, MAX_MISSED + 1) and t.job and not t.job.closed]
        if gone:  # roll left the frame: its job gets no more crops (the track is kept up to MAX_MISSED frames)
            with cond:
                for t in gone:
                    t.job.closed = True
                cond.notify_all()
        new = {j: Track(rolls[j]) for j in unmatched}
        roll_track.update(new)
        tracks = [t for t in tracks if t.missed <= MAX_MISSED] + list(new.values())

        # --- assign text boxes to the roll whose rotated box contains their centre ---
        roll_texts = {}
        text_track = {}  # id(text detection) -> roll track id, for the test log
        for d in texts:
            for t in tracks:
                if t.missed == 0 and cv2.pointPolygonTest(rect_points(t.rect), d[2][:2], False) >= 0:
                    roll_texts.setdefault(t, []).append(d)
                    text_track[id(d)] = t.id
                    break

        # --- test log: one row per detected box ---
        names = {PLY: "ply", RANGE: "range", ROLL: "roll", TEXT: "text"}
        n_roll = 0
        for d in dets:
            c, s, r = d
            stats["classes"][names.get(c, c)] += 1
            if c == ROLL:
                owner, n_roll = roll_track[n_roll].id, n_roll + 1
            else:
                owner = text_track.get(id(d), "")
            if det_log:
                det_log.writerow([frame_no, f"{(frame_no - 1) / FPS:.2f}" if is_file else "",
                                  f"{time.time() - t_start:.2f}", names.get(c, c), f"{s:.3f}",
                                  *(f"{v:.1f}" for v in r[:4]), f"{r[4]:.3f}", owner, f"{det_ms:.1f}"])

        # --- text orientation from layout (ply above start-end), then every straightened crop into the job ---
        for t, ds in roll_texts.items():
            rng = [d[2] for d in ds if d[0] == RANGE]
            if rng:  # point every text box of this roll the same way as the range box
                ds = [(c, s, align_rect(r, rng[0][4])) for c, s, r in ds]
            ply = [d[2] for d in ds if d[0] == PLY]
            rotation = text_rotation(ply[0], rng[0]) if ply and rng else None  # None: OCR tries both ways
            for cls, conf, rect in ds:
                field = {PLY: "ply", RANGE: "range", TEXT: "auto" if text_only else "text"}[cls]
                job = t.job
                if job is None or job.result is None:  # every box on every frame until the roll is decided
                    crop = crop_rect(clean, rect)
                    if crop.size:
                        with cond:
                            if t.job is None:  # first text on this roll: it joins the queue
                                seq += 1
                                t.job = RollJob(seq, t.id)
                                jobs.append(t.job)
                            if t.job.closed and t.job.result is None:  # seen again before it was decided
                                t.job.closed = False
                            waiting = sum(len(q) for j in jobs for q in j.crops.values()) + 1
                            t.job.add(field, crop, rotation, frame_no,
                                      {"conf": conf, "det_ms": det_ms, "t_det": t0, "queue_len": waiting,
                                       "t_join": time.perf_counter()})
                            cond.notify_all()
                if not text_only:
                    cv2.polylines(frame, [np.int32(rect_points(rect))], True, (255, 200, 0), 1)

        # --- draw rolls with their queue state ---
        for t in tracks:
            if t.missed or (text_only and t.job is None):  # text-only: hide text that isn't a reading
                continue
            job = t.job
            if job is None:
                label, color = f"#{t.id}", (0, 165, 255)
            else:
                with cond:
                    ply = job.ply or current_ply(job)  # chosen ply, or what it would be now
                    rng = job.rng or top(job.votes["range"])[0]
                    res = job.result
                label = f"#{t.id} seq {job.seq} Ply {ply or '?'} | {rng or '?'}"
                if res and res["status"] == "picked":
                    label += f" -> {res['row']['bin']}"
                    color = (0, 200, 0)
                elif res:
                    color = (0, 0, 255)
                else:
                    color = (0, 165, 255)
            x1, y1 = map(int, rect_bounds(t.rect)[:2])
            cv2.polylines(frame, [np.int32(rect_points(t.rect))], True, color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            ty = max(th + 8, y1)
            cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 6, ty), color, -1)
            cv2.putText(frame, label, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_fps, 1e-6)
        t_fps = now
        with cond:
            waiting = len(jobs)
        cv2.putText(frame, f"{fps:.1f} fps  queue {waiting}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (fw, fh))
        writer.write(frame)
        if not args.no_show:
            cv2.imshow("RTracker OCR queue v8", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # source ended: close the rolls still in view and let the queue finish in order
    with cond:
        for t in tracks:
            if t.job:
                t.job.closed = True
        cond.notify_all()
    try:
        while True:
            with cond:
                if not jobs:
                    break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    finish_log()
    cap.release()
    if writer:
        writer.release()
    csv_file.close()
    lights.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
