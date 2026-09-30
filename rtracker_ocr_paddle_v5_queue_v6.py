"""Roll tracking + OCR (PP-OCRv5 mobile) with an ordered result queue that drives the ESP32 bin lights -- queue v6.
queue_v5 (read ahead, --ocr-workers, fast confirm, --close-after, in-order commit thread) merged with the reading
logic of logic_test_ocr_reading.py:
  every box       every ply / range box on a roll is cropped and queued on EVERY frame (no --snap-every, no crop
                  cap, no blur ranking); crops are read oldest first. A roll stops queuing ply crops once its ply
                  is confirmed, and all crops once it is reported.
  free reading    the recognizer (default: the fine-tuned model) is read with its whole dictionary: no per-field
                  character restriction, no format check, no dot-to-dash fix. The whole read is kept as the value
                  (ply '85W' stays '85W'); only the master.csv lookup uses the ply's digits ('85W' -> '85').
                  A ply read with no digits cannot be looked up (not in master).
  no threshold    every non-empty read is a vote, whatever its score (--votes 3 confirm). Fast confirm stays:
                  --fast-votes (2) matching reads that each scored >= --fast-score (0.95) also confirm.
  orientation     way up from the layout (ply above range); when unknown, the crop is read both ways, both reads
                  are logged and the higher-scoring one is used.
  auto boxes      (text-only mode, --model ppocr_det.onnx) are read and logged but never vote: without a format
                  check they cannot be told apart as ply or range.
  crops           every crop that is read is saved to <test log>/crops/ and named in ocr_reads.csv.
--read-timeout counts from when a roll becomes the next one to be decided, and restarts once a case 3 ply is
confirmed. When it expires, the crops already waiting are still read first.
How a confirmed ply picks its bin in master.csv
(a ply may have several rows, one per bin):

  case 1  ply has one row                 -> that bin, lit as soon as the ply is confirmed
  case 2  all its rows have the same range -> first unused bin in the order Before-DW, Before-UW, After-DW, After-UW (relay 1..4),
          (ranges within --range-tol count as the same, e.g. 2.75-19.2 and 2.8-19.2)
  case 3  its rows have different ranges   -> waits for the roll's range reading (best reading so far), picks the
          sheet range with the fewest differing digits (at most --range-max-diff), then the first unused bin in it
Every bin row is used once: a roll that would need an already used row is a DUPLICATE (terminal only, no light).
Type `reset` + Enter in the terminal to clear the used rows for the next set of rolls.
Not in master, range not readable, range matching no row, UNREADABLE: terminal only, lights unchanged.
The lit bin stays on until the next roll lights one; the same bin again is switched off and on.

Pi 5 (USB camera):  python3 rtracker_ocr_paddle_v5_queue_v6.py --master data/master.csv --source /dev/video0 --model models/rolls.hef --esp32-port /dev/ttyUSB0
PC (video file):    python rtracker_ocr_paddle_v5_queue_v6.py --master data/master.csv --source videos/cam0_onsite.mp4 --model models/rolls_v2.pt --roi 240,150,840,220
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
--no-log turns it off (no crops saved; annotated video then goes to output/annotated_queue_v6.mp4).
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
from detector import load_detector, rect_bounds, rect_points, PLY, RANGE, ROLL, TEXT
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
        self.crops = {"ply": deque(), "range": deque(), "auto": deque()}  # waiting: (crop, rotation, frame)
        self.votes = {"ply": Counter(), "range": Counter()}
        self.sure = {"ply": Counter(), "range": Counter()}  # votes scoring >= --fast-score
        self.reads = 0          # OCR attempts on this roll
        self.busy = 0           # OCR reads in progress on this roll
        self.closed = False     # roll left the frame, no more crops coming
        self.ply = None         # confirmed ply, the whole read (e.g. '85W')
        self.needs_range = None  # case 3: the range decides the bin (set with ply)
        self.ply_after = None   # seconds from first seen to ply confirmed
        self.result = None      # BinPicker.pick() result, or {"status": "unreadable"}
        self.decided_after = 0.0
        self.done = False       # printed; no more crops wanted

    def add(self, field, crop, rotation, frame=None):
        """Every crop is kept (no cap, nothing dropped)."""
        self.crops[field].append((crop, rotation, frame))

    def take(self, fields):
        """Oldest waiting crop of the first field that has one -> (field, crop, rotation, frame), or None."""
        for f in fields:
            if self.crops[f]:
                return (f,) + self.crops[f].popleft()
        return None


def top(votes):
    return votes.most_common(1)[0] if votes else (None, 0)


def confirmed(job, field, votes, fast_votes):
    """Confirmed value of a field: --votes matching reads, or --fast-votes matching high-score reads. Else None."""
    text, n = top(job.votes[field])
    if n >= votes:
        return text
    text, n = top(job.sure[field])
    return text if n >= fast_votes else None


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
    ap.add_argument("--range-max-diff", type=int, default=2,
                    help="case 3: most differing digits (start + end) for a range to match a sheet row")
    ap.add_argument("--range-tol", type=float, default=0.1,
                    help="sheet ranges of one ply within this on start and end count as the same range (case 2)")
    ap.add_argument("--key-col", default="ply_no")
    ap.add_argument("--bin-col", default="no_of_ply")
    ap.add_argument("--esp32-port", default=None, help="ESP32 serial port, e.g. /dev/ttyUSB0 (omit = dry run)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--blink-gap", type=float, default=0.3, help="seconds off before re-lighting the same bin")
    ap.add_argument("--out", default=None, help="annotated video (default: annotated.mp4 in the run's test log "
                                                 "folder, or output/annotated_queue_v6.mp4 with --no-log)")
    ap.add_argument("--csv", default="output/readings_queue_v6.csv")
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
        args.out = str(run_dir / "annotated.mp4") if run_dir else "output/annotated_queue_v6.mp4"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    lights = BinLightController(port=args.esp32_port, baud=args.baud)
    picker = BinPicker(args.master, args.key_col, args.bin_col, lights.bin_map, args.range_max_diff, args.range_tol)
    lights.open()
    detector = load_detector(args.model, args.conf)
    reader = FreeReader(args.rec_model, threads=args.ocr_threads)  # one session, shared by the OCR workers
    print(f"recognizer: {args.rec_model}  ({args.ocr_workers} OCR workers x {args.ocr_threads} threads)")
    cap, is_file = open_source(args.source, args.width, args.height, args.exposure)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source}")

    new_csv = not os.path.exists(args.csv)
    csv_file = open(args.csv, "a", newline="", encoding="utf-8")  # free reads may hold any character
    log = csv.writer(csv_file)
    result_cols = ["timestamp", "seq", "track_id", "status", "case", "ply", "ply_lookup", "range_read", "sheet_range",
                   "digits_off", "bin", "relay", "sheet_row", "seconds_after_first_seen", "queue_wait_s",
                   "ply_confirmed_s"]
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
                          "text_other_way", "score_other_way", "used", "counted_as_vote", "ocr_ms", "worker", "crop"])
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
                 "reading     whole dictionary, no character restriction, no format check, no dot-to-dash fix; "
                 "every box on every frame, oldest crop first",
                 f"frames      {stats['frames']} at {FPS} fps in {elapsed:.1f} s "
                 f"({stats['frames'] / max(elapsed, 1e-6):.1f} fps processed)",
                 f"detection   {ms(det)}" + (f"  (~{1000 / statistics.mean(det):.0f} fps)" if det else "") + warm,
                 "detections  " + (", ".join(f"{k} {v}" for k, v in stats["classes"].most_common()) or "none"),
                 f"ocr         {len(stats['ocr_ms'])} reads, {stats['votes']} counted as votes, {ms(stats['ocr_ms'])}"
                 f", {stats['crops']} crops saved",
                 f"ocr workers {args.ocr_workers} x {args.ocr_threads} threads, busy "
                 f"{sum(stats['ocr_ms']) / 1000 / max(elapsed * args.ocr_workers, 1e-6):.0%} of the time   "
                 f"fast votes {args.fast_votes} at >= {args.fast_score} ({stats['fast']} values confirmed early)   "
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

    def decide(job, rng=None, unreadable=False):
        """Pick the bin and light it. Only the commit thread calls this, one roll at a time in seq order."""
        if unreadable:
            res = {"status": "unreadable"}
        elif not ply_key(job.ply):  # no digits in the ply read: nothing to look up
            res = {"status": "not_in_master"}
        else:
            res = picker.pick(ply_key(job.ply), rng)
        with cond:
            job.result = res
            job.decided_after = time.time() - job.first_seen
            cond.notify_all()
        if res["status"] == "picked" and res["row"]["relay"] is not None:
            light(res["row"]["relay"])

    def report(job):
        """One terminal line + CSV row per roll, in seq order."""
        res, head = job.result, f"[seq {job.seq}] roll #{job.track_id}"
        status, case, g = res["status"], res.get("case", ""), res.get("group")
        rng = confirmed(job, "range", args.votes, args.fast_votes)
        rng_txt = rng or (f"{top(job.votes['range'])[0]} (unconfirmed)" if job.votes["range"] else "?")
        rng = rng or top(job.votes["range"])[0]
        sheet = f"{g['start']}-{g['end']}" if g else ""
        diff = res.get("diff")
        off = "" if diff is None else f", {diff} digit{'s' if diff != 1 else ''} off"
        row = res.get("row")
        if status == "unreadable":
            guess, votes = top(job.votes["ply"])
            hint = f"best guess ply {guess} ({votes}/{args.votes} reads)" if guess else "no ply read"
            print(f"{head}  UNREADABLE  ({hint}, {job.reads} OCR reads)")
        elif status == "not_in_master":
            key = ply_key(job.ply)
            looked = f"looked up as {key}" if key else "no digits to look up"
            print(f"{head}  ply {job.ply} ({looked}) not in {picker.path.name}  |  range {rng_txt}")
        elif status == "range_unreadable":
            options = " / ".join(f"{x['start']}-{x['end']}" for x in res["groups"])
            print(f"{head}  ply {job.ply}  range NOT READABLE -- cannot choose between {options}  -- no light")
        elif status == "range_no_match":
            print(f"{head}  ply {job.ply}  range {rng_txt} does not match any row in {picker.path.name} "
                  f"(closest {sheet}{off})  -- no light")
        elif status == "duplicate":
            bins = ", ".join(r["bin"] for r in g["rows"])
            print(f"{head}  ply {job.ply}  DUPLICATE -- bin{'s' if len(g['rows']) > 1 else ''} {bins} for range "
                  f"{sheet} already used  |  range {rng_txt}  (case {case})  -- no light")
        else:
            sent = f"BIN {row['relay']}" if row["relay"] is not None else f"no relay mapped for {row['bin']}"
            if row["relay"] is not None and not lights.enabled:
                sent += " (dry run)"
            print(f"{head}  ply {job.ply} -> bin {row['bin']}  {sent}  |  range {rng_txt}  (sheet {sheet}, case {case}, "
                  f"{res['nth']} of {len(g['rows'])} for this range{off})  |  lit {job.decided_after:.1f} s after first seen")
        result = [datetime.now().isoformat(timespec="seconds"), job.seq, job.track_id, status, case,
                  job.ply or "", ply_key(job.ply), rng or "", sheet, "" if diff is None else diff,
                  row["bin"] if row else "", row["relay"] if row and row["relay"] is not None else "",
                  row["line"] if row else "", round(job.decided_after or time.time() - job.first_seen, 1),
                  "" if job.ocr_started is None else round(job.ocr_started - job.first_seen, 2),
                  "" if job.ply_after is None else round(job.ply_after, 2)]
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
        stats["lines"].append(f"{head}  {status}  ply {job.ply or top(job.votes['ply'])[0] or '?'}  range {rng_txt}"
                              + (f"  -> bin {row['bin']}" if row else "") + f"  ({job.reads} OCR reads)")

    def timed_out(job):
        return args.read_timeout is not None and job.started is not None \
            and time.time() - job.started >= args.read_timeout

    def rng_ok(job):
        return confirmed(job, "range", args.votes, args.fast_votes) is not None

    def fields_to_read(job, extra):
        """Crop fields worth reading on this job, in order (under the lock). extra=False: what its bin still
        depends on; extra=True: the range, only for the printed report."""
        if job.done or (job.result and job.result["status"] == "unreadable") or rng_ok(job):
            return ("ply", "auto") if job.ply is None and job.result is None and not extra else ()
        if job.result is not None or (job.ply is not None and not job.needs_range):
            return ("range", "auto") if extra else ()
        if job.ply is None:  # while no ply crop is waiting, read the range instead of idling (v4)
            return () if extra else ("ply", "auto") + (() if job.closed or timed_out(job) else ("range",))
        return () if extra else ("range", "auto")  # case 3: the range decides the bin

    def next_item():
        """Read ahead: the first job in seq order with a crop its bin depends on, then report-only range crops."""
        for extra in (False, True):
            for job in jobs:
                fields = fields_to_read(job, extra)
                item = job.take(fields) if fields else None
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

            field, crop, rotation, crop_frame = item
            t0 = time.perf_counter()
            text, score, text2, score2, used = reader.read(crop, rotation)
            ocr_ms = (time.perf_counter() - t0) * 1000
            if rotation is None and used == 180:  # way up unknown: the higher-scoring way is used
                text, score, text2, score2 = text2, score2, text, score
            # every non-empty read is a vote (no score threshold); "auto" boxes are only logged
            vote = bool(text) and field != "auto" and not (field == "ply" and job.ply is not None)
            crop_name = ""
            if run_dir:
                with cond:
                    stats["crops"] += 1
                    crop_name = f"crops/s{job.seq:03d}_f{crop_frame:05d}_{field}_{stats['crops']:06d}.png"
                # saved the way it was read when the way up is known, as cut out when both ways were read
                cv2.imwrite(str(run_dir / crop_name), crop if rotation in (None, 0) else cv2.rotate(crop, cv2.ROTATE_180))
            with cond:
                job.reads += 1
                stats["ocr_ms"].append(ocr_ms)
                stats["votes"] += vote
                if ocr_log and not log_files[0].closed:
                    # text/score = the read used (used = 0 or 180 deg); *_other_way = the other way up, only when
                    # the way up was unknown
                    ocr_log.writerow([f"{time.time() - t_start:.2f}", job.seq, job.track_id, crop_frame, field,
                                      "layout" if rotation is not None else "unknown", text, f"{score:.3f}",
                                      "" if text2 is None else text2, "" if score2 is None else f"{score2:.3f}",
                                      used, int(vote), f"{ocr_ms:.1f}", worker, crop_name])
                    log_files[1].flush()
                if vote:
                    before = confirmed(job, field, args.votes, args.fast_votes)
                    job.votes[field][text] += 1
                    if score >= args.fast_score:
                        job.sure[field][text] += 1
                    if before is None and top(job.votes[field])[1] < args.votes \
                            and confirmed(job, field, args.votes, args.fast_votes) is not None:
                        stats["fast"] += 1
            with cond:
                job.busy -= 1
                ply = confirmed(job, "ply", args.votes, args.fast_votes)
                if job.ply is None and ply is not None:
                    job.needs_range = bool(ply_key(ply)) and picker.needs_range(ply_key(ply))
                    job.ply, job.ply_after = ply, time.time() - job.first_seen
                    if job.needs_range and job.started is not None:
                        job.started = time.time()  # case 3 at the head: the range wait gets its own --read-timeout
                cond.notify_all()

    def ready_to_decide(job):
        """(under the lock) None = keep waiting, else the decide() arguments."""
        if job.busy:
            return None
        if job.ply is None:  # on a timeout, the ply crops already waiting are still read first
            if not job.crops["ply"] and not job.crops["auto"] and (timed_out(job) or job.closed):
                return {"unreadable": True}
            return None
        if not job.needs_range:  # cases 1 and 2 (and not in master)
            return {}
        rng = confirmed(job, "range", args.votes, args.fast_votes)
        if rng:
            return {"rng": rng}
        if (timed_out(job) or job.closed) and not job.crops["range"] and not job.crops["auto"]:
            return {"rng": top(job.votes["range"])[0]}  # best reading so far
        return None

    def ready_to_report(job):
        """(under the lock) decided, and its range is confirmed or no range crop is left to read."""
        return job.result is not None and not job.busy and (
            job.result["status"] == "unreadable" or rng_ok(job)
            or not (job.crops["range"] or job.crops["auto"]))

    def committer():
        """Decides, lights and reports rolls strictly in seq order. A decided roll that is still reading its range
        for the report does not hold up the next roll's light."""
        while True:
            with cond:
                while True:
                    if jobs and ready_to_report(jobs[0]):
                        job = jobs.popleft()
                        job.done = True
                        action = None
                        break
                    job = next((j for j in jobs if j.result is None), None)
                    if job is not None:
                        if job.started is None:
                            job.started = time.time()  # it is the next roll to decide: --read-timeout starts
                        action = ready_to_decide(job)
                        if action is not None:
                            break
                    cond.wait(0.05)
            if action is None:
                report(job)
            else:
                decide(job, **action)

    for w in range(1, max(1, args.ocr_workers) + 1):
        threading.Thread(target=ocr_worker, args=(w,), daemon=True).start()
    threading.Thread(target=committer, daemon=True).start()

    tracks, writer, frame_no, t_fps, fps, seq = [], None, 0, time.time(), 0.0, 0
    for frame in frames_15fps(cap, is_file):
        frame_no += 1
        fh, fw = frame.shape[:2]
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
            for cls, _, rect in ds:
                field = {PLY: "ply", RANGE: "range", TEXT: "auto"}[cls]
                job = t.job
                wanted = job is None or not (job.done or (field == "ply" and job.ply is not None)
                                             or (job.result and job.result["status"] == "unreadable"))
                if wanted:  # every box on every frame
                    crop = crop_rect(frame, rect)
                    if crop.size:
                        with cond:
                            if t.job is None:  # first text on this roll: it joins the queue
                                seq += 1
                                t.job = RollJob(seq, t.id)
                                jobs.append(t.job)
                            if t.job.closed and t.job.result is None:  # seen again before it was decided
                                t.job.closed = False
                            t.job.add(field, crop, rotation, frame_no)
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
                    ply = job.ply or top(job.votes["ply"])[0]
                    rng = top(job.votes["range"])[0]
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
            cv2.imshow("RTracker OCR queue v6", frame)
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
