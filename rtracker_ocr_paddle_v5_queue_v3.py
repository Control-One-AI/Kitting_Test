"""Roll tracking + OCR (PP-OCRv5 mobile) with an ordered result queue that drives the ESP32 bin lights -- queue v3.
Same queue and bin selection as rtracker_ocr_paddle_v5_queue_v2.py; the only addition is --rec-model, which chooses the
PP-OCRv5 recognizer (stock models/v5/ppocr_rec.onnx by default, or a fine-tuned one such as
models/v5/ppocr_rec_fine_tuned.onnx) so both can be compared on the same video.
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

Pi 5 (USB camera):  python3 rtracker_ocr_paddle_v5_queue_v3.py --master data/master.csv --source /dev/video0 --model models/rolls.hef --esp32-port /dev/ttyUSB0
PC (video file):    python rtracker_ocr_paddle_v5_queue_v3.py --master data/master.csv --source videos/cam0_onsite.mp4 --model models/rolls_v2.pt --rec-model models/v5/ppocr_rec_fine_tuned.onnx
(omit --esp32-port for a dry run: results in the terminal only)
"""
import argparse
import csv
import os
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from bin_light_controller import BinLightController
from detector import load_detector, rect_bounds, rect_points, PLY, RANGE, ROLL, TEXT
from ocr_paddle_v5 import TextReader, PLY_RE, RANGE_RE, align_rect, crop_rect, text_rotation
from ply_bin_lights import norm_ply
from rtracker_ocr_paddle_v5 import FPS, MIN_SCORE, MAX_MISSED, iou, open_source, frames_15fps


class Track:
    next_id = 1

    def __init__(self, rect):
        self.id, Track.next_id = Track.next_id, Track.next_id + 1
        self.rect, self.missed, self.job = rect, 0, None  # rotated rect (cx, cy, w, h, theta)
        self.last_snap = {"ply": -1000, "range": -1000, "auto": -1000}


class RollJob:
    """One roll in the result queue. The main loop adds crops, the OCR worker takes them (both under the lock)."""

    def __init__(self, seq, track_id, cap):
        self.seq, self.track_id, self.cap = seq, track_id, cap
        self.first_seen = time.time()
        self.started = None     # when OCR started on this roll (it reached the head of the queue)
        self.crops = {"ply": [], "range": [], "auto": []}  # waiting crops: (sharpness, crop, rotation)
        self.votes = {"ply": Counter(), "range": Counter()}
        self.reads = 0          # OCR attempts on this roll
        self.closed = False     # roll left the frame, no more crops coming
        self.ply = None         # confirmed ply
        self.result = None      # BinPicker.pick() result, or {"status": "unreadable"}
        self.decided_after = 0.0
        self.done = False       # printed; no more crops wanted

    def add(self, field, crop, rotation):
        """Keep at most `cap` crops per field waiting; the blurriest is dropped first."""
        sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        waiting = self.crops[field]
        waiting.append((sharp, crop, rotation))
        if len(waiting) > self.cap:
            waiting.remove(min(waiting, key=lambda c: c[0]))

    def take(self, fields):
        """Sharpest waiting crop of the first field that has one -> (field, crop, rotation), or None."""
        for f in fields:
            waiting = self.crops[f]
            if waiting:
                _, crop, rotation = waiting.pop(max(range(len(waiting)), key=lambda i: waiting[i][0]))
                return f, crop, rotation
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
    ap.add_argument("--rec-model", default="models/v5/ppocr_rec.onnx",
                    help="PP-OCRv5 text recognizer, e.g. models/v5/ppocr_rec_fine_tuned.onnx")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--exposure", type=int, default=None, help="manual exposure (100 us units), omit for auto")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--roi", default=None, help="static ROI 'x,y,w,h' in source pixels; only this region is detected")
    ap.add_argument("--votes", type=int, default=3, help="matching OCR reads needed to confirm a value")
    ap.add_argument("--read-timeout", type=float, default=None,
                    help="seconds of OCR on a roll (from when it reaches the head of the queue) before it is "
                         "UNREADABLE; omit to wait until the roll has left the frame and all its crops are read")
    ap.add_argument("--range-max-diff", type=int, default=2,
                    help="case 3: most differing digits (start + end) for a range to match a sheet row")
    ap.add_argument("--range-tol", type=float, default=0.1,
                    help="sheet ranges of one ply within this on start and end count as the same range (case 2)")
    ap.add_argument("--snapshots", type=int, default=6, help="crops kept waiting per field of a roll")
    ap.add_argument("--snap-every", type=int, default=2, help="frames between snapshots per field of a roll")
    ap.add_argument("--key-col", default="ply_no")
    ap.add_argument("--bin-col", default="no_of_ply")
    ap.add_argument("--esp32-port", default=None, help="ESP32 serial port, e.g. /dev/ttyUSB0 (omit = dry run)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--blink-gap", type=float, default=0.3, help="seconds off before re-lighting the same bin")
    ap.add_argument("--out", default="output/annotated_queue_v3.mp4")
    ap.add_argument("--csv", default="output/readings_queue_v3.csv")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    lights = BinLightController(port=args.esp32_port, baud=args.baud)
    picker = BinPicker(args.master, args.key_col, args.bin_col, lights.bin_map, args.range_max_diff, args.range_tol)
    lights.open()
    detector = load_detector(args.model, args.conf)
    reader = TextReader(args.rec_model)
    print(f"recognizer: {args.rec_model}")
    cap, is_file = open_source(args.source, args.width, args.height, args.exposure)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source}")

    new_csv = not os.path.exists(args.csv)
    csv_file = open(args.csv, "a", newline="")
    log = csv.writer(csv_file)
    if new_csv:
        log.writerow(["timestamp", "seq", "track_id", "status", "case", "ply", "range_read", "sheet_range",
                      "digits_off", "bin", "relay", "sheet_row", "seconds_after_first_seen"])

    jobs = deque()  # RollJobs in arrival order; jobs[0] is the one being read
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

    def decide(job, rng):
        """Pick the bin (in seq order, since only the head job gets here) and light it."""
        job.result = picker.pick(job.ply, rng)
        job.decided_after = time.time() - job.first_seen
        if job.result["status"] == "picked" and job.result["row"]["relay"] is not None:
            light(job.result["row"]["relay"])

    def report(job):
        """One terminal line + CSV row per roll, in seq order."""
        res, head = job.result, f"[seq {job.seq}] roll #{job.track_id}"
        status, case, g = res["status"], res.get("case", ""), res.get("group")
        rng, n = top(job.votes["range"])
        rng_txt = "?" if rng is None else rng if n >= args.votes else f"{rng} (unconfirmed)"
        sheet = f"{g['start']}-{g['end']}" if g else ""
        diff = res.get("diff")
        off = "" if diff is None else f", {diff} digit{'s' if diff != 1 else ''} off"
        row = res.get("row")
        if status == "unreadable":
            guess, votes = top(job.votes["ply"])
            hint = f"best guess ply {guess} ({votes}/{args.votes} reads)" if guess else "no ply read"
            print(f"{head}  UNREADABLE  ({hint}, {job.reads} OCR reads)")
        elif status == "not_in_master":
            print(f"{head}  ply {job.ply} not in {picker.path.name}  |  range {rng_txt}")
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
        log.writerow([datetime.now().isoformat(timespec="seconds"), job.seq, job.track_id, status, case,
                      job.ply or "", rng or "", sheet, "" if diff is None else diff,
                      row["bin"] if row else "", row["relay"] if row and row["relay"] is not None else "",
                      row["line"] if row else "", round(job.decided_after or time.time() - job.first_seen, 1)])
        csv_file.flush()

    def ocr_worker():
        while True:
            ready = finish = False
            item = None
            with cond:
                while not jobs:
                    cond.wait()
                job = jobs[0]
                if job.started is None:
                    job.started = time.time()
                timed_out = args.read_timeout is not None and time.time() - job.started >= args.read_timeout
                rng_ok = top(job.votes["range"])[1] >= args.votes
                if job.ply is None:  # reading the ply
                    item = None if timed_out else job.take(("ply", "auto"))
                    if item is None and (timed_out or job.closed):
                        job.result = {"status": "unreadable"}
                        finish = True
                elif job.result is None:  # case 3: the bin depends on the range, keep reading it
                    item = None if (timed_out or rng_ok) else job.take(("range", "auto"))
                    ready = item is None and (timed_out or rng_ok or job.closed)
                else:  # decided: range read from the crops already there, for reference only
                    item = None if rng_ok else job.take(("range", "auto"))
                    finish = item is None
                if finish:
                    job.done = True
                    jobs.popleft()
                elif item is None and not ready:
                    cond.wait(0.05)
                    continue
            if ready:
                decide(job, top(job.votes["range"])[0])  # best reading so far
                continue
            if finish:
                report(job)
                continue

            field, crop, rotation = item
            # "auto" = unclassified text line (no-training mode): keep whichever format it matches
            for f in (["range", "ply"] if field == "auto" else [field]):
                if f == "ply" and job.ply is not None:
                    continue
                text, score = reader.read(crop, PLY_RE if f == "ply" else RANGE_RE, rotation)
                job.reads += 1
                if text and score >= MIN_SCORE:
                    with cond:
                        job.votes[f][text] += 1
                    break
            ply, n = top(job.votes["ply"])
            if job.ply is None and n >= args.votes:
                job.ply = ply
                if not picker.needs_range(ply):  # cases 1 and 2 (and not in master): decide now
                    decide(job, None)

    threading.Thread(target=ocr_worker, daemon=True).start()

    tracks, writer, frame_no, t_fps, fps, seq = [], None, 0, time.time(), 0.0, 0
    for frame in frames_15fps(cap, is_file):
        frame_no += 1
        fh, fw = frame.shape[:2]
        if roi:  # detect inside the ROI only (more pixels on the text), then shift back to frame coords
            rx, ry = max(0, roi[0]), max(0, roi[1])
            rw, rh = min(roi[2], fw - rx), min(roi[3], fh - ry)
            dets = [(c, s, (r[0] + rx, r[1] + ry, r[2], r[3], r[4]))
                    for c, s, r in detector.detect(frame[ry:ry + rh, rx:rx + rw])]
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 1)
        else:
            dets = detector.detect(frame)
        rolls = [d[2] for d in dets if d[0] == ROLL]
        texts = [d for d in dets if d[0] in (PLY, RANGE, TEXT)]

        # --- track rolls (greedy IoU matching on the rotated boxes' outer bounds) ---
        unmatched = list(range(len(rolls)))
        for t in tracks:
            best = max(unmatched, key=lambda j: iou(rect_bounds(t.rect), rect_bounds(rolls[j])), default=None)
            if best is not None and iou(rect_bounds(t.rect), rect_bounds(rolls[best])) > 0.3:
                t.rect, t.missed = rolls[best], 0
                unmatched.remove(best)
            else:
                t.missed += 1
        gone = [t for t in tracks if t.missed > MAX_MISSED and t.job]
        if gone:  # roll left the frame: its job gets no more crops
            with cond:
                for t in gone:
                    t.job.closed = True
                cond.notify_all()
        tracks = [t for t in tracks if t.missed <= MAX_MISSED] + [Track(rolls[j]) for j in unmatched]

        # --- assign text boxes to the roll whose rotated box contains their centre ---
        roll_texts = {}
        for d in texts:
            for t in tracks:
                if t.missed == 0 and cv2.pointPolygonTest(rect_points(t.rect), d[2][:2], False) >= 0:
                    roll_texts.setdefault(t, []).append(d)
                    break

        # --- text orientation from layout (ply above start-end), then snapshot straightened crops into the job ---
        for t, ds in roll_texts.items():
            rng = [d[2] for d in ds if d[0] == RANGE]
            if rng:  # point every text box of this roll the same way as the range box
                ds = [(c, s, align_rect(r, rng[0][4])) for c, s, r in ds]
            ply = [d[2] for d in ds if d[0] == PLY]
            rotation = text_rotation(ply[0], rng[0]) if ply and rng else None  # None: OCR tries both ways
            for cls, _, rect in ds:
                field = {PLY: "ply", RANGE: "range", TEXT: "auto"}[cls]
                job = t.job
                wanted = job is None or not (job.done or (field == "ply" and job.ply is not None))
                due = frame_no - t.last_snap[field] >= args.snap_every or t.last_snap[field] == frame_no
                if wanted and due:
                    crop = crop_rect(frame, rect)
                    if crop.size:
                        with cond:
                            if t.job is None:  # first text on this roll: it joins the queue
                                seq += 1
                                t.job = RollJob(seq, t.id, args.snapshots)
                                jobs.append(t.job)
                            t.job.add(field, crop, rotation)
                            cond.notify_all()
                        t.last_snap[field] = frame_no
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
            cv2.imshow("RTracker OCR queue v3", frame)
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

    cap.release()
    if writer:
        writer.release()
    csv_file.close()
    lights.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
