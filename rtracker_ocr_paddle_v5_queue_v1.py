"""Roll tracking + OCR (PP-OCRv5 mobile) with an ordered result queue that drives the ESP32 bin lights -- queue v1.
Same detection/tracking as rtracker_ocr_paddle_v5.py; what changes is how OCR results are turned into lights.

Each roll gets a job (seq = arrival order) holding snapshot crops of its ply and range boxes. One OCR worker reads
the jobs strictly in seq order, so a slow read delays the lights but never swaps them:
  ply confirmed (--votes matching reads) and in master_list.csv -> its bin lights at once, and stays lit until the
      next roll is decided; the same bin again is switched off and on (re-highlighted), not left on
  ply not in master_list.csv -> terminal only, lights unchanged
  no confirmed ply before the roll leaves the frame (or --read-timeout, if given) -> UNREADABLE, terminal only
The range is read and printed per roll (with the sheet's start-end for that ply) as a reference only.
master_list.csv is re-read whenever the file changes.

Pi 5 (USB camera):  python3 rtracker_ocr_paddle_v5_queue_v1.py --source /dev/video0 --model models/rolls.hef --esp32-port /dev/ttyUSB0
PC (video file):    python rtracker_ocr_paddle_v5_queue_v1.py --source videos/cam0_onsite.mp4 --model models/rolls_v2.pt
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
        self.done = False       # decided and printed; no more crops wanted

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


class MasterList:
    """ply -> (bin, start, end) from master_list.csv, re-read when the file changes (last good copy kept on error)."""

    def __init__(self, path, key_col, bin_col):
        self.path, self.key_col, self.bin_col = Path(path), key_col, bin_col
        self.mtime, self.rows = None, {}
        self.get()
        if not self.rows:
            sys.exit(f"no plys loaded from {self.path}")

    def get(self):
        try:
            mtime = self.path.stat().st_mtime
            if mtime == self.mtime:
                return self.rows
            with open(self.path, newline="", encoding="utf-8-sig", errors="replace") as f:
                rows = {}
                for r in csv.DictReader(f):
                    key, val = norm_ply(r[self.key_col] or ""), (r[self.bin_col] or "").strip()
                    if key and val:
                        try:
                            val = int(float(val))  # plain relay number
                        except ValueError:
                            pass                   # bin label such as 1-DW
                        rows[key] = (val, (r.get("start") or "").strip(), (r.get("end") or "").strip())
        except (OSError, KeyError, csv.Error) as e:
            print(f"{self.path}: cannot read ({e!r}) -- keeping the previous {len(self.rows)} plys")
            return self.rows
        self.rows, self.mtime = rows, mtime
        print(f"{self.path}: {len(rows)} plys loaded")
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/dev/video0", help="camera index, /dev/videoN or video file")
    ap.add_argument("--model", default="models/rolls.hef", help=".hef on the Pi, .pt on a PC")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--exposure", type=int, default=None, help="manual exposure (100 us units), omit for auto")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--roi", default=None, help="static ROI 'x,y,w,h' in source pixels; only this region is detected")
    ap.add_argument("--votes", type=int, default=3, help="matching OCR reads needed to confirm a value")
    ap.add_argument("--read-timeout", type=float, default=None,
                    help="seconds of OCR on a roll (from when it reaches the head of the queue) before it is "
                         "UNREADABLE; omit to wait until the roll has left the frame and all its crops are read")
    ap.add_argument("--snapshots", type=int, default=6, help="crops kept waiting per field of a roll")
    ap.add_argument("--snap-every", type=int, default=2, help="frames between snapshots per field of a roll")
    ap.add_argument("--master", default="data/master_list.csv", help="ply -> bin sheet")
    ap.add_argument("--key-col", default="ply_no")
    ap.add_argument("--bin-col", default="no_of_ply")
    ap.add_argument("--esp32-port", default=None, help="ESP32 serial port, e.g. /dev/ttyUSB0 (omit = dry run)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--blink-gap", type=float, default=0.3, help="seconds off before re-lighting the same bin")
    ap.add_argument("--out", default="output/annotated_queue_v1.mp4")
    ap.add_argument("--csv", default="output/readings_queue_v1.csv")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    master = MasterList(args.master, args.key_col, args.bin_col)
    lights = BinLightController(port=args.esp32_port, baud=args.baud)
    lights.open()
    detector = load_detector(args.model, args.conf)
    reader = TextReader()
    cap, is_file = open_source(args.source, args.width, args.height, args.exposure)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source}")

    new_csv = not os.path.exists(args.csv)
    csv_file = open(args.csv, "a", newline="")
    log = csv.writer(csv_file)
    if new_csv:
        log.writerow(["timestamp", "seq", "track_id", "status", "ply", "range", "sheet_range", "bin", "relay",
                      "seconds_after_first_seen"])

    jobs = deque()  # RollJobs in arrival order; jobs[0] is the one being read
    cond = threading.Condition()
    text_only = getattr(detector, "text_only", False)
    last_relay = None

    def light(relay):
        nonlocal last_relay
        if relay == last_relay:  # same bin as the previous roll: switch off first so it visibly re-lights
            lights.all_off()
            time.sleep(args.blink_gap)
        lights.set_relay(relay)
        last_relay = relay

    def decide_ply(job):
        """Ply confirmed: light its bin now (in seq order, since only the head job gets here)."""
        row = master.get().get(norm_ply(job.ply))
        job.sheet = row
        job.relay = None
        if row is not None:
            job.relay = row[0] if isinstance(row[0], int) else lights.bin_map.get(row[0])
            if job.relay is not None:
                light(job.relay)
        job.lit_after = time.time() - job.first_seen

    def report(job, status):
        """One terminal line + CSV row per roll, in seq order."""
        head = f"[seq {job.seq}] roll #{job.track_id}"
        if status == "unreadable":
            guess, n = top(job.votes["ply"])
            hint = f"best guess ply {guess} ({n}/{args.votes} reads)" if guess else "no ply read"
            print(f"{head}  UNREADABLE  ({hint}, {job.reads} OCR reads)")
            row = [job.seq, job.track_id, "unreadable", "", "", "", "", "", round(time.time() - job.first_seen, 1)]
        else:
            rng, n = top(job.votes["range"])
            rng_txt = "?" if rng is None else rng if n >= args.votes else f"{rng} (unconfirmed)"
            if job.sheet is None:
                print(f"{head}  ply {job.ply} not in {master.path.name}  |  range {rng_txt}")
                row = [job.seq, job.track_id, "not_in_master", job.ply, rng or "", "", "", ""]
            else:
                bin_, start, end = job.sheet
                sent = f"BIN {job.relay}" if job.relay is not None else f"no relay mapped for {bin_}"
                if job.relay is not None and not lights.enabled:
                    sent += " (dry run)"
                print(f"{head}  ply {job.ply} -> bin {bin_}  {sent}  |  range {rng_txt}  (sheet {start}-{end})"
                      f"  |  lit {job.lit_after:.1f} s after first seen")
                row = [job.seq, job.track_id, "lit", job.ply, rng or "", f"{start}-{end}", bin_, job.relay or ""]
            row.append(round(job.lit_after, 1))
        log.writerow([datetime.now().isoformat(timespec="seconds")] + row)
        csv_file.flush()

    def ocr_worker():
        while True:
            with cond:
                while not jobs:
                    cond.wait()
                job = jobs[0]
                if job.started is None:
                    job.started = time.time()
                status, item = None, None
                if job.ply is None:
                    timed_out = args.read_timeout is not None and time.time() - job.started >= args.read_timeout
                    item = None if timed_out else job.take(("ply", "auto"))
                    if item is None and (timed_out or job.closed):
                        status = "unreadable"
                elif top(job.votes["range"])[1] < args.votes:
                    item = job.take(("range", "auto"))  # range: only the crops already there, no waiting
                if item is None and status is None and job.ply is not None:
                    status = "read"
                if status:
                    job.done = True
                    jobs.popleft()
                elif item is None:
                    cond.wait(0.05)
                    continue
            if status:
                report(job, status)
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
                decide_ply(job)

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
                label = f"#{t.id} seq {job.seq} Ply {ply or '?'} | {rng or '?'}"
                color = (0, 200, 0) if job.ply else (0, 165, 255)
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
            cv2.imshow("RTracker OCR queue", frame)
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
