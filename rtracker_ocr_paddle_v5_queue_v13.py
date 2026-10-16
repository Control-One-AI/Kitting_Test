"""Roll tracking + OCR (PP-OCRv5 mobile) with an ordered result queue that drives the ESP32 bin lights -- queue v13.
New in v13: output folder and video only (reading, votes, queue and decisions are those of v12):
  one output folder  test_logs/latest_run/, emptied and rewritten at the start of every run (no per-run folders, no
                  shared output/readings_*.csv); everything a run writes is in it (list at the end of this text).
  snapshots/      every crop sent to the OCR, as read (right way up), with a band underneath showing what the OCR
                  read: text, score, field, vote or not, the other way up when both ways were read, frame, ms.
  OCR sidebar     a column on the right of the video (the video is widened, nothing is covered) showing the crops
                  in real time: READING NOW (the crop each OCR worker is reading, or its last crop + read when idle)
                  and NEXT IN QUEUE (the waiting crops in the order the workers will take them, '+N more').
  after the video when the source ends while crops are still being read, the last frame stays on screen (and in
                  annotated.mp4) with the live panel and sidebar and an 'VIDEO ENDED' banner until the queue is
                  done; q stops waiting.
  read after decision  every box is still cut out and read after its roll's bin is decided, to study the read
                  pattern. Undecided rolls are always served first, so decisions are not delayed; a decided roll's
                  crops are read when no undecided roll has any waiting. These reads never change a decision: they
                  are not votes, have after_decision = 1 in ocr_reads.csv, 'AFTER DECISION' on their snapshot, and
                  are totalled per roll at the end of summary.txt. The live read tags on the video keep updating.
                  --stop-after-decision: stop reading a roll once decided (v12).
Everything below is queue v12's description, unchanged:
New in v12: a redesigned annotated video only (reading, votes, queue, decisions and logs are those of v11):
  top panel       translucent rounded cards: title bar with fps badge; the last result as a large card with a
                  colour accent and a BIN / status badge; the queue as rounded pills with status dots; one row
                  per OCR worker with a status dot. Anti-aliased text and lines throughout.
  roll boxes      corner brackets with a light translucent fill and a rounded "#track  seq N" tag.
  text boxes      thin outlines with a small tag showing the latest raw read; the ROI is a dashed box.
Everything below is queue v11's description, unchanged:
queue v11:
New in v11: two changes to HOW the ply is chosen (reading, votes, queue, video and logs are those of v10):
  most read wins  at the end of a roll, the in-sheet ply that was read MOST often wins (reads of '85' and '85W'
                  count together; tie -> the most recent), not the most recent one. A ply read fewer than
                  --min-reads times (default 2) does not beat a confirmed value that is not in the sheet (that roll
                  is then 'not in master', as when nothing in the sheet was read). v10: the most recent in-sheet
                  read won, even if read once ('201' x1 beat '271' x25).
  range tie-break the recognizer's usual misreads turn one sheet ply into another ('124' is read as '121'). When
                  the chosen ply has such look-alikes in the sheet (MISREADS / DROPPED below: 4 read as 1, 7 as 1,
                  9 as 7, 6 as 2, ..., a dropped 4 / 1 / 5 / 8, a doubled digit read once, 4 read as 11), the
                  roll's range reads decide between them: for each candidate, the average number of digits by which
                  the range reads differ from its closest sheet range (each read counted --range-max-diff + 1 at
                  most). A look-alike replaces the chosen ply only if it is at least --tie-margin digits closer
                  and the roll has --tie-min-reads range reads or more; otherwise the chosen ply stays, as in v10.
                  An early (green) decision on a ply with look-alikes waits up to --tie-wait seconds after the ply
                  was confirmed for those range reads (0 = decide at once, v10 timing); a roll that leaves the frame
                  first is decided at the end as usual.
                  The ply rule column / terminal line shows a switch as e.g. 'most_read_in_sheet+range_tie(121->124)'.
  Not changed: a ply read that is not in the sheet is never turned into a look-alike that is.
Everything below is queue v10's description, unchanged:
New in v10: the annotated video separates live reading from final results (reading, votes and decisions unchanged):
  text boxes      each ply / range / text box shows the LATEST raw OCR read of that box ('ply: 7841-DW 0.93'), which
                  changes with every read (no voting). Reads lag the video a little: the OCR workers read queued crops
                  of earlier frames. A box is matched to its reads by its position relative to the roll's centre.
  roll boxes      only '#track seq N' and the state colour (orange reading, green picked, red no bin)
  top panel       fixed height, shows the queue at work:
                    line 1   fps + the LAST final result: seq, roll, ply | range -> bin / status
                    line 2   the queue, one chip per roll in seq order: 'next' marks the roll being decided,
                             P/R/T = its crops still waiting (ply / range / text), 'left' = roll left the frame;
                             orange = reading, green / red = decided, waiting to be reported in seq order
                    1 line per OCR worker (--ocr-workers): the crop it is reading now (roll, field, frame, ms so
                    far) or idle, and its last read (text, score, ms)
  results_table.csv  in the test log folder: one row per decided roll, written when it is decided, with the video
                  frame / time it appeared on the panel and the last raw reads at that moment
Everything below is queue v9's description, unchanged:
New in v9: the text recognizer can run on the Hailo-8L too -- two .hef files on one chip:
  --rec-model x.hef  recognizer on the Hailo (ocr_hailo_v1.HailoReader, characters from x_chars.txt or --rec-chars)
  --rec-model x.onnx recognizer on the CPU (ONNX Runtime), exactly as v8
  one Hailo device   when the detector and/or the recognizer is a .hef, one VDevice with the HailoRT scheduler
                     (detector_v3.open_hailo_device) is shared; the scheduler switches the chip between the models
Reading, votes and decisions are unchanged from v8 (single text class support included). Uses detector_v3.py.
Everything below is queue v8's description, unchanged:
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

Pi 5 (USB camera):  python3 rtracker_ocr_paddle_v5_queue_v13.py --master data/master.csv --source /dev/video0 --model models/rolls.hef --rec-model models/v5/ppocr_rec_fine_tuned.hef --esp32-port /dev/ttyUSB0
PC (video file):    python rtracker_ocr_paddle_v5_queue_v13.py --master data/master.csv --source videos/cam0_onsite.mp4 --model models/rolls_v2.pt --rec-model models/v5/ppocr_rec_fine_tuned.onnx --roi 240,150,840,220
(omit --esp32-port for a dry run: results in the terminal only)

Output folder (v13): ONE folder, test_logs/latest_run/, emptied at the start of every run (copy it elsewhere to keep
a run):
  run_info.json   source, date, script, all arguments, machine
  detections.csv  one row per detected box per frame (class, confidence, box, roll track id, detect ms)
  ocr_reads.csv   one row per OCR read (roll, field, orientation, read used + score, the other way up + score,
                  which way was used (0/180), counted as a vote, OCR ms, worker, crop image, snapshot image)
  crops/          every crop that was read, as PNG, plain (for labelling / training sets)
  snapshots/      the same crops with what the OCR read written underneath
  results.csv     one row per roll (what v12 also appended to output/readings_queue_v12.csv)
  results_table.csv  one row per decided roll as shown on the video's top panel (+ decision frame / video time)
  annotated.mp4   the annotated video with the OCR sidebar (unless --out is given)
  summary.txt     totals and average speeds, written at the end (also on q / Ctrl-C)
--no-log turns it off (no folder; annotated video then goes to output/annotated_queue_v13.mp4). --csv adds a CSV of
the results outside the folder (appended, as in v12) if wanted.
"""
import argparse
import atexit
import csv
import json
import os
import platform
import re
import shutil
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
from detector_v3 import load_detector, open_hailo_device, rect_bounds, rect_points, PLY, RANGE, ROLL, TEXT
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


# The recognizer's usual misreads (rec v5 on the labelled held-out crops and on det_v5 boxes of videos with a known
# value): written digit -> digits it is read as.
MISREADS = {"4": "1653", "7": "1", "9": "7", "6": "23", "5": "3", "8": "0", "2": "467"}
DROPPED = "4158"  # written digits that are often missing from the read


def written_candidates(key):
    """Ply numbers that are read as `key` after ONE usual misread: a swapped digit (MISREADS), a dropped digit
    (DROPPED), a doubled digit read once ('7844' -> '784') or a 4 read as 11 ('124' -> '1211'). The caller keeps
    those that are in the sheet."""
    out = set()
    for i, r in enumerate(key):
        out.update(key[:i] + w + key[i + 1:] for w, reads in MISREADS.items() if r in reads)
        out.add(key[:i] + r + key[i:])
        if key[i:i + 2] == "11":
            out.add(key[:i] + "4" + key[i + 2:])
    for i in range(len(key) + 1):
        out.update(key[:i] + w + key[i:] for w in DROPPED)
    out.discard(key)
    return {c for c in out if not c.startswith("0")}


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
        self.live = []          # latest raw read per text box: {"rel", "size", "field", "text", "score", "frame"}
        self.reads = 0          # OCR attempts on this roll
        self.busy = 0           # OCR reads in progress on this roll
        self.closed = False     # roll left the frame, no more crops coming
        self.ply = None         # chosen ply, the whole read (e.g. '72'), set when decided
        self.rule = ""          # how the ply was chosen
        self.rng = None         # chosen range read (case 3), set when decided
        self.result = None      # BinPicker.pick() result, or {"status": "unreadable"}
        self.decided_after = 0.0
        self.done = False       # printed (reported in seq order)
        self.after = {"ply": Counter(), "range": Counter()}  # reads AFTER the decision (logged only, no votes)
        self.after_reads = 0    # OCR reads after the decision, any text

    def add(self, field, crop, rotation, frame=None, timing=None):
        """Every crop is kept (no cap, nothing dropped). timing: detection conf / times for the test log."""
        self.crops[field].append((crop, rotation, frame, timing or {}))

    def take(self, fields):
        """Oldest waiting crop of the first field that has one -> (field, crop, rotation, frame, timing), or None."""
        for f in fields:
            if self.crops[f]:
                return (f,) + self.crops[f].popleft()
        return None

    def live_slot(self, rel, size):
        """(under the lock) The live entry of the text box at roll-relative centre rel (nearest within the box's
        short side), or None."""
        near = [(np.hypot(e["rel"][0] - rel[0], e["rel"][1] - rel[1]), i) for i, e in enumerate(self.live)]
        near = [(d, i) for d, i in near if d <= max(size, self.live[i]["size"])]
        return self.live[min(near)[1]] if near else None

    def set_live(self, rel, size, field, text, score, frame):
        """(under the lock) Latest raw read of the text box at rel; an older crop never overwrites a newer read."""
        e = self.live_slot(rel, size)
        if e is None:
            self.live.append({"rel": rel, "size": size, "field": field, "text": text, "score": score, "frame": frame})
        elif frame >= e["frame"]:
            e.update(rel=rel, size=size, field=field, text=text, score=score, frame=frame)


# --- annotated video look (v12): one palette (BGR), anti-aliased text and shapes, sizes scaled to the frame ---
FONT, FONT_B, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.LINE_AA
C_PENDING = (40, 170, 255)  # amber: roll still being read
C_PICKED = (95, 205, 75)  # green: bin picked
C_ERROR = (80, 80, 235)  # red: no bin
C_TEXTBOX = (235, 200, 60)  # cyan: ply / range / text boxes and their live reads
C_ACTIVE = (60, 220, 255)  # yellow: OCR worker reading
C_ROI = (210, 130, 230)  # soft magenta
C_PANEL, C_CARD, C_LINE = (24, 20, 17), (48, 42, 36), (85, 78, 70)
C_FG, C_MUTED, C_DARK = (245, 245, 245), (160, 160, 160), (25, 25, 25)


def ui_scale(width):
    """Size factor for the overlay: 1 at 1280 px wide."""
    return max(0.8, min(1.6, width / 1280))


def state_color(res):
    return C_PENDING if not res else C_PICKED if res["status"] == "picked" else C_ERROR


def put(img, text, org, scale, color, thick=1, font=FONT):
    cv2.putText(img, text, (int(org[0]), int(org[1])), font, scale, color, thick, AA)


def rounded_rect(img, p1, p2, color, r, thick=-1):
    """Filled (thick=-1) or outlined rectangle with corner radius r."""
    (x1, y1), (x2, y2) = map(int, p1), map(int, p2)
    r = int(max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    if thick < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for c in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, c, r, color, -1, AA)
        return
    for a, b in (((x1 + r, y1), (x2 - r, y1)), ((x1 + r, y2), (x2 - r, y2)),
                 ((x1, y1 + r), (x1, y2 - r)), ((x2, y1 + r), (x2, y2 - r))):
        cv2.line(img, a, b, color, thick, AA)
    for c, angle in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270), ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
        cv2.ellipse(img, c, (r, r), angle, 0, 90, color, thick, AA)


def pill_width(text, scale, k, thick=1, dot=False):
    return cv2.getTextSize(text, FONT, scale, thick)[0][0] + int(16 * k) + (int(12 * k) if dot else 0)


def pill(img, x, cy, text, scale, k, fg, bg, outline=None, dot=None, thick=1, outline_thick=1):
    """Rounded pill, left edge x, vertical centre cy, optional status dot; returns its right edge."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    pad, hh = int(8 * k), int(th / 2 + 6 * k)
    x2 = x + pill_width(text, scale, k, thick, dot is not None)
    rounded_rect(img, (x, cy - hh), (x2, cy + hh), bg, hh)
    if outline:
        rounded_rect(img, (x, cy - hh), (x2, cy + hh), outline, hh, outline_thick)
    tx = x + pad
    if dot:
        cv2.circle(img, (tx + int(3 * k), cy), max(2, int(3.5 * k)), dot, -1, AA)
        tx += int(12 * k)
    put(img, text, (tx, cy + th // 2), scale, fg, thick)
    return x2


def tag(frame, text, xy, k, fg, bg, outline=None, scale=0.5):
    """Small rounded label whose bottom-left corner is at xy, kept inside the frame."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale * k, 1)
    hh = int(th / 2 + 6 * k)
    x = int(min(max(0, xy[0]), frame.shape[1] - pill_width(text, scale * k, k) - 1))
    cy = int(max(2 * hh + 2, xy[1] - 3 * k) - hh)
    pill(frame, x, cy, text, scale * k, k, fg, bg, outline)


def draw_roll(frame, rect, color, label, k):
    """Roll box: light translucent fill, thin outline, bold corner brackets and a rounded id tag."""
    pts = np.int32(rect_points(rect))
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        roi = frame[y0:y1, x0:x1]
        layer = roi.copy()
        cv2.fillPoly(layer, [pts - (x0, y0)], color, AA)
        roi[:] = cv2.addWeighted(layer, 0.14, roi, 0.86, 0)
    cv2.polylines(frame, [pts], True, color, 1, AA)
    for i in range(4):  # corner brackets, 22% of each side
        p, a, b = pts[i], pts[i - 1], pts[(i + 1) % 4]
        for q in (a, b):
            end = p + (q - p) * 0.22
            cv2.line(frame, tuple(map(int, p)), tuple(map(int, end)), color, max(2, int(3 * k)), AA)
    tag(frame, label, (x, y - 4 * k), k, C_DARK, color, scale=0.6)


def dashed_rect(frame, p1, p2, color, k, dash=10):
    (x1, y1), (x2, y2) = p1, p2
    step = max(4, int(dash * k))
    for a, b in (((x1, y1), (x2, y1)), ((x1, y2), (x2, y2)), ((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // step))
        for i in range(0, n, 2):
            s, e = i / n, min(1.0, (i + 1) / n)
            cv2.line(frame, (int(a[0] + (b[0] - a[0]) * s), int(a[1] + (b[1] - a[1]) * s)),
                     (int(a[0] + (b[0] - a[0]) * e), int(a[1] + (b[1] - a[1]) * e)), color, 1, AA)
    put(frame, "ROI", (x1 + 4 * k, y1 + 14 * k), 0.4 * k, color)


def draw_panel(frame, fps, frame_no, last, chips, lanes):
    """Top panel, fixed height, on a dark translucent band: title bar with fps badge; the last final result as a card
    (main text + BIN / status badge); the queue as pills in seq order ('+N' when they don't fit); one row per OCR
    worker. last: (main, badge, color) or None; chips: [(text, color, is_head)]; lanes: [(busy, text, last_text)]."""
    fw = frame.shape[1]
    k = ui_scale(fw)
    m, gap = int(12 * k), int(7 * k)
    title_h, card_h, queue_h, lane_h = int(26 * k), int(50 * k), int(30 * k), int(22 * k)
    h = min(frame.shape[0], m + title_h + gap + card_h + gap + queue_h + gap + lane_h * len(lanes) + m // 2)
    band = frame[:h]
    band[:] = cv2.addWeighted(band, 0.25, np.full_like(band, C_PANEL), 0.75, 0)
    cv2.line(frame, (0, h - 1), (fw, h - 1), C_LINE, 1)

    # title bar: name on the left, frame number + fps badge on the right
    cy = m + title_h // 2
    put(frame, "RTracker OCR", (m, cy + 7 * k), 0.62 * k, C_FG, 1, FONT_B)
    tw = cv2.getTextSize("RTracker OCR", FONT_B, 0.62 * k, 1)[0][0]
    put(frame, "roll tracking  /  bin lights", (m + tw + 14 * k, cy + 5 * k), 0.42 * k, C_MUTED)
    fps_txt = f"{fps:4.1f} FPS"
    fps_col = C_PICKED if fps >= 12 else C_PENDING if fps >= 8 else C_ERROR
    x = fw - m - pill_width(fps_txt, 0.45 * k, k, dot=True)
    pill(frame, x, cy, fps_txt, 0.45 * k, k, C_FG, C_CARD, fps_col, dot=fps_col)
    f_txt = f"frame {frame_no}"
    put(frame, f_txt, (x - cv2.getTextSize(f_txt, FONT, 0.42 * k, 1)[0][0] - 12 * k, cy + 5 * k), 0.42 * k, C_MUTED)

    # last result card: accent strip in the result colour, BIN / status badge on the right
    y0 = m + title_h + gap
    color = last[2] if last else C_LINE
    rounded_rect(frame, (m, y0), (fw - m, y0 + card_h), C_CARD, 8 * k)
    rounded_rect(frame, (m, y0), (m + 6 * k, y0 + card_h), color, 3 * k)
    put(frame, "LAST RESULT", (m + 18 * k, y0 + 17 * k), 0.38 * k, C_MUTED)
    if last:
        put(frame, last[0], (m + 18 * k, y0 + card_h - 11 * k), 0.68 * k, C_FG, 1, FONT_B)
        bx = fw - m - 12 * k - pill_width(last[1], 0.62 * k, k, 2)
        pill(frame, int(bx), y0 + card_h // 2, last[1], 0.62 * k, k, C_DARK, color, thick=2)
    else:
        put(frame, "waiting for the first roll ...", (m + 18 * k, y0 + card_h - 13 * k), 0.55 * k, C_MUTED)

    # queue: one pill per roll in seq order, the head (being decided) outlined thicker and marked NEXT
    y0 += card_h + gap
    cy = y0 + queue_h // 2
    put(frame, "QUEUE", (m, cy + 5 * k), 0.42 * k, C_MUTED)
    x = m + int(58 * k)
    x = pill(frame, x, cy, str(len(chips)), 0.42 * k, k, C_FG, C_LINE) + int(10 * k)
    more_w = pill_width("+99", 0.45 * k, k) + int(8 * k)
    for i, (text, color, is_head) in enumerate(chips):
        text = f"NEXT  {text}" if is_head else text
        w = pill_width(text, 0.45 * k, k, dot=True)
        if x + w > fw - m - (more_w if i < len(chips) - 1 else 0):
            pill(frame, x, cy, f"+{len(chips) - i}", 0.45 * k, k, C_FG, C_LINE)
            break
        x = pill(frame, x, cy, text, 0.45 * k, k, C_FG, C_CARD, color, dot=color,
                 outline_thick=2 if is_head else 1) + int(6 * k)
    if not chips:
        put(frame, "empty", (x, cy + 5 * k), 0.45 * k, C_MUTED)

    # OCR workers: status dot, what it reads now, its last read in a second column
    y0 += queue_h + gap
    col = max(int(fw * 0.42), m + int(420 * k))
    for i, (busy, text, last_text) in enumerate(lanes):
        cy = y0 + i * lane_h + lane_h // 2
        dot = (m + int(5 * k), cy)
        if busy:
            cv2.circle(frame, dot, max(3, int(4 * k)), C_ACTIVE, -1, AA)
        else:
            cv2.circle(frame, dot, max(3, int(4 * k)), C_MUTED, 1, AA)
        put(frame, text, (m + 18 * k, cy + 5 * k), 0.45 * k, C_FG if busy else C_MUTED)
        if last_text:
            put(frame, last_text, (col, cy + 5 * k), 0.42 * k, C_MUTED)


def ascii_text(text):
    """cv2.putText draws ASCII only: anything else becomes '?'."""
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(text))


def as_read(crop, rotation):
    """The crop the right way up for the OCR (rotation 180: turned)."""
    return cv2.rotate(crop, cv2.ROTATE_180) if rotation == 180 else crop


def snapshot(crop, lines, color, height=64, min_w=380):
    """Snapshot for the output folder: the crop as read, scaled to `height`, on top of a dark band with `lines`
    (the first one large, in `color`; the others small and muted)."""
    h, w = crop.shape[:2]
    cw = max(1, int(w * height / max(h, 1)))
    img = cv2.resize(crop, (cw, height), interpolation=cv2.INTER_CUBIC) if crop.size else np.zeros((height, 1, 3), np.uint8)
    W, pad, line_h = max(cw, min_w) + 16, 8, 20
    out = np.full((height + 2 * pad + 30 + line_h * (len(lines) - 1) + pad, W, 3), C_PANEL, np.uint8)
    out[pad:pad + height, (W - cw) // 2:(W - cw) // 2 + cw] = img
    y = pad + height + 30
    put(out, ascii_text(lines[0]), (pad, y - 6), 0.7, color, 2, FONT_B)
    for line in lines[1:]:
        y += line_h
        put(out, ascii_text(line), (pad, y - 4), 0.45, C_MUTED)
    return out


def thumb(canvas, crop, x, y, max_w, max_h, outline=None, dim=False):
    """Crop scaled to fit max_w x max_h, pasted at (x, y); returns its (w, h)."""
    h, w = crop.shape[:2]
    if not crop.size or not h or not w:
        return 0, 0
    s = min(max_w / w, max_h / h)
    tw, th = max(1, int(w * s)), max(1, int(h * s))
    img = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if dim:
        img = cv2.addWeighted(img, 0.45, np.full_like(img, C_PANEL), 0.55, 0)
    canvas[y:y + th, x:x + tw] = img
    if outline:
        cv2.rectangle(canvas, (x - 1, y - 1), (x + tw, y + th), outline, 1, AA)
    return tw, th


def draw_sidebar(canvas, x0, k, reading, waiting, n_waiting):
    """OCR sidebar, from x0 to the right edge of canvas: READING NOW (one entry per worker: (busy, label, crop))
    and NEXT IN QUEUE (waiting: [(label, crop)] in the order the workers take them; n_waiting = all of them)."""
    fh, W = canvas.shape[:2]
    m, sw = int(10 * k), W - x0
    canvas[:, x0:] = C_PANEL
    cv2.line(canvas, (x0, 0), (x0, fh), C_LINE, 1)
    x, iw = x0 + m, sw - 2 * m
    y = m + int(18 * k)
    put(canvas, "OCR INPUT", (x, y), 0.6 * k, C_FG, 1, FONT_B)
    n_txt = f"{n_waiting} waiting"
    pill(canvas, W - m - pill_width(n_txt, 0.42 * k, k), int(y - 6 * k), n_txt, 0.42 * k, k, C_FG, C_LINE)
    y += int(14 * k)

    th, lab = int(40 * k), int(16 * k)  # thumbnail height, label line
    y += lab
    put(canvas, "READING NOW", (x, y), 0.4 * k, C_MUTED)
    y += int(6 * k)
    for busy, label, crop in reading:
        rounded_rect(canvas, (x0 + m // 2, y), (W - m // 2, y + lab + th + int(10 * k)), C_CARD, 6 * k)
        cv2.circle(canvas, (x + int(4 * k), y + lab // 2 + int(3 * k)), max(3, int(4 * k)), C_ACTIVE if busy else C_MUTED,
                   -1 if busy else 1, AA)
        put(canvas, label, (x + int(14 * k), y + lab), 0.42 * k, C_FG if busy else C_MUTED)
        if crop is not None:
            thumb(canvas, crop, x, y + lab + int(4 * k), iw, th, C_ACTIVE if busy else C_LINE, dim=not busy)
        y += lab + th + int(16 * k)

    y += lab
    put(canvas, "NEXT IN QUEUE", (x, y), 0.4 * k, C_MUTED)
    y += int(6 * k)
    qh = int(30 * k)  # smaller thumbnails for the queue
    if not waiting:
        put(canvas, "empty", (x, y + lab), 0.42 * k, C_MUTED)
    for i, (label, crop) in enumerate(waiting):
        if y + lab + qh + int(8 * k) + lab > fh - m:  # keep a line for '+N more'
            put(canvas, f"+{n_waiting - i} more", (x, y + lab), 0.45 * k, C_FG)
            break
        put(canvas, label, (x, y + lab - int(2 * k)), 0.38 * k, C_MUTED)
        thumb(canvas, crop, x, y + lab + int(2 * k), iw, qh, C_TEXTBOX)
        y += lab + qh + int(8 * k)
    else:
        if n_waiting > len(waiting):
            put(canvas, f"+{n_waiting - len(waiting)} more", (x, y + lab), 0.45 * k, C_FG)


def banner(frame, text, k):
    """Large centred pill near the bottom of the video."""
    s = 0.7 * k
    w = pill_width(text, s, k, 2, dot=True)
    pill(frame, (frame.shape[1] - w) // 2, frame.shape[0] - int(40 * k), text, s, k, C_DARK, C_PENDING,
         dot=C_DARK, thick=2)


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
    ap.add_argument("--rec-model", default="models/v5/ppocr_rec_fine_tuned.hef",
                    help="PP-OCRv5 text recognizer: .hef on the Hailo-8L (compile_hailo_rec_v1.py), "
                         ".onnx on the CPU (e.g. models/v5/ppocr_rec_fine_tuned.onnx on a PC)")
    ap.add_argument("--rec-chars", default=None,
                    help="character list of a .hef recognizer (default: <rec-model>_chars.txt next to it)")
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
    ap.add_argument("--min-reads", type=int, default=2,
                    help="end of roll: an in-sheet ply read fewer times than this does not beat a confirmed value "
                         "that is not in the sheet")
    ap.add_argument("--tie-margin", type=float, default=1.0,
                    help="range tie-break: a look-alike sheet ply replaces the chosen one only if its sheet range is "
                         "at least this many digits closer to the range reads (a very large value turns it off)")
    ap.add_argument("--tie-min-reads", type=int, default=2,
                    help="range tie-break: range reads a roll needs before the range may change its ply")
    ap.add_argument("--tie-wait", type=float, default=2.0,
                    help="early decision on a ply with look-alikes in the sheet: seconds to wait after the ply was "
                         "confirmed for --tie-min-reads range reads (0 = do not wait)")
    ap.add_argument("--range-tol", type=float, default=0.1,
                    help="sheet ranges of one ply within this on start and end count as the same range (case 2)")
    ap.add_argument("--key-col", default="ply_no")
    ap.add_argument("--bin-col", default="no_of_ply")
    ap.add_argument("--esp32-port", default=None, help="ESP32 serial port, e.g. /dev/ttyUSB0 (omit = dry run)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--blink-gap", type=float, default=0.3, help="seconds off before re-lighting the same bin")
    ap.add_argument("--out", default=None, help="annotated video (default: annotated.mp4 in the output folder, "
                                                 "or output/annotated_queue_v13.mp4 with --no-log)")
    ap.add_argument("--csv", default=None, help="also append the results to this CSV outside the output folder "
                                                 "(default: none; results.csv in the output folder has them)")
    ap.add_argument("--log-dir", default="test_logs",
                    help="the output folder is <log-dir>/latest_run, emptied at the start of every run")
    ap.add_argument("--sidebar", type=int, default=300, help="width of the OCR sidebar at 1280 px (0 = none)")
    ap.add_argument("--no-log", action="store_true", help="no output folder for this run")
    ap.add_argument("--stop-after-decision", action="store_true",
                    help="stop reading a roll once its bin is decided (v12 behaviour); default: keep reading every "
                         "box after the decision, logged only (after_decision column), never changing the result")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    for stream in (sys.stdout, sys.stderr):  # free reads may hold characters the console cannot show
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    run_dir = None
    if not args.no_log:  # <log-dir>/latest_run/: one folder, emptied and rewritten by every run
        run_dir = Path(args.log_dir) / "latest_run"
        if run_dir.exists():
            try:
                shutil.rmtree(run_dir)
            except OSError as e:
                sys.exit(f"cannot empty {run_dir} ({e}) -- is a file in it open (e.g. annotated.mp4 in a player)?")
        run_dir.mkdir(parents=True)
        (run_dir / "crops").mkdir()
        (run_dir / "snapshots").mkdir()
    if args.out is None:
        args.out = str(run_dir / "annotated.mp4") if run_dir else "output/annotated_queue_v13.mp4"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    lights = BinLightController(port=args.esp32_port, baud=args.baud)
    picker = BinPicker(args.master, args.key_col, args.bin_col, lights.bin_map, args.range_max_diff, args.range_tol)
    lights.open()
    uses_hailo = args.model.endswith(".hef") or args.rec_model.endswith(".hef")
    device = open_hailo_device() if uses_hailo else None  # one Hailo-8L for both .hef models (HailoRT scheduler)
    detector = load_detector(args.model, args.conf, args.classes.split(",") if args.classes else None,
                             args.merge_text, device)
    single_text = TEXT in getattr(detector, "ids", [])
    print(f"detector: {args.model}  classes {getattr(detector, 'names', '?')}"
          + ("  -> single text class: the reading decides ply / range" if single_text else ""))
    if args.rec_model.endswith(".hef"):  # on the shared Hailo-8L; the OCR workers take turns on the chip
        from ocr_hailo_v1 import HailoReader
        reader = HailoReader(args.rec_model, device, args.rec_chars)
        where = "Hailo-8L"
    else:
        reader = FreeReader(args.rec_model, threads=args.ocr_threads)  # one session, shared by the OCR workers
        where = f"CPU, {args.ocr_threads} threads each"
    print(f"recognizer: {args.rec_model}  ({args.ocr_workers} OCR workers, {where})"
          + ("   Hailo: one shared device, HailoRT scheduler" if device else ""))
    cap, is_file = open_source(args.source, args.width, args.height, args.exposure)
    if not cap.isOpened():
        sys.exit(f"Cannot open source {args.source}")

    result_cols = ["timestamp", "seq", "track_id", "status", "case", "ply", "ply_lookup", "ply_rule", "ply_reads",
                   "range_read", "range_reads", "sheet_range", "digits_off", "bin", "relay", "sheet_row",
                   "seconds_after_first_seen", "queue_wait_s", "ply_first_read_s"]
    csv_file = log = None
    if args.csv:  # optional extra copy of the results outside the output folder, appended (as v12)
        new_csv = not os.path.exists(args.csv)
        csv_file = open(args.csv, "a", newline="", encoding="utf-8")  # free reads may hold any character
        log = csv.writer(csv_file)
        if new_csv:
            log.writerow(result_cols)

    # --- test log of this run ---
    t_start = time.time()
    stats = {"frames": 0, "det_ms": [], "classes": Counter(), "ocr_ms": [], "votes": 0, "results": Counter(),
             "lines": [], "fast": 0, "queue_wait": [], "lit": [], "crops": 0}
    det_log = ocr_log = res_log = table_log = None
    log_files = []
    video = {"frame": 0}  # frame the main loop is on, for the results table
    shown = deque(maxlen=1)  # last final result on the top panel: (main text, badge, color)
    workers = {w: {"job": None, "last": None, "crop": None, "last_crop": None}  # for the panel / sidebar
               for w in range(1, max(1, args.ocr_workers) + 1)}
    if run_dir:
        info = {"source": args.source, "is_file": is_file, "started": datetime.now().isoformat(timespec="seconds"),
                "script": Path(__file__).name, "args": vars(args), "machine": platform.node(),
                "platform": platform.platform(), "processor": platform.processor(),
                "source_fps": cap.get(cv2.CAP_PROP_FPS), "source_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "processing_fps": FPS}
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        files = {name: open(run_dir / name, "w", newline="", encoding="utf-8-sig")  # -sig: Excel shows any symbol
                 for name in ("detections.csv", "ocr_reads.csv", "results.csv", "results_table.csv")}
        log_files = list(files.values())
        det_log, ocr_log, res_log, table_log = (csv.writer(files[n]) for n in ("detections.csv", "ocr_reads.csv",
                                                                               "results.csv", "results_table.csv"))
        # the top panel of the video, one row per decided roll in seq order
        table_log.writerow(["seq", "track_id", "status", "ply", "range", "bin", "relay", "panel_text",
                            "decided_frame", "decided_video_time_s", "elapsed_s", "decided_after_first_seen_s",
                            "ply_rule", "ocr_reads", "ply_reads", "range_reads", "last_raw_reads"])
        det_log.writerow(["frame", "video_time_s", "elapsed_s", "class", "conf", "cx", "cy", "w", "h", "theta",
                          "roll_track_id", "detect_ms"])
        ocr_log.writerow(["elapsed_s", "seq", "roll_track_id", "crop_frame", "field", "orientation", "text", "score",
                          "text_other_way", "score_other_way", "used", "counted_as_vote", "ocr_ms", "worker", "crop",
                          # timing of this box: detection confidence, detection ms of its frame, crops waiting in
                          # the whole queue when it joined (itself included), ms it waited before an OCR worker
                          # took it, and total ms from the start of its frame's detection to the end of its read
                          "det_conf", "detect_ms", "queue_len_at_join", "queue_ms", "total_ms", "snapshot",
                          "after_decision"])  # 1 = read after the roll's bin was decided: logged only, no vote
        res_log.writerow(result_cols)
        print(f"output folder (emptied for this run): {run_dir}")

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
                 "ply = most read in the sheet (fewer than --min-reads reads: a confirmed value goes first), else "
                 "latest confirmed; case 3 range = read closest to a sheet range",
                 f"v11         min reads {args.min_reads}   range tie-break between look-alike sheet plys: margin "
                 f"{args.tie_margin} digits, from {args.tie_min_reads} range reads, early decision waits up to "
                 f"{args.tie_wait} s for them",
                 f"frames      {stats['frames']} at {FPS} fps in {elapsed:.1f} s "
                 f"({stats['frames'] / max(elapsed, 1e-6):.1f} fps processed)",
                 f"detection   {ms(det)}" + (f"  (~{1000 / statistics.mean(det):.0f} fps)" if det else "") + warm,
                 "detections  " + (", ".join(f"{k} {v}" for k, v in stats["classes"].most_common()) or "none"),
                 f"ocr         {len(stats['ocr_ms'])} reads, {stats['votes']} counted as votes, {ms(stats['ocr_ms'])}"
                 f", {stats['crops']} crops + snapshots saved",
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
        if not args.stop_after_decision:  # reads after each decision: logged only, they never changed a result
            lines += ["", "reads after the decision (not counted as votes; after_decision = 1 in ocr_reads.csv):"]
            for j in all_jobs:
                if j.result is not None:
                    lines.append(f"[seq {j.seq}] roll #{j.track_id}  decided ply {j.ply or '?'}  "
                                 f"{j.after_reads} reads after: ply reads {reads_txt(j.after['ply']) or 'none'}; "
                                 f"range reads {reads_txt(j.after['range']) or 'none'}")
        (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
        print(f"output written to {run_dir}")

    atexit.register(finish_log)

    jobs = deque()  # RollJobs in arrival order; jobs[0] is the next one to report
    late = deque()  # reported rolls still read after their decision (in view, or crops waiting / being read)
    all_jobs = []   # every roll, for the after-decision lines of summary.txt
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

    def most_read_in_sheet(job):
        """(under the lock) -> (text, reads) of the in-sheet ply read most often: reads are counted per lookup key
        ('85' and '85W' together), tie -> the key read most recently; text = that key's most read spelling.
        (None, 0) if no read is in the sheet."""
        keys = {}  # lookup key -> [reads, latest frame]
        for t, n in job.votes["ply"].items():
            k = ply_key(t)
            if picker.in_sheet(k):
                e = keys.setdefault(k, [0, 0])
                e[0], e[1] = e[0] + n, max(e[1], job.last_frame["ply"][t])
        if not keys:
            return None, 0
        best = max(keys, key=lambda k: tuple(keys[k]))
        return max((t for t in job.votes["ply"] if ply_key(t) == best), key=lambda t: job.votes["ply"][t]), keys[best][0]

    def current_ply(job):
        """(under the lock) Ply the roll would get now: the most read in-sheet ply, else the most read text."""
        return most_read_in_sheet(job)[0] or top(job.votes["ply"])[0]

    def choose_ply(job):
        """(under the lock) -> (ply, rule). The in-sheet ply read most often -- unless it was read fewer than
        --min-reads times and another value is confirmed, then that confirmed value (not in the sheet); nothing
        in the sheet: the latest confirmed value (--votes or fast confirm); else (None, 'unreadable')."""
        ply, n = most_read_in_sheet(job)
        confirmed = job.confirmed_at["ply"]
        if ply is not None and (n >= args.min_reads or not confirmed):
            return ply, "most_read_in_sheet"
        if confirmed:
            return max(confirmed, key=confirmed.get), "confirmed_not_in_sheet"
        return None, "unreadable"

    def look_alikes(key):
        """Sheet plys that the recognizer's usual misreads turn into `key` (sorted)."""
        return sorted(c for c in written_candidates(key) if picker.in_sheet(c))

    def range_tie(job, key):
        """(under the lock) Range tie-break -> (key to look up, note or ''). Among `key` and its look-alikes in
        the sheet, the one whose sheet ranges are closest to the roll's range reads (average digits off per read,
        each read counted --range-max-diff + 1 at most). `key` stays unless a look-alike is at least --tie-margin
        closer and there are --tie-min-reads range reads."""
        reads = job.votes["range"]
        others = look_alikes(key)
        if not others or sum(reads.values()) < args.tie_min_reads:
            return key, ""
        cap, total = args.range_max_diff + 1, sum(reads.values())

        def off(ply):
            ranges = picker.ranges(ply)
            return sum(n * min(cap, min(range_diff(t, s, e) for s, e in ranges)) for t, n in reads.items()) / total

        best = min(others, key=lambda c: (off(c), c))
        if off(key) - off(best) >= args.tie_margin:
            return best, f"+range_tie({key}->{best})"
        return key, ""

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
        # look-alikes in the sheet: give the range reads --tie-wait seconds to arrive, so they can decide
        if (args.tie_wait > 0 and sum(job.votes["range"].values()) < args.tie_min_reads
                and time.time() - job.confirmed_at["ply"][ply] < args.tie_wait and look_alikes(ply_key(ply))):
            return None
        ranges = picker.ranges(ply_key(ply))
        rng = None
        if len(ranges) > 1:  # case 3
            near = {t: min(range_diff(t, s, e) for s, e in ranges) for t in job.confirmed_at["range"]}
            near = {t: d for t, d in near.items() if d <= args.range_max_diff}
            if not near:
                return None
            rng = min(near, key=lambda t: (near[t], -job.votes["range"][t]))
        return {"ply": ply, "rule": "confirmed_in_sheet", "rng": rng}

    STATUS_TXT = {"unreadable": "UNREADABLE", "not_in_master": "NOT IN MASTER", "duplicate": "DUPLICATE",
                  "range_unreadable": "RANGE UNREADABLE", "range_no_match": "RANGE NO MATCH"}

    def show_result(job):
        """(under the lock) A just-decided roll -> the top panel of the video + one row of results_table.csv."""
        res = job.result
        row = res.get("row") if res["status"] == "picked" else None
        rng = job.rng if job.rng is not None else top(job.votes["range"])[0]
        text = f"seq {job.seq}  #{job.track_id}  Ply {job.ply or '?'} | {rng or '?'}  " + \
               (f"-> BIN {row['bin']}" if row else STATUS_TXT.get(res["status"], res["status"].upper()))
        shown.append((f"seq {job.seq}   #{job.track_id}     Ply {job.ply or '?'}   |   {rng or '?'}",
                      f"BIN {row['bin']}" if row else STATUS_TXT.get(res["status"], res["status"].upper()),
                      C_PICKED if row else C_ERROR))
        if table_log and not log_files[3].closed:
            frame = video["frame"]
            raw = "; ".join(f"{e['field']}: {e['text'] or '-'} {e['score']:.2f} @f{e['frame']}"
                            for e in sorted(job.live, key=lambda e: e["rel"][1]))
            table_log.writerow([job.seq, job.track_id, res["status"], job.ply or "", rng or "",
                                row["bin"] if row else "",
                                row["relay"] if row and row["relay"] is not None else "", text, frame,
                                f"{(frame - 1) / FPS:.2f}" if is_file and frame else "",
                                f"{time.time() - t_start:.2f}", f"{job.decided_after:.2f}", job.rule, job.reads,
                                reads_txt(job.votes["ply"]), reads_txt(job.votes["range"]), raw])
            log_files[3].flush()

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
            switched = ""
            if picker.in_sheet(key):  # look-alike sheet plys: the range reads decide between them
                with cond:
                    key, switched = range_tie(job, key)
            if switched:
                ply, rule = key, rule + switched
            ranges = picker.ranges(key)
            if len(ranges) > 1 and (switched or not plan):  # case 3
                with cond:
                    rng = choose_range(job, ranges)
            res = picker.pick(key, rng)
        with cond:
            job.ply, job.rule, job.rng = ply, rule, rng
            job.result = res
            job.decided_after = time.time() - job.first_seen
            show_result(job)
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
        if log:
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

    def read_order():
        """(under the lock) Rolls in the order the OCR workers serve them: undecided rolls in seq order first (they
        decide the bins), then decided ones (after-decision reads, logged only): reported rolls, then decided rolls
        still waiting to be reported (not the head of the queue, so its report is not held up by reads)."""
        undecided = [j for j in jobs if j.result is None]
        if args.stop_after_decision:
            return undecided
        return undecided + list(late) + [j for j in list(jobs)[1:] if j.result is not None]

    def unfinished():
        """(under the lock) Anything left to read or report: rolls in the queue, or after-decision crops."""
        return bool(jobs) or any(j.busy or any(j.crops.values()) for j in late)

    def next_item():
        """Read ahead: the oldest waiting crop of the first roll in read_order() (ply before range)."""
        while late and late[0].closed and not late[0].busy and not any(late[0].crops.values()):
            late.popleft()  # left the frame and fully read
        for job in read_order():
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
                after = job.result is not None  # taken after the decision: logged only, never a vote
                t_taken = time.perf_counter()
                workers[worker].update(job=job, field=item[0], frame=item[3], t0=t_taken,  # for the panel / sidebar
                                       crop=as_read(item[1], item[2]))

            field, crop, rotation, crop_frame, timing = item
            t0 = time.perf_counter()
            text, score, text2, score2, used = reader.read(crop, rotation)
            ocr_ms = (time.perf_counter() - t0) * 1000
            if rotation is None and used == 180:  # way up unknown: the higher-scoring way is used
                text, score, text2, score2 = text2, score2, text, score
            # every non-empty read is a vote (no score threshold); "auto" boxes are only logged;
            # single-class "text" boxes vote for the field their reading shows
            voted = text_field(text) if field == "text" else field
            vote = bool(text) and voted in ("ply", "range") and not after
            crop_name = snap_name = ""
            seen = as_read(crop, used)  # the crop the right way up for the read that was used
            if run_dir:
                with cond:
                    stats["crops"] += 1
                    n_crop = stats["crops"]
                crop_name = f"crops/s{job.seq:03d}_f{crop_frame:05d}_{field}_{n_crop:06d}.png"
                snap_name = "snapshots/" + crop_name.split("/", 1)[1]
                # saved the way it was read when the way up is known, as cut out when both ways were read
                cv2.imwrite(str(run_dir / crop_name), crop if rotation in (None, 0) else cv2.rotate(crop, cv2.ROTATE_180))
                lines = [f"read: '{text}'  {score:.2f}" if text else f"read: (nothing)  {score:.2f}",
                         f"seq {job.seq}  roll #{job.track_id}  box {field}"
                         + (f" -> {voted}" if field == "text" else "")
                         + ("   AFTER DECISION (no vote)" if after else "   VOTE" if vote else "   no vote"),
                         f"frame {crop_frame}   OCR {ocr_ms:.0f} ms   worker {worker}   "
                         + ("way up from layout" if rotation is not None else f"read both ways, used {used} deg")]
                if text2 is not None:
                    lines.append(f"other way up: '{text2}'  {score2:.2f}")
                color = (C_TEXTBOX if text else C_ERROR) if after else C_PICKED if vote else C_ERROR
                cv2.imwrite(str(run_dir / snap_name), snapshot(seen, lines, color))
            with cond:
                job.reads += 1
                job.busy -= 1
                workers[worker].update(job=None, last=(job.seq, voted or field, text, score, ocr_ms), last_crop=seen)
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
                                      f"{(time.perf_counter() - timing['t_det']) * 1000:.1f}" if "t_det" in timing else "",
                                      snap_name, int(after)])
                    log_files[1].flush()
                if after:
                    job.after_reads += 1
                    if text and voted in ("ply", "range"):
                        job.after[voted][text] += 1
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
                if "rel" in timing:  # live (raw) read of this box for the video, vote or not
                    job.set_live(timing["rel"], timing["size"], (voted or "text") if field == "text" else field,
                                 text, score, crop_frame)
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
                        if not args.stop_after_decision:
                            late.append(job)  # its boxes are still read, after-decision reads
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

    WINDOW = "RTracker OCR queue v13"
    SIDEBAR_ITEMS = 24  # most waiting crops collected for the sidebar (it shows as many as fit)

    def render(frame, fps, frame_no, ended=False):
        """Video frame with rolls / text boxes drawn -> the picture shown and recorded: top panel, 'VIDEO ENDED'
        banner after the source ended, and the OCR sidebar on the right (the picture is widened for it)."""
        fh, fw = frame.shape[:2]
        k = ui_scale(fw)
        with cond:
            last = shown[-1] if shown else None
            chips, head = [], next((j for j in jobs if j.result is None), None)  # head: the roll being decided
            for j in jobs:
                left = {f: len(q) for f, q in j.crops.items() if q}
                txt = (f"s{j.seq} #{j.track_id}  "
                       + (" ".join(f"{f[0].upper()}{n}" for f, n in left.items()) or "0")
                       + ("  left" if j.closed else ""))
                chips.append((txt, state_color(j.result), j is head))
            lanes, reading, now_pc = [], [], time.perf_counter()
            for w, s in workers.items():
                last_txt = ""
                if s["last"]:
                    seq_, fld, text, score, ms_ = s["last"]
                    last_txt = f"last   s{seq_} {fld}  '{text}'  {score:.2f}   {ms_:.0f} ms"
                if s["job"] is not None:
                    ms_now = (now_pc - s["t0"]) * 1000
                    after_txt = "  after decision" if s["job"].result is not None else ""
                    txt = (f"OCR {w}   reading s{s['job'].seq} #{s['job'].track_id} {s['field']}   "
                           f"crop f{s['frame']}   {ms_now:.0f} ms{after_txt}")
                    reading.append((True, ascii_text(f"OCR {w}  s{s['job'].seq} {s['field']}  f{s['frame']}  "
                                                     f"{ms_now:.0f} ms" + (" (after)" if after_txt else "")),
                                    s["crop"]))
                else:
                    txt = f"OCR {w}   idle"
                    side = f"OCR {w}  idle" + (f"  last '{s['last'][2]}' {s['last'][3]:.2f}" if s["last"] else "")
                    reading.append((False, ascii_text(side), s["last_crop"]))
                lanes.append((s["job"] is not None, txt, last_txt))
            # waiting crops in the order the workers take them (next_item): undecided rolls in seq order, then
            # decided rolls (after-decision reads); ply before range before text, oldest first
            waiting, n_waiting = [], 0
            for j in read_order():
                for f in ("ply", "range", "text", "auto"):
                    q = j.crops[f]
                    n_waiting += len(q)
                    for crop, rotation, crop_frame, _ in q:
                        if len(waiting) >= SIDEBAR_ITEMS:
                            break
                        waiting.append((f"s{j.seq} #{j.track_id} {f}  f{crop_frame}"
                                        + ("  after decision" if j.result is not None else ""),
                                        as_read(crop, rotation)))
            n_busy = sum(s["job"] is not None for s in workers.values())
            busy_left = unfinished()
        draw_panel(frame, fps, frame_no, last, chips, lanes)
        if ended:
            banner(frame, f"VIDEO ENDED  -  OCR still reading: {n_busy} now, {n_waiting} waiting"
                   if busy_left else "VIDEO ENDED  -  all reads done", k)
        if args.sidebar <= 0:
            return frame
        canvas = np.empty((fh, fw + int(args.sidebar * k), 3), np.uint8)
        canvas[:, :fw] = frame
        draw_sidebar(canvas, fw, k, reading, waiting, n_waiting)
        return canvas

    tracks, writer, frame_no, t_fps, fps, seq = [], None, 0, time.time(), 0.0, 0
    last_base, quit_ = None, False
    for frame in frames_15fps(cap, is_file):
        frame_no += 1
        video["frame"] = frame_no
        fh, fw = frame.shape[:2]
        k = ui_scale(fw)
        marks = []  # text boxes + their live reads, drawn after the rolls so the roll fill doesn't tint them
        clean = frame.copy()  # crops are cut from this, so nothing drawn on `frame` gets into them
        t0 = time.perf_counter()
        if roi:  # detect inside the ROI only (more pixels on the text), then shift back to frame coords
            rx, ry = max(0, roi[0]), max(0, roi[1])
            rw, rh = min(roi[2], fw - rx), min(roi[3], fh - ry)
            dets = [(c, s, (r[0] + rx, r[1] + ry, r[2], r[3], r[4]))
                    for c, s, r in detector.detect(frame[ry:ry + rh, rx:rx + rw])]
            dashed_rect(frame, (rx, ry), (rx + rw, ry + rh), C_ROI, k)
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
                # every box on every frame -- also after the decision (logged only), unless --stop-after-decision
                if job is None or job.result is None or not args.stop_after_decision:
                    crop = crop_rect(clean, rect)
                    if crop.size:
                        with cond:
                            if t.job is None:  # first text on this roll: it joins the queue
                                seq += 1
                                t.job = RollJob(seq, t.id)
                                jobs.append(t.job)
                                all_jobs.append(t.job)
                            if t.job.closed and t.job.result is None:  # seen again before it was decided
                                t.job.closed = False
                            waiting = sum(len(q) for j in jobs for q in j.crops.values()) + 1
                            t.job.add(field, crop, rotation, frame_no,
                                      {"conf": conf, "det_ms": det_ms, "t_det": t0, "queue_len": waiting,
                                       "t_join": time.perf_counter(),
                                       "rel": (rect[0] - t.rect[0], rect[1] - t.rect[1]),
                                       "size": min(rect[2], rect[3])})
                            cond.notify_all()
                if not text_only:
                    marks.append((rect, None))
                if t.job is not None:  # latest raw OCR read of this box, changes with every read
                    with cond:
                        live = t.job.live_slot((rect[0] - t.rect[0], rect[1] - t.rect[1]), min(rect[2], rect[3]))
                        live = dict(live) if live else None
                    if live:
                        marks.append((rect, f"{live['field']}  {live['text'] or '-'}  {live['score']:.2f}"))

        # --- draw rolls with their queue state ---
        for t in tracks:
            if t.missed or (text_only and t.job is None):  # text-only: hide text that isn't a reading
                continue
            job = t.job  # final results go to the top panel; the roll box only shows its state
            if job is None:
                label, color = f"#{t.id}", C_PENDING
            else:
                with cond:
                    res = job.result
                label, color = f"#{t.id}   seq {job.seq}", state_color(res)
            draw_roll(frame, t.rect, color, label, k)
        for rect, text in marks:
            if text is None:
                cv2.polylines(frame, [np.int32(rect_points(rect))], True, C_TEXTBOX, 1, AA)
            else:
                tag(frame, text, rect_bounds(rect)[:2], k, C_FG, C_CARD, C_TEXTBOX, scale=0.48)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_fps, 1e-6)
        t_fps = now
        last_base = frame.copy()  # rolls and text boxes drawn, no panel: shown again after the video ends
        out = render(frame, fps, frame_no)
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (out.shape[1], out.shape[0]))
        writer.write(out)
        if not args.no_show:
            cv2.imshow(WINDOW, out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit_ = True
                break

    # source ended: close the rolls still in view and let the queue finish in order
    with cond:
        for t in tracks:
            if t.job:
                t.job.closed = True
        cond.notify_all()
    try:
        if last_base is not None and not quit_:
            # keep the last frame on screen (and in the video) with the live panel + sidebar until the queue is done
            with cond:
                waiting_ = unfinished()
            if waiting_:
                print("source ended -- reading the rest of the queue (q in the video window stops waiting)")
            while waiting_:
                t_tick = time.time()
                out = render(last_base.copy(), fps, frame_no, ended=True)
                writer.write(out)
                if not args.no_show:
                    cv2.imshow(WINDOW, out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(max(0.0, 1 / FPS - (time.time() - t_tick)))
                with cond:
                    waiting_ = unfinished()
            else:
                time.sleep(0.2)  # the last report is written just after its job leaves the queue
                out = render(last_base.copy(), fps, frame_no, ended=True)
                for _ in range(int(FPS)):  # one second of the final state in the video
                    writer.write(out)
                if not args.no_show:
                    cv2.imshow(WINDOW, out)
                    cv2.waitKey(1)
        else:
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
    if csv_file:
        csv_file.close()
    lights.close()
    for model in (reader, detector):  # Hailo models first, then the shared device
        if hasattr(model, "configured"):
            model.close()
    if device is not None:
        device.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
