"""YOLOv8n-OBB detector -- version 3. Hailo-8L (.hef) on the Raspberry Pi 5, Ultralytics (.pt) for testing on a PC.

New in v3 (detector.py and detector_v2.py are unchanged): the Hailo-8L can be SHARED with the text recognizer
(ocr_hailo_v1.HailoReader), two .hef files on one chip:
  open_hailo_device()   one VDevice with the HailoRT scheduler (round robin); the scheduler switches the chip between
                        the detector and the recognizer on its own -- no model is activated by hand
  load_detector(..., device=dev)   the detector uses that shared device; without device it opens its own and
                        activates its model like v2 (detector only on the chip)
Everything else (class mapping by name, single text class, decoding) is exactly detector_v2.

Returns a list of (class_id, score, rect) with rect = (cx, cy, w, h, theta) in frame pixels / radians,
normalized so w >= h (w runs along the text line).

New in v2 (detector.py is unchanged and still used by the older scripts): the model's own class list is mapped by
NAME onto the pipeline's ids, so a detector with other classes than 0=ply, 1=range, 2=roll works too:
  roll                      -> ROLL
  range                     -> RANGE
  ply, when there is range  -> PLY     (old 3-class models, e.g. rolls_v2.pt: fields told apart by the detector)
  ply, without range        -> TEXT    (single text class, both text boxes labelled "ply": the reading decides)
  any other name            -> TEXT
merge_text=True maps ply and range both to TEXT, e.g. for a model that still lists "range" but was trained with
every text box labelled "ply".
Class names: Ultralytics .pt carries them. A .hef does not: pass classes=[...] in the model's training order, or they
are taken from the number of class channels (3 -> ply, range, roll; 2 -> ply, roll -- Roboflow's alphabetical order).
TEXT is also what the no-training PaddleOCR text detector (ppocr_det.onnx) gives for extra text lines."""
import math

import cv2
import numpy as np

PLY, RANGE, ROLL, TEXT = 0, 1, 2, 3
CLASSES = ["ply", "range", "roll"]
DEFAULT_NAMES = {3: ["ply", "range", "roll"], 2: ["ply", "roll"]}


def open_hailo_device():
    """One Hailo-8L device shared by several .hef models; the HailoRT scheduler (round robin) switches between them."""
    from hailo_platform import VDevice, HailoSchedulingAlgorithm  # HailoRT 4.x (apt: hailo-all)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    return VDevice(params)


def class_map(names, merge_text=False):
    """Model class names (index order) -> list of pipeline ids (PLY / RANGE / ROLL / TEXT)."""
    names = [str(n).strip().lower() for n in names]
    split = "range" in names and not merge_text
    ids = []
    for n in names:
        if n == "roll":
            ids.append(ROLL)
        elif n == "range" and split:
            ids.append(RANGE)
        elif n == "ply" and split:
            ids.append(PLY)
        else:
            ids.append(TEXT)
    return ids


def norm_rect(cx, cy, w, h, theta):
    """Long side along theta, theta in [-pi/2, pi/2) so the long axis always points right."""
    if h > w:
        w, h, theta = h, w, theta + math.pi / 2
    return float(cx), float(cy), float(w), float(h), (theta + math.pi / 2) % math.pi - math.pi / 2


def rect_points(rect):
    cx, cy, w, h, t = rect
    return cv2.boxPoints(((cx, cy), (w, h), math.degrees(t)))


def rect_bounds(rect):
    p = rect_points(rect)
    return (*p.min(0), *p.max(0))


def letterbox(frame, w, h):
    s = min(w / frame.shape[1], h / frame.shape[0])
    nw, nh = int(frame.shape[1] * s), int(frame.shape[0] * s)
    px, py = (w - nw) // 2, (h - nh) // 2
    img = np.full((h, w, 3), 114, np.uint8)
    img[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
    return img, s, px, py


def rotated_nms(dets, iou=0.5):
    """Class-wise rotated NMS on (cls, score, rect) tuples."""
    keep = []
    for c in {d[0] for d in dets}:
        ds = [d for d in dets if d[0] == c]
        boxes = [((r[0], r[1]), (r[2], r[3]), math.degrees(r[4])) for _, _, r in ds]
        idx = cv2.dnn.NMSBoxesRotated(boxes, [d[1] for d in ds], 0.0, iou)
        keep += [ds[i] for i in np.array(idx).flatten()]
    return keep


class HailoDetector:
    """HEF compiled from the YOLOv8-OBB ONNX: 9 raw head outputs (box DFL 64ch, class Nch, angle 1ch at 3 scales).
    Box/angle decoding and rotated NMS run here on the Pi CPU (only for anchors above the confidence threshold)."""

    def __init__(self, hef_path, conf=0.4, classes=None, merge_text=False, device=None):
        from hailo_platform import VDevice, FormatType  # HailoRT 4.x (apt: hailo-all)
        self.conf = conf
        self.shared = device is not None  # shared: the scheduler runs the model, no manual activation
        self.device = device if self.shared else VDevice()
        self.model = self.device.create_infer_model(hef_path)
        self.model.set_batch_size(1)
        for out in self.model.outputs:
            out.set_format_type(FormatType.FLOAT32)  # dequantized outputs
        self.configured = self.model.configure()
        if not self.shared:
            self.configured.activate()
        self.bindings = self.configured.create_bindings()
        self.h, self.w = self.model.input().shape[:2]
        self.outputs = {o.name: o.shape for o in self.model.outputs}  # (H, W, C)
        n_cls = next(s[2] for s in self.outputs.values() if s[2] not in (64, 1))
        self.names = list(classes) if classes else DEFAULT_NAMES.get(n_cls)
        if not self.names or len(self.names) != n_cls:
            raise SystemExit(f"{hef_path}: {n_cls} classes -- pass their names in training order (--classes)")
        self.ids = class_map(self.names, merge_text)

    def detect(self, frame):
        img, s, px, py = letterbox(frame, self.w, self.h)
        self.bindings.input().set_buffer(np.ascontiguousarray(img[:, :, ::-1]))  # BGR->RGB
        for name, shape in self.outputs.items():
            self.bindings.output(name).set_buffer(np.empty(shape, np.float32))
        self.configured.run([self.bindings], 1000)
        grids = {}  # grid size -> {"box"|"cls"|"ang": array}
        for name, shape in self.outputs.items():
            kind = {64: "box", 1: "ang"}.get(shape[2], "cls")
            grids.setdefault(shape[0], {})[kind] = self.bindings.output(name).get_buffer()
        dets = decode_obb(grids, self.h, self.conf)
        return [(self.ids[c], sc, ((r[0] - px) / s, (r[1] - py) / s, r[2] / s, r[3] / s, r[4]))
                for c, sc, r in dets]

    def close(self):
        if not self.shared:
            self.configured.deactivate()
        self.configured.shutdown()


def decode_obb(grids, input_size, conf):
    """Decode YOLOv8-OBB raw head outputs (NHWC per grid) the same way Ultralytics does, then rotated NMS.
    Class ids are the model's own (mapped by the caller)."""
    dets = []
    for g, out in grids.items():
        stride = input_size / g
        cls = 1 / (1 + np.exp(-out["cls"].reshape(g * g, -1)))
        score, c = cls.max(1), cls.argmax(1)
        keep = np.where(score >= conf)[0]
        if not len(keep):
            continue
        box = out["box"].reshape(g * g, 4, 16)[keep]
        box = np.exp(box - box.max(2, keepdims=True))
        dist = (box / box.sum(2, keepdims=True) * np.arange(16)).sum(2)  # DFL -> l, t, r, b
        ang = ((1 / (1 + np.exp(-out["ang"].reshape(g * g)[keep]))) - 0.25) * math.pi
        ax, ay = keep % g + 0.5, keep // g + 0.5
        xf, yf = (dist[:, 2] - dist[:, 0]) / 2, (dist[:, 3] - dist[:, 1]) / 2
        cx = (ax + xf * np.cos(ang) - yf * np.sin(ang)) * stride
        cy = (ay + xf * np.sin(ang) + yf * np.cos(ang)) * stride
        w, h = (dist[:, 0] + dist[:, 2]) * stride, (dist[:, 1] + dist[:, 3]) * stride
        dets += [(int(c[k]), float(score[k]), norm_rect(cx[i], cy[i], w[i], h[i], ang[i])) for i, k in enumerate(keep)]
    return rotated_nms(dets)


class UltralyticsDetector:
    def __init__(self, pt_path, conf=0.4, classes=None, merge_text=False):
        from ultralytics import YOLO
        self.model, self.conf = YOLO(pt_path), conf
        names = self.model.names  # {0: "ply", ...} from training
        self.names = list(classes) if classes else [names[i] for i in sorted(names)]
        self.ids = class_map(self.names, merge_text)

    def detect(self, frame):
        obb = self.model.predict(frame, conf=self.conf, imgsz=640, verbose=False)[0].obb
        return [(self.ids[int(c)], float(s), norm_rect(*r))
                for c, s, r in zip(obb.cls.tolist(), obb.conf.tolist(), obb.xywhr.tolist())]


class TextOnlyDetector:
    """No training needed (PC trial): PaddleOCR text detector finds text lines (class 3, read as ply or range
    by OCR); nearby lines are grouped into one box that stands in for the roll (class 2)."""
    text_only = True
    names = ["ply", "range", "roll", "text"]

    def __init__(self, onnx_path, conf=0.4, max_side=800):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp, self.conf, self.max_side = self.sess.get_inputs()[0].name, conf, max_side

    def detect(self, frame):
        fh, fw = frame.shape[:2]
        s = min(1.0, self.max_side / max(fh, fw))
        h, w = max(32, round(fh * s / 32) * 32), max(32, round(fw * s / 32) * 32)
        img = (cv2.resize(frame, (w, h)).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        prob = self.sess.run(None, {self.inp: img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        contours, _ = cv2.findContours((prob > 0.3).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < 16 or prob[y:y + bh, x:x + bw].mean() < 0.3:
                continue
            d = 1.5 * bw * bh / (2 * (bw + bh))  # expand box like PaddleOCR's unclip
            lines.append([(x - d) * fw / w, (y - d) * fh / h, (x + bw + d) * fw / w, (y + bh + d) * fh / h])

        # group lines that are close together (ply number above start-end) into one box
        groups = []  # [box, member lines]
        for box in lines:
            m = 0.7 * min(box[2] - box[0], box[3] - box[1])
            for g, members in groups:
                if box[0] - m < g[2] and box[2] + m > g[0] and box[1] - m < g[3] and box[3] + m > g[1]:
                    g[:] = [min(g[0], box[0]), min(g[1], box[1]), max(g[2], box[2]), max(g[3], box[3])]
                    members.append(box)
                    break
            else:
                groups.append([list(box), [box]])

        def rect(b, pad=0):
            return norm_rect((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad, 0.0)

        dets = []
        for g, members in groups:
            dets.append((ROLL, 1.0, rect(g, 20)))
            if len(members) >= 2:  # shortest line = ply number, longest line = start-end
                members.sort(key=lambda b: max(b[2] - b[0], b[3] - b[1]))
                dets += [(PLY, 1.0, rect(members[0])), (RANGE, 1.0, rect(members[-1]))]
                dets += [(TEXT, 1.0, rect(b)) for b in members[1:-1]]
            else:
                dets.append((TEXT, 1.0, rect(members[0])))
        return dets


def load_detector(path, conf=0.4, classes=None, merge_text=False, device=None):
    """classes: the model's class names in training order (needed for a .hef that is not 3- or 2-class);
    merge_text: treat ply and range boxes alike (TEXT) even if the model has both classes;
    device: a shared Hailo device from open_hailo_device() (only used for a .hef)."""
    if path.endswith(".hef"):
        return HailoDetector(path, conf, classes, merge_text, device)
    if path.endswith("ppocr_det.onnx"):
        return TextOnlyDetector(path, conf)
    return UltralyticsDetector(path, conf, classes, merge_text)
