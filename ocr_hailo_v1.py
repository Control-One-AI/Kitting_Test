"""PP-OCRv5 text recognizer on the Hailo-8L -- version 1 (the .hef from compile_hailo_rec_v1.py).

HailoReader has the same read() as FreeReader in rtracker_ocr_paddle_v5_queue_v7/v8.py, so the queue logic does not
change: greedy CTC over the model's whole dictionary, no character restriction, no format check.
  preprocessing  like the ONNX reader: height 48, width by aspect (max 320), BGR -> RGB, padded right to 320 with
                 grey; the (x - 127.5) / 127.5 normalization is inside the .hef, so the input is uint8
  output         40 time steps x N classes; softmax is applied here if the .hef stops before the model's softmax
  characters     <name>_chars.txt next to the .hef (written by compile_hailo_rec_v1.py), or --rec-chars
  device         the Hailo-8L shared with the detector: detector_v3.open_hailo_device() (HailoRT scheduler)
The OCR worker threads share one reader; reads go to the chip one at a time (lock)."""
import os
import threading

import cv2
import numpy as np

H, W = 48, 320


class HailoReader:
    def __init__(self, hef_path, device, chars_path=None):
        from hailo_platform import FormatType  # HailoRT 4.x (apt: hailo-all)
        chars_path = chars_path or os.path.splitext(hef_path)[0] + "_chars.txt"
        if not os.path.exists(chars_path):
            raise SystemExit(f"character list {chars_path} not found -- it is written by compile_hailo_rec_v1.py "
                             f"(or pass --rec-chars)")
        chars = open(chars_path, encoding="utf-8").read().splitlines()
        self.model = device.create_infer_model(hef_path)
        self.model.set_batch_size(1)
        for out in self.model.outputs:
            out.set_format_type(FormatType.FLOAT32)  # dequantized outputs
        self.configured = self.model.configure()  # shared device: the scheduler activates it, no activate() here
        self.bindings = self.configured.create_bindings()
        self.in_shape = tuple(self.model.input().shape)
        self.out_name, self.out_shape = self.model.outputs[0].name, tuple(self.model.outputs[0].shape)
        n_out = self.out_shape[-1]
        self.chars = [""] + chars + [" "] * max(0, n_out - 1 - len(chars))  # 0 = CTC blank, extra slot = space
        self.lock = threading.Lock()

    @staticmethod
    def prepare(crop):
        h, w = crop.shape[:2]
        new_w = min(W, max(16, int(H * w / h)))
        img = np.full((H, W, 3), 128, np.uint8)  # grey = 0.0 after normalization (the ONNX reader pads with 0.0)
        img[:, :new_w] = cv2.resize(crop, (new_w, H))[:, :, ::-1]
        return img

    def _read(self, crop):
        img = np.ascontiguousarray(self.prepare(crop).reshape(self.in_shape))
        with self.lock:
            self.bindings.input().set_buffer(img)
            self.bindings.output(self.out_name).set_buffer(np.empty(self.out_shape, np.float32))
            self.configured.run([self.bindings], 1000)
            probs = self.bindings.output(self.out_name).get_buffer().reshape(-1, len(self.chars)).copy()
        if abs(float(probs[0].sum()) - 1.0) > 0.05:  # logits (cut before the softmax): softmax here
            e = np.exp(probs - probs.max(1, keepdims=True))
            probs = e / e.sum(1, keepdims=True)
        best, conf = probs.argmax(1), probs.max(1)
        text, scores, prev = "", [], 0
        for k, p in zip(best, conf):
            if k != prev and k != 0:
                text += self.chars[k] if k < len(self.chars) else "?"
                scores.append(p)
            prev = k
        return text.strip(), float(np.mean(scores)) if scores else 0.0

    def read(self, crop, rotation=None):
        """-> (text, score, flipped_text, flipped_score, used), same as FreeReader.read."""
        if crop.size == 0:
            return "", 0.0, None, None, rotation or 0
        if rotation is not None:
            text, score = self._read(crop if rotation == 0 else cv2.rotate(crop, cv2.ROTATE_180))
            return text, score, None, None, rotation
        text, score = self._read(crop)
        text2, score2 = self._read(cv2.rotate(crop, cv2.ROTATE_180))
        return text, score, text2, score2, 180 if score2 > score else 0

    def close(self):
        self.configured.shutdown()
