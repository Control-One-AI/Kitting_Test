"""Compile the PP-OCRv5 mobile text recognizer (ONNX) to a .hef for the Hailo-8L -- version 1.
Run on x86 Linux or WSL2 with the Hailo Dataflow Compiler (hailo_sdk_client), from this folder, like compile_hailo.py.
The DFC version must match the HailoRT version on the Pi.

  python compile_hailo_rec_v1.py                                    # optimize + compile the fine-tuned v1 model
  python compile_hailo_rec_v1.py --onnx models/v5/ppocr_rec_fine_tuned_v2.onnx
  python compile_hailo_rec_v1.py --stage optimize                   # only parse + quantize -> .har (the slow part)
  python compile_hailo_rec_v1.py --stage compile                    # only .har -> .hef (e.g. after a failed compile)

For <name>.onnx it writes, next to it (never overwrites):
  <name>_optimized.har   quantized model (reuse it with --stage compile)
  <name>.hef             for ocr_hailo_v1.HailoReader / rtracker_ocr_paddle_v5_queue_v9.py --rec-model
  <name>_chars.txt       the model's character list (a .hef has no metadata; the reader needs it to decode)

Input: fixed 1x3x48x320 (the pipeline pads every crop to 320 px wide). On the chip the input is RGB uint8 0-255;
normalization (x - 127.5) / 127.5 is part of the model (= the ONNX reader's (x / 255 - 0.5) / 0.5).
Calibration: real crops from ocr_crops_v2 (train + val split, never the augmented copies or test), prepared exactly
like the pipeline does before reading: height 48, width by aspect (max 320), RGB, padded right with grey.
Output: 40 time steps x N classes (softmax); greedy CTC decoding runs on the Pi CPU in ocr_hailo_v1.

If parsing stops at an unsupported layer (the SVTR neck has attention: MatMul / Softmax / LayerNorm), the DFC prints
the node names it can reach; pass the last supported one with --end-nodes and the reader applies softmax on the CPU
when the output is not already probabilities (anything after that node would still need a CPU part -- ask first).
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

H, W = 48, 320


def prepare(crop):
    """Crop (BGR) -> RGB uint8 48x320, exactly like ocr_hailo_v1.HailoReader before a read."""
    h, w = crop.shape[:2]
    new_w = min(W, max(16, int(H * w / h)))
    img = np.full((H, W, 3), 128, np.uint8)  # grey = 0.0 after normalization, like the ONNX reader's zero padding
    img[:, :new_w] = cv2.resize(crop, (new_w, H))[:, :, ::-1]
    return img


def calibration_set(crops_dir, labels, n):
    rows = [r for r in csv.DictReader(open(labels, encoding="utf-8-sig")) if r.get("split") in ("train", "val")]
    files = [os.path.join(crops_dir, r["file"]) for r in rows] if rows else \
        sorted(glob.glob(os.path.join(crops_dir, "images", "*.png")))
    step = max(1, len(files) // n)
    imgs = [prepare(c) for c in (cv2.imread(f) for f in files[::step][:n]) if c is not None]
    return np.stack(imgs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/v5/ppocr_rec_fine_tuned.onnx")
    ap.add_argument("--stage", choices=["all", "optimize", "compile"], default="all")
    ap.add_argument("--crops", default="ocr_crops_v2", help="calibration crops")
    ap.add_argument("--labels", default="ocr_crops_v2/labels_v2.csv", help="for the split column")
    ap.add_argument("--calib", type=int, default=512, help="calibration images")
    ap.add_argument("--end-nodes", default=None, help="comma-separated ONNX node names to cut at (default: full model)")
    ap.add_argument("--opt-level", type=int, default=1, help="optimization_level (1 = CPU-friendly, as for rolls)")
    args = ap.parse_args()

    base = os.path.splitext(args.onnx)[0]
    har, hef, chars_txt = base + "_optimized.har", base + ".hef", base + "_chars.txt"
    from hailo_sdk_client import ClientRunner  # imported here so --help works without the DFC

    if args.stage in ("all", "optimize"):
        for p in (har, chars_txt):
            if os.path.exists(p):
                raise SystemExit(f"{p} already exists -- versions are never overwritten (rename it or the .onnx)")
        import onnxruntime as ort
        chars = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]) \
            .get_modelmeta().custom_metadata_map.get("character")
        if not chars:
            raise SystemExit(f"{args.onnx} has no 'character' metadata -- pass a PaddleOCR-exported recognizer")
        with open(chars_txt, "w", encoding="utf-8", newline="\n") as f:
            f.write(chars.strip("\n") + "\n")
        print(f"characters ({len(chars.splitlines())}) -> {chars_txt}")

        runner = ClientRunner(hw_arch="hailo8l")
        kw = {"end_node_names": args.end_nodes.split(",")} if args.end_nodes else {}
        runner.translate_onnx_model(args.onnx, "ppocr_rec", start_node_names=["x"],
                                    net_input_shapes={"x": [1, 3, H, W]}, **kw)
        runner.load_model_script(
            "normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])\n"
            f"model_optimization_flavor(optimization_level={args.opt_level}, compression_level=0)\n")
        print("parsed")
        calib = calibration_set(args.crops, args.labels, args.calib)
        print(f"calibration: {calib.shape} from {args.crops}")
        runner.optimize(calib)
        runner.save_har(har)
        print(f"optimized -> {har}")

    if args.stage in ("all", "compile"):
        if os.path.exists(hef):
            raise SystemExit(f"{hef} already exists -- versions are never overwritten")
        runner = ClientRunner(har=har)
        runner.load_model_script("performance_param(compiler_optimization_level=1)\n"
                                 "allocator_param(timeout=600, cluster_timeout=600, splitter_timeout=900)\n")
        with open(hef, "wb") as f:
            f.write(runner.compile())
        print(f"compiled -> {hef}   (characters: {chars_txt})")


if __name__ == "__main__":
    main()
