"""
Document OCR Pipeline
=====================
Document Image → YOLO (detect regions) → Crop → OCR → Structured JSON

Supports:
  - YOLOv8 (ultralytics) for layout/field detection
  - Tesseract, EasyOCR, or PaddleOCR as the OCR backend
  - Outputs structured JSON with bounding boxes, labels, and text

Install dependencies:
  pip install ultralytics easyocr pytesseract paddleocr opencv-python pillow

For Tesseract, also install the binary:
  Ubuntu:  sudo apt install tesseract-ocr
  macOS:   brew install tesseract
  Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, asdict
from typing import Literal, Optional
import time


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class DetectedField:
    field_id: int
    label: str
    confidence: float
    bbox: dict          # {x1, y1, x2, y2}  (absolute pixels)
    bbox_norm: dict     # {x1, y1, x2, y2}  (normalised 0-1)
    ocr_text: str
    ocr_engine: str


@dataclass
class PipelineResult:
    source_image: str
    image_width: int
    image_height: int
    ocr_engine: str
    num_fields: int
    processing_time_s: float
    fields: list


# ─────────────────────────────────────────────
# OCR backends
# ─────────────────────────────────────────────

class TesseractOCR:
    def __init__(self, lang: str = "eng"):
        import pytesseract
        self.tess = pytesseract
        self.lang = lang

    def read(self, image_crop: np.ndarray) -> str:
        pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        return self.tess.image_to_string(pil, lang=self.lang).strip()


class EasyOCR_Backend:
    def __init__(self, lang_list: list = None):
        import easyocr
        self.reader = easyocr.Reader(lang_list or ["en"], gpu=False)

    def read(self, image_crop: np.ndarray) -> str:
        results = self.reader.readtext(image_crop, detail=0)
        return " ".join(results).strip()


class PaddleOCR_Backend:
    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def read(self, image_crop: np.ndarray) -> str:
        result = self.ocr.ocr(image_crop, cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append(line[1][0])
        return " ".join(texts).strip()


def get_ocr_engine(name: str):
    name = name.lower()
    if name == "tesseract":
        return TesseractOCR()
    elif name == "easyocr":
        return EasyOCR_Backend()
    elif name == "paddleocr":
        return PaddleOCR_Backend()
    else:
        raise ValueError(f"Unknown OCR engine: {name}. Choose tesseract | easyocr | paddleocr")


# ─────────────────────────────────────────────
# YOLO detector
# ─────────────────────────────────────────────

class YOLODetector:
    """
    Wraps YOLOv8 (ultralytics).
    Pass a custom .pt model trained on document layouts, or use the default
    YOLOv8n weights as a stand-in (detects generic objects).
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.class_names = self.model.names  # dict {id: name}

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
          {label, confidence, x1, y1, x2, y2}
        """
        results = self.model(image, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "label": self.class_names[cls_id],
                "confidence": float(box.conf[0]),
                "x1": int(box.xyxy[0][0]),
                "y1": int(box.xyxy[0][1]),
                "x2": int(box.xyxy[0][2]),
                "y2": int(box.xyxy[0][3]),
            })
        return detections


# ─────────────────────────────────────────────
# Fallback: simple contour-based region detector
# (use when no custom YOLO model is available)
# ─────────────────────────────────────────────

class ContourDetector:
    """
    Lightweight fallback detector using morphological operations + contours.
    Useful for forms, tables, or structured docs without a trained YOLO model.
    """

    def __init__(self, min_area: int = 2000):
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold → find dark text blocks
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )
        # Dilate to merge nearby text into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 6))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area:
                continue
            detections.append({
                "label": "text_region",
                "confidence": 1.0,
                "x1": x, "y1": y,
                "x2": x + w, "y2": y + h,
            })
        # Sort top-to-bottom, left-to-right (reading order)
        detections.sort(key=lambda d: (d["y1"] // 50, d["x1"]))
        return detections


# ─────────────────────────────────────────────
# Cropping + padding
# ─────────────────────────────────────────────

def crop_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2]


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    ocr_engine_name: str = "easyocr",
    detector: Literal["yolo", "contour"] = "contour",
    yolo_model_path: str = "yolov8n.pt",
    yolo_conf: float = 0.25,
    crop_padding: int = 5,
    output_json: Optional[str] = None,
    save_annotated: bool = False,
) -> PipelineResult:

    t0 = time.time()
    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"

    # ── Load image ──────────────────────────────────────
    image = cv2.imread(str(image_path))
    assert image is not None, f"Could not read image: {image_path}"
    h, w = image.shape[:2]
    print(f"[1/4] Loaded image: {image_path.name}  ({w}×{h})")

    # ── Detect regions ──────────────────────────────────
    if detector == "yolo":
        det = YOLODetector(model_path=yolo_model_path, conf_threshold=yolo_conf)
        print(f"[2/4] Running YOLOv8 detection (model={yolo_model_path}) …")
    else:
        det = ContourDetector()
        print("[2/4] Running contour-based detection (fallback) …")

    detections = det.detect(image)
    print(f"      → {len(detections)} region(s) found")

    # ── Load OCR engine ─────────────────────────────────
    print(f"[3/4] Loading OCR engine: {ocr_engine_name} …")
    ocr = get_ocr_engine(ocr_engine_name)

    # ── Crop + OCR ──────────────────────────────────────
    print("[4/4] Cropping and running OCR on each region …")
    fields = []
    annotated = image.copy()

    for idx, det_box in enumerate(detections):
        x1, y1, x2, y2 = det_box["x1"], det_box["y1"], det_box["x2"], det_box["y2"]
        crop = crop_region(image, x1, y1, x2, y2, padding=crop_padding)

        text = ""
        try:
            text = ocr.read(crop)
        except Exception as e:
            text = f"[OCR ERROR: {e}]"

        field = DetectedField(
            field_id=idx,
            label=det_box["label"],
            confidence=round(det_box["confidence"], 4),
            bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            bbox_norm={
                "x1": round(x1 / w, 4), "y1": round(y1 / h, 4),
                "x2": round(x2 / w, 4), "y2": round(y2 / h, 4),
            },
            ocr_text=text,
            ocr_engine=ocr_engine_name,
        )
        fields.append(field)
        print(f"      [{idx:03d}] {det_box['label']:20s} → \"{text[:60]}{'…' if len(text)>60 else ''}\"")

        # Annotate image
        color = (0, 200, 80)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{idx}:{det_box['label']}", (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    elapsed = round(time.time() - t0, 2)
    result = PipelineResult(
        source_image=str(image_path.resolve()),
        image_width=w,
        image_height=h,
        ocr_engine=ocr_engine_name,
        num_fields=len(fields),
        processing_time_s=elapsed,
        fields=[asdict(f) for f in fields],
    )

    # ── Save outputs ────────────────────────────────────
    out_dir = Path(output_json).parent if output_json else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(output_json) if output_json else out_dir / (image_path.stem + "_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    print(f"\n✅  JSON saved → {json_path}")

    if save_annotated:
        ann_path = out_dir / (image_path.stem + "_annotated.jpg")
        cv2.imwrite(str(ann_path), annotated)
        print(f"✅  Annotated image saved → {ann_path}")

    print(f"⏱   Total time: {elapsed}s  |  {len(fields)} field(s) extracted\n")
    return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document Image → YOLO/Contour → OCR → JSON pipeline"
    )
    parser.add_argument("image", help="Path to the input document image")
    parser.add_argument(
        "--ocr", default="easyocr",
        choices=["tesseract", "easyocr", "paddleocr"],
        help="OCR backend to use (default: easyocr)"
    )
    parser.add_argument(
        "--detector", default="contour",
        choices=["yolo", "contour"],
        help="Region detector: 'yolo' (requires ultralytics) or 'contour' fallback (default: contour)"
    )
    parser.add_argument("--yolo-model", default="yolov8n.pt",
                        help="Path to YOLO .pt weights (default: yolov8n.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--padding", type=int, default=5,
                        help="Pixel padding around each crop (default: 5)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <image_stem>_result.json)")
    parser.add_argument("--annotate", action="store_true",
                        help="Save annotated image with bounding boxes")

    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        ocr_engine_name=args.ocr,
        detector=args.detector,
        yolo_model_path=args.yolo_model,
        yolo_conf=args.yolo_conf,
        crop_padding=args.padding,
        output_json=args.output,
        save_annotated=args.annotate,
    )