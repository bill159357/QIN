from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = PROJECT_ROOT / "resources" / "deepjiandu"
DEFAULT_SOURCE_DIR = RESOURCE_ROOT

DEFAULT_SCORE_THRESH = 0.30
DEFAULT_TOPK_DETECT = 300
DEFAULT_NMS_IOU = 0.35
DEFAULT_MIN_BOX_SIZE = 8.0
DEFAULT_MAX_ASPECT = 1.8
DEFAULT_REC_TOP_K = 5


def _coerce_float(
    value: Any,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _coerce_int(
    value: Any,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def normalize_deepjiandu_options(raw_options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = raw_options or {}
    return {
        "score_thresh": _coerce_float(
            options.get("jiandu_score_thresh"),
            DEFAULT_SCORE_THRESH,
            min_value=0.01,
            max_value=0.95,
        ),
        "topk_detect": _coerce_int(
            options.get("jiandu_topk_detect"),
            DEFAULT_TOPK_DETECT,
            min_value=1,
            max_value=2000,
        ),
        "nms_iou": _coerce_float(
            options.get("jiandu_nms_iou"),
            DEFAULT_NMS_IOU,
            min_value=0.0,
            max_value=1.0,
        ),
        "min_box_size": _coerce_float(
            options.get("jiandu_min_box_size"),
            DEFAULT_MIN_BOX_SIZE,
            min_value=1.0,
            max_value=4096.0,
        ),
        "max_aspect": _coerce_float(
            options.get("jiandu_max_aspect"),
            DEFAULT_MAX_ASPECT,
            min_value=1.0,
            max_value=6.0,
        ),
        "rec_top_k": _coerce_int(
            options.get("jiandu_rec_top_k"),
            DEFAULT_REC_TOP_K,
            min_value=1,
            max_value=20,
        ),
    }


def _resolve_source_dir() -> Path:
    configured = os.getenv("DEEPJIANDU_SOURCE_DIR", "").strip()
    return Path(configured) if configured else DEFAULT_SOURCE_DIR


def _resolve_existing_path(env_name: str, candidates: list[Path], label: str) -> Path:
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        resolved = Path(explicit)
        if resolved.exists():
            return resolved
        raise RuntimeError(f"{label} not found: {resolved}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(str(path) for path in candidates)
    raise RuntimeError(f"{label} not found. Tried:\n{tried}")


def resolve_detector_checkpoint() -> Path:
    source_dir = _resolve_source_dir()
    candidates = [
        source_dir / "checkpoints" / "detector_best.pt",
        source_dir / "new_results" / "runs" / "deepjiandu_char_detector" / "best.pt",
        source_dir / "runs" / "deepjiandu_char_detector" / "best.pt",
        source_dir / "server_results" / "runs" / "deepjiandu_char_detector" / "best.pt",
    ]
    return _resolve_existing_path(
        "DEEPJIANDU_DETECTOR_CHECKPOINT",
        candidates,
        "DeepJiandu detector checkpoint",
    )


def resolve_recognizer_checkpoint() -> Path:
    source_dir = _resolve_source_dir()
    candidates = [
        source_dir / "checkpoints" / "recognizer_best.pt",
        source_dir / "new_results" / "runs" / "deepjiandu_recognizer" / "best.pt",
        source_dir / "server_results" / "runs" / "deepjiandu_recognizer" / "best.pt",
        source_dir / "runs" / "deepjiandu_recognizer" / "best.pt",
    ]
    return _resolve_existing_path(
        "DEEPJIANDU_RECOGNIZER_CHECKPOINT",
        candidates,
        "DeepJiandu recognizer checkpoint",
    )


def resolve_device_arg() -> str:
    return os.getenv("DEEPJIANDU_DEVICE", "auto").strip() or "auto"


@lru_cache(maxsize=1)
def _import_fullpage_ocr_module() -> Any:
    source_dir = _resolve_source_dir()
    if not source_dir.exists():
        raise RuntimeError(f"DeepJiandu source directory not found: {source_dir}")

    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    try:
        return importlib.import_module("deepjiandu_ocr.fullpage_ocr")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Failed to import deepjiandu_ocr.fullpage_ocr from {source_dir}"
        ) from exc


@lru_cache(maxsize=8)
def _load_runtime(
    detector_checkpoint_text: str,
    recognizer_checkpoint_text: str,
    device_arg: str,
) -> dict[str, Any]:
    fullpage_ocr = _import_fullpage_ocr_module()
    detector_path = Path(detector_checkpoint_text)
    recognizer_path = Path(recognizer_checkpoint_text)
    device = fullpage_ocr.resolve_device(device_arg)
    detector, det_w, det_h = fullpage_ocr.load_detector_checkpoint(detector_path, device=device)
    recognizer, idx_to_class, rec_size = fullpage_ocr.load_recognizer_checkpoint(
        recognizer_path,
        device=device,
    )
    return {
        "module": fullpage_ocr,
        "device": device,
        "device_name": str(device),
        "detector": detector,
        "recognizer": recognizer,
        "idx_to_class": idx_to_class,
        "det_input_w": det_w,
        "det_input_h": det_h,
        "rec_img_size": rec_size,
        "detector_checkpoint": detector_path,
        "recognizer_checkpoint": recognizer_path,
    }


def run_deepjiandu_page(
    image: Image.Image,
    raw_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Image.Image]:
    options = normalize_deepjiandu_options(raw_options)
    detector_checkpoint = resolve_detector_checkpoint()
    recognizer_checkpoint = resolve_recognizer_checkpoint()
    runtime = _load_runtime(
        str(detector_checkpoint.resolve()),
        str(recognizer_checkpoint.resolve()),
        resolve_device_arg(),
    )
    fullpage_ocr = runtime["module"]
    result = fullpage_ocr.run_fullpage_ocr(
        detector=runtime["detector"],
        recognizer=runtime["recognizer"],
        idx_to_class=runtime["idx_to_class"],
        image=image,
        device=runtime["device"],
        det_input_w=runtime["det_input_w"],
        det_input_h=runtime["det_input_h"],
        det_score_thresh=options["score_thresh"],
        det_topk=options["topk_detect"],
        det_nms_iou=options["nms_iou"],
        det_min_box_size=options["min_box_size"],
        max_aspect=options["max_aspect"],
        rec_img_size=runtime["rec_img_size"],
        rec_top_k=options["rec_top_k"],
    )
    result["inference_config"] = {
        "device": runtime["device_name"],
        "detector_checkpoint": str(runtime["detector_checkpoint"]),
        "recognizer_checkpoint": str(runtime["recognizer_checkpoint"]),
        **options,
    }
    overlay = fullpage_ocr.draw_overlay(image, result)
    return result, overlay
