from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import torch

from deepjiandu_ocr.checkpoint_utils import torch_load_compatible
from deepjiandu_ocr.fullpage_detector import (
    CharCenterDetector,
    decode_detector_outputs,
    image_to_tensor,
    letterbox_grayscale_with_meta,
    map_boxes_from_letterbox,
)
from deepjiandu_ocr.image_utils import image_to_tensor as rec_image_to_tensor
from deepjiandu_ocr.image_utils import letterbox_grayscale
from deepjiandu_ocr.model import JianduRecognizer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CJK_FONT = _PROJECT_ROOT / "datafile" / "simsunb.ttf"
_FONT_PROBE_TEXT = "到還問隆少反毋塞狀"


def _font_candidates() -> list[Path]:
    return [
        _DEFAULT_CJK_FONT,
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
    ]


def _font_looks_like_cjk(font: ImageFont.ImageFont) -> bool:
    signatures: set[tuple[tuple[int, int], bytes]] = set()
    for ch in _FONT_PROBE_TEXT:
        mask = font.getmask(ch, mode="L")
        payload = bytes(mask)
        if not payload or max(payload) == 0:
            return False
        signatures.add((mask.size, payload))
    return len(signatures) >= 3


@lru_cache(maxsize=16)
def load_overlay_font(font_size: int) -> ImageFont.ImageFont:
    size = max(10, int(font_size))
    for path in _font_candidates():
        if path.exists():
            try:
                font = ImageFont.truetype(str(path), size=size)
                if _font_looks_like_cjk(font):
                    return font
            except OSError:
                continue
    return ImageFont.load_default()


def _median_int(values: list[int], default: int) -> int:
    if not values:
        return default
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return int(ordered[mid])
    return int(round((ordered[mid - 1] + ordered[mid]) / 2.0))


def sort_chars_rtl_then_ttb(chars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chars:
        return []

    enriched: list[dict[str, Any]] = []
    widths: list[int] = []
    for item in chars:
        box = item["box_xyxy"]
        x1 = int(box["x1"])
        x2 = int(box["x2"])
        y1 = int(box["y1"])
        y2 = int(box["y2"])
        w = max(1, x2 - x1)
        widths.append(w)
        enriched.append(
            {
                "item": item,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )

    # Use character width as an adaptive horizontal tolerance for column grouping.
    med_w = _median_int(widths, default=18)
    col_tol = max(8.0, float(med_w) * 0.75)

    # Build vertical columns from right to left, then sort each column top to bottom.
    columns: list[dict[str, Any]] = []
    for node in sorted(enriched, key=lambda n: n["cx"], reverse=True):
        placed = False
        for col in columns:
            if abs(node["cx"] - col["mean_cx"]) <= col_tol:
                col["nodes"].append(node)
                count = len(col["nodes"])
                col["mean_cx"] = ((col["mean_cx"] * (count - 1)) + node["cx"]) / float(count)
                placed = True
                break
        if not placed:
            columns.append({"mean_cx": node["cx"], "nodes": [node]})

    columns.sort(key=lambda c: c["mean_cx"], reverse=True)
    ordered: list[dict[str, Any]] = []
    for col in columns:
        for node in sorted(col["nodes"], key=lambda n: n["cy"]):
            ordered.append(node["item"])

    for idx, item in enumerate(ordered, start=1):
        item["index"] = idx
    return ordered


def _box_tuple_from_char_item(item: dict[str, Any]) -> tuple[int, int, int, int]:
    box = item["box_xyxy"]
    return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])


def _merge_fragmented_chars(
    chars: list[dict[str, Any]],
    *,
    image: Image.Image,
    recognizer: JianduRecognizer,
    idx_to_class: dict[int, str],
    rec_img_size: int,
    device: torch.device,
    rec_top_k: int,
    max_aspect: float,
) -> list[dict[str, Any]]:
    if len(chars) < 2:
        return chars

    boxes = [_box_tuple_from_char_item(c) for c in chars]
    widths = [max(1, x2 - x1) for x1, _, x2, _ in boxes]
    heights = [max(1, y2 - y1) for _, y1, _, y2 in boxes]
    med_w = float(_median_int(widths, default=18))
    med_h = float(_median_int(heights, default=18))

    n = len(chars)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    def is_fragment_candidate(i: int) -> bool:
        c = chars[i]
        det_score = float(c.get("det_score", 0.0))
        top1_prob = float(c.get("top1", {}).get("prob", 0.0))
        h = float(heights[i])
        return (det_score < 0.62) or (top1_prob < 0.12) or (h < med_h * 0.72)

    for i in range(n):
        for j in range(i + 1, n):
            if not (is_fragment_candidate(i) or is_fragment_candidate(j)):
                continue

            x1a, y1a, x2a, y2a = boxes[i]
            x1b, y1b, x2b, y2b = boxes[j]
            wa, ha = max(1, x2a - x1a), max(1, y2a - y1a)
            wb, hb = max(1, x2b - x1b), max(1, y2b - y1b)
            cxa = (x1a + x2a) / 2.0
            cxb = (x1b + x2b) / 2.0
            x_center_dist = abs(cxa - cxb)
            x_tol = max(10.0, med_w * 0.42)
            if x_center_dist > x_tol:
                continue

            width_sim = float(min(wa, wb)) / float(max(wa, wb))
            if width_sim < 0.50:
                continue

            y_overlap = min(y2a, y2b) - max(y1a, y1b)
            y_gap = max(0, max(y1a, y1b) - min(y2a, y2b))
            min_h = float(min(ha, hb))
            close_vertically = (y_overlap >= max(3.0, min_h * 0.16)) or (y_gap <= max(2.0, min_h * 0.12))
            if not close_vertically:
                continue

            ux1, uy1 = min(x1a, x1b), min(y1a, y1b)
            ux2, uy2 = max(x2a, x2b), max(y2a, y2b)
            uw, uh = max(1, ux2 - ux1), max(1, uy2 - uy1)
            if uh > med_h * 1.80:
                continue
            uaspect = max(float(uw), float(uh)) / float(min(float(uw), float(uh)))
            if uaspect > max(2.35, float(max_aspect)):
                continue

            union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    replaced: set[int] = set()
    merged_items: list[dict[str, Any]] = []

    for member_idxs in clusters.values():
        if len(member_idxs) < 2 or len(member_idxs) > 4:
            continue

        member_idxs = sorted(member_idxs, key=lambda k: boxes[k][1])
        cluster_boxes = [boxes[k] for k in member_idxs]
        ux1 = min(b[0] for b in cluster_boxes)
        uy1 = min(b[1] for b in cluster_boxes)
        ux2 = max(b[2] for b in cluster_boxes)
        uy2 = max(b[3] for b in cluster_boxes)
        uw, uh = max(1, ux2 - ux1), max(1, uy2 - uy1)
        if uh > med_h * 1.80:
            continue
        aspect = max(float(uw), float(uh)) / float(min(float(uw), float(uh)))
        if aspect > max(2.35, float(max_aspect)):
            continue

        # Avoid merging two true adjacent characters with a visible gap.
        gaps = []
        for a, b in zip(cluster_boxes[:-1], cluster_boxes[1:]):
            gaps.append(max(0, b[1] - a[3]))
        max_gap = max(gaps) if gaps else 0
        if max_gap > max(3.0, med_h * 0.12):
            continue

        member_probs = [float(chars[k].get("top1", {}).get("prob", 0.0)) for k in member_idxs]
        member_dets = [float(chars[k].get("det_score", 0.0)) for k in member_idxs]
        max_member_prob = max(member_probs) if member_probs else 0.0
        cluster_det = max(member_dets) if member_dets else 0.0

        # Be conservative for 2-box clusters; these can be two adjacent valid chars.
        if len(member_idxs) == 2:
            if max_member_prob >= 0.11:
                continue
            if max(member_dets) >= 0.62:
                continue

        crop = image.crop((ux1, uy1, ux2, uy2))
        rec = recognize_crop(
            recognizer=recognizer,
            idx_to_class=idx_to_class,
            crop=crop,
            img_size=rec_img_size,
            device=device,
            top_k=rec_top_k,
        )
        top1 = rec[0] if rec else {"rank": 1, "class_idx": -1, "char": "", "prob": 0.0}
        merged_prob = float(top1.get("prob", 0.0))
        merged_char = str(top1.get("char", ""))

        # Only replace when merged recognition is clearly better.
        if merged_prob < (max_member_prob + 0.02):
            continue
        if merged_char in ("", "□") and merged_prob < (max_member_prob + 0.06):
            continue

        merged_items.append(
            {
                "box_xyxy": {"x1": int(ux1), "y1": int(uy1), "x2": int(ux2), "y2": int(uy2)},
                "box_xywh": {"x": int(ux1), "y": int(uy1), "w": int(uw), "h": int(uh)},
                "det_score": cluster_det,
                "top1": top1,
                "topk": rec,
                "merged_from": [int(k + 1) for k in member_idxs],
            }
        )
        replaced.update(member_idxs)

    if not replaced:
        return chars

    out = [c for i, c in enumerate(chars) if i not in replaced]
    out.extend(merged_items)
    return out


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(max(1, area_a + area_b - inter))


def _resolve_box_overlaps(
    chars: list[dict[str, Any]],
    *,
    image: Image.Image,
) -> list[dict[str, Any]]:
    if len(chars) < 2:
        return chars

    img_w, img_h = image.size
    boxes = [list(_box_tuple_from_char_item(c)) for c in chars]
    widths = [max(1, b[2] - b[0]) for b in boxes]
    heights = [max(1, b[3] - b[1]) for b in boxes]
    med_w = float(_median_int(widths, default=18))
    med_h = float(_median_int(heights, default=18))
    col_tol = max(10.0, med_w * 0.58)
    min_w = int(max(10.0, med_w * 0.36))
    min_h = int(max(10.0, med_h * 0.36))

    changed = [False] * len(chars)

    for _ in range(3):
        any_change = False
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                x1a, y1a, x2a, y2a = boxes[i]
                x1b, y1b, x2b, y2b = boxes[j]
                ovx = min(x2a, x2b) - max(x1a, x1b)
                ovy = min(y2a, y2b) - max(y1a, y1b)
                if ovx <= 0 or ovy <= 0:
                    continue

                iou = _iou_xyxy((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b))
                cxa = (x1a + x2a) / 2.0
                cya = (y1a + y2a) / 2.0
                cxb = (x1b + x2b) / 2.0
                cyb = (y1b + y2b) / 2.0
                same_col = abs(cxa - cxb) <= col_tol

                if same_col:
                    # Vertical ordering: upper and lower boxes should not overlap.
                    if ovy < max(2, int(min_h * 0.12)) and iou < 0.20:
                        continue
                    upper_idx, lower_idx = (i, j) if cya <= cyb else (j, i)
                    up = boxes[upper_idx]
                    low = boxes[lower_idx]
                    if low[1] >= up[3]:
                        continue
                    split = int(round((up[3] + low[1]) / 2.0))
                    new_up_y2 = max(up[1] + min_h, split)
                    new_low_y1 = min(low[3] - min_h, split + 1)
                    did = False
                    if new_up_y2 < up[3]:
                        up[3] = new_up_y2
                        changed[upper_idx] = True
                        did = True
                    if new_low_y1 > low[1]:
                        low[1] = new_low_y1
                        changed[lower_idx] = True
                        did = True
                    if did:
                        any_change = True
                    continue

                # Cross-column overlap: shrink horizontal conflict on both sides.
                if iou < 0.24:
                    continue
                right_idx, left_idx = (i, j) if cxa >= cxb else (j, i)
                right = boxes[right_idx]
                left = boxes[left_idx]
                boundary = int(round((cxa + cxb) / 2.0))
                did = False

                new_right_x1 = max(right[0], boundary)
                if (right[2] - new_right_x1) >= min_w and new_right_x1 > right[0]:
                    right[0] = new_right_x1
                    changed[right_idx] = True
                    did = True

                new_left_x2 = min(left[2], boundary)
                if (new_left_x2 - left[0]) >= min_w and new_left_x2 < left[2]:
                    left[2] = new_left_x2
                    changed[left_idx] = True
                    did = True

                if did:
                    any_change = True

        if not any_change:
            break

    if not any(changed):
        return chars

    out: list[dict[str, Any]] = []
    for idx, c in enumerate(chars):
        x1, y1, x2, y2 = boxes[idx]
        x1 = int(max(0, min(img_w - 1, x1)))
        y1 = int(max(0, min(img_h - 1, y1)))
        x2 = int(max(x1 + 1, min(img_w, x2)))
        y2 = int(max(y1 + 1, min(img_h, y2)))

        c["box_xyxy"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        c["box_xywh"] = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}

        out.append(c)
    return out


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_detector_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[CharCenterDetector, int, int]:
    ckpt = torch_load_compatible(checkpoint_path, map_location=device, weights_only=False)
    model = CharCenterDetector()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    train_args = ckpt.get("train_args", {})
    input_w = int(train_args.get("input_w", 256))
    input_h = int(train_args.get("input_h", 640))
    if input_w <= 0 or input_h <= 0:
        input_w, input_h = 256, 640
    return model, input_w, input_h


def load_recognizer_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[JianduRecognizer, dict[int, str], int]:
    ckpt = torch_load_compatible(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = ckpt.get("class_to_idx")
    if not class_to_idx:
        raise RuntimeError("Recognizer checkpoint missing class_to_idx.")
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    model = JianduRecognizer(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    train_args = ckpt.get("train_args", {})
    img_size = int(train_args.get("img_size", 96))
    if img_size <= 0:
        img_size = 96
    return model, idx_to_class, img_size


@torch.no_grad()
def recognize_crop(
    recognizer: JianduRecognizer,
    idx_to_class: dict[int, str],
    crop: Image.Image,
    img_size: int,
    device: torch.device,
    top_k: int,
) -> list[dict]:
    crop_gray = letterbox_grayscale(crop.convert("L"), img_size)
    tensor = rec_image_to_tensor(crop_gray).unsqueeze(0).to(device)
    logits = recognizer(tensor)
    probs = torch.softmax(logits, dim=1)
    k = min(int(top_k), probs.shape[1])
    scores, indices = torch.topk(probs, k=k, dim=1)
    out = []
    for rank, (score, idx) in enumerate(zip(scores.squeeze(0).tolist(), indices.squeeze(0).tolist()), start=1):
        out.append(
            {
                "rank": rank,
                "class_idx": int(idx),
                "char": idx_to_class.get(int(idx), ""),
                "prob": float(score),
            }
        )
    return out


@torch.no_grad()
def run_fullpage_ocr(
    detector: CharCenterDetector,
    recognizer: JianduRecognizer,
    idx_to_class: dict[int, str],
    image: Image.Image,
    device: torch.device,
    *,
    det_input_w: int,
    det_input_h: int,
    det_score_thresh: float = 0.30,
    det_topk: int = 300,
    det_nms_iou: float = 0.35,
    det_min_box_size: float = 8.0,
    max_aspect: float = 1.8,
    rec_img_size: int = 96,
    rec_top_k: int = 5,
) -> dict:
    gray = image.convert("L")
    det_img, det_meta = letterbox_grayscale_with_meta(gray, dst_w=det_input_w, dst_h=det_input_h)
    det_tensor = image_to_tensor(det_img).unsqueeze(0).to(device)
    heat_logits, size_norm = detector(det_tensor)

    pred_boxes, pred_scores = decode_detector_outputs(
        heat_logits,
        size_norm,
        score_thresh=det_score_thresh,
        topk=det_topk,
        min_box_size=det_min_box_size,
        nms_iou=det_nms_iou,
    )
    abs_boxes = map_boxes_from_letterbox(pred_boxes, det_meta)

    merged = []
    for (x1, y1, x2, y2), score in zip(abs_boxes, pred_scores):
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect = max(float(w), float(h)) / float(min(w, h))
        if aspect > float(max_aspect):
            continue
        merged.append((x1, y1, x2, y2, float(score)))
    chars = []
    for x1, y1, x2, y2, det_score in merged:
        crop = image.crop((x1, y1, x2, y2))
        rec = recognize_crop(
            recognizer=recognizer,
            idx_to_class=idx_to_class,
            crop=crop,
            img_size=rec_img_size,
            device=device,
            top_k=rec_top_k,
        )
        top1 = rec[0] if rec else {"rank": 1, "class_idx": -1, "char": "", "prob": 0.0}
        chars.append(
            {
                "box_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "box_xywh": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                "det_score": det_score,
                "top1": top1,
                "topk": rec,
            }
        )

    chars = _merge_fragmented_chars(
        chars,
        image=image,
        recognizer=recognizer,
        idx_to_class=idx_to_class,
        rec_img_size=rec_img_size,
        device=device,
        rec_top_k=rec_top_k,
        max_aspect=max_aspect,
    )
    chars = _resolve_box_overlaps(
        chars,
        image=image,
    )
    chars = sort_chars_rtl_then_ttb(chars)
    return {
        "num_chars": len(chars),
        "text": "".join(c["top1"]["char"] for c in chars),
        "chars": chars,
    }


def draw_overlay(image: Image.Image, result: dict) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for c in result.get("chars", []):
        box = c["box_xyxy"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 64, 64), width=2)
        txt = f"{c['index']}:{c['top1']['char']}"
        box_h = max(1, int(y2) - int(y1))
        font_size = int(max(12, min(28, round(box_h * 0.28))))
        font = load_overlay_font(font_size)
        tx = int(x1) + 2
        ty = max(0, int(y1) - font_size - 2)
        try:
            tb = draw.textbbox((tx, ty), txt, font=font)
            draw.rectangle((tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1), fill=(255, 244, 244))
        except Exception:
            pass
        draw.text((tx, ty), txt, fill=(220, 36, 36), font=font)
    return overlay
