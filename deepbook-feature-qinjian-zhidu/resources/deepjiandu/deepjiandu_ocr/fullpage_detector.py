from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import random
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class CharBox:
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class PageEntry:
    split: str
    image_member: str
    image_file: str
    width: int
    height: int
    boxes: list[CharBox]


@dataclass
class LetterboxMeta:
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    pad_x: int
    pad_y: int
    resized_w: int
    resized_h: int


def as_int(text: str | None, fallback: int = 0) -> int:
    if text is None:
        return fallback
    text = text.strip()
    if not text:
        return fallback
    try:
        return int(round(float(text)))
    except ValueError:
        return fallback


def infer_split(member_name: str) -> str:
    parts = member_name.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2].lower()
    return "train"


def parse_xml_annotation(xml_bytes: bytes, fallback_image_name: str) -> tuple[str, int, int, list[CharBox]]:
    root = ET.fromstring(xml_bytes)
    filename = root.findtext("filename")
    image_file = (filename.strip() if filename else fallback_image_name).strip()

    size = root.find("size")
    width = as_int(size.findtext("width") if size is not None else None, fallback=0)
    height = as_int(size.findtext("height") if size is not None else None, fallback=0)

    boxes: list[CharBox] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        bndbox = obj.find("bndbox")
        if not class_name or bndbox is None:
            continue

        xmin = as_int(bndbox.findtext("xmin"))
        ymin = as_int(bndbox.findtext("ymin"))
        xmax = as_int(bndbox.findtext("xmax"))
        ymax = as_int(bndbox.findtext("ymax"))
        boxes.append(CharBox(class_name=class_name, x1=float(xmin), y1=float(ymin), x2=float(xmax), y2=float(ymax)))

    return image_file, width, height, boxes


def build_image_member_index(image_zip: zipfile.ZipFile) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for member in image_zip.namelist():
        low = member.lower()
        if not (low.endswith(".bmp") or low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg")):
            continue
        norm = member.replace("\\", "/")
        parts = norm.split("/")
        if len(parts) < 2:
            continue
        split = parts[-2].lower()
        filename = parts[-1]
        index[(split, filename)] = member
    return index


def load_page_entries(images_zip_path: Path, labels_zip_path: Path, *, limit_xml: int = 0) -> list[PageEntry]:
    if not images_zip_path.exists():
        raise FileNotFoundError(f"Missing images zip: {images_zip_path}")
    if not labels_zip_path.exists():
        raise FileNotFoundError(f"Missing labels zip: {labels_zip_path}")

    with zipfile.ZipFile(images_zip_path, "r") as image_zip, zipfile.ZipFile(labels_zip_path, "r") as label_zip:
        image_index = build_image_member_index(image_zip)
        xml_members = [m for m in label_zip.namelist() if m.lower().endswith(".xml")]
        xml_members.sort()
        if limit_xml > 0 and len(xml_members) > limit_xml:
            rng = random.Random(42)
            rng.shuffle(xml_members)
            xml_members = xml_members[:limit_xml]

        entries: list[PageEntry] = []
        for member in xml_members:
            split = infer_split(member)
            fallback_image_name = f"{Path(member).stem}.bmp"
            image_file, width, height, boxes = parse_xml_annotation(
                label_zip.read(member), fallback_image_name=fallback_image_name
            )
            if not boxes:
                continue

            image_member = image_index.get((split, image_file))
            if image_member is None:
                for alt_split in ("train", "val", "test"):
                    image_member = image_index.get((alt_split, image_file))
                    if image_member is not None:
                        split = alt_split
                        break
            if image_member is None:
                continue

            entries.append(
                PageEntry(
                    split=split,
                    image_member=image_member,
                    image_file=image_file,
                    width=width,
                    height=height,
                    boxes=boxes,
                )
            )
    return entries


def letterbox_grayscale_with_meta(image: Image.Image, dst_w: int, dst_h: int, fill: int = 0) -> tuple[Image.Image, LetterboxMeta]:
    if image.mode != "L":
        image = image.convert("L")
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid image size.")

    scale = min(float(dst_w) / float(src_w), float(dst_h) / float(src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)

    pad_x = (dst_w - resized_w) // 2
    pad_y = (dst_h - resized_h) // 2

    canvas = Image.new("L", (dst_w, dst_h), color=fill)
    canvas.paste(resized, (pad_x, pad_y))
    meta = LetterboxMeta(
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        resized_w=resized_w,
        resized_h=resized_h,
    )
    return canvas, meta


def map_boxes_to_letterbox(boxes: list[CharBox], meta: LetterboxMeta) -> list[tuple[float, float, float, float]]:
    mapped: list[tuple[float, float, float, float]] = []
    src_w = max(1, meta.src_w)
    src_h = max(1, meta.src_h)
    for box in boxes:
        x1 = max(0.0, min(float(src_w - 1), float(box.x1)))
        y1 = max(0.0, min(float(src_h - 1), float(box.y1)))
        x2 = max(x1 + 1.0, min(float(src_w), float(box.x2 + 1.0)))
        y2 = max(y1 + 1.0, min(float(src_h), float(box.y2 + 1.0)))

        nx1 = x1 * meta.scale + float(meta.pad_x)
        ny1 = y1 * meta.scale + float(meta.pad_y)
        nx2 = x2 * meta.scale + float(meta.pad_x)
        ny2 = y2 * meta.scale + float(meta.pad_y)
        mapped.append((nx1, ny1, nx2, ny2))
    return mapped


def map_boxes_from_letterbox(
    boxes_xyxy: list[tuple[float, float, float, float]],
    meta: LetterboxMeta,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        ox1 = (x1 - float(meta.pad_x)) / max(meta.scale, 1e-6)
        oy1 = (y1 - float(meta.pad_y)) / max(meta.scale, 1e-6)
        ox2 = (x2 - float(meta.pad_x)) / max(meta.scale, 1e-6)
        oy2 = (y2 - float(meta.pad_y)) / max(meta.scale, 1e-6)

        ox1 = max(0.0, min(float(meta.src_w - 1), ox1))
        oy1 = max(0.0, min(float(meta.src_h - 1), oy1))
        ox2 = max(ox1 + 1.0, min(float(meta.src_w), ox2))
        oy2 = max(oy1 + 1.0, min(float(meta.src_h), oy2))
        out.append((int(round(ox1)), int(round(oy1)), int(round(ox2)), int(round(oy2))))
    return out


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0)


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int, k: float = 1.0) -> None:
    h, w = heatmap.shape
    radius = max(1, int(radius))
    diameter = radius * 2 + 1
    sigma = max(1.0, diameter / 6.0)

    ys = np.arange(0, diameter, dtype=np.float32) - float(radius)
    xs = np.arange(0, diameter, dtype=np.float32) - float(radius)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    gaussian = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32) * float(k)

    left = min(cx, radius)
    right = min(w - cx - 1, radius)
    top = min(cy, radius)
    bottom = min(h - cy - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return

    y0 = cy - top
    y1 = cy + bottom + 1
    x0 = cx - left
    x1 = cx + right + 1

    gy0 = radius - top
    gy1 = radius + bottom + 1
    gx0 = radius - left
    gx1 = radius + right + 1

    patch = heatmap[y0:y1, x0:x1]
    np.maximum(patch, gaussian[gy0:gy1, gx0:gx1], out=patch)


def build_detector_targets(
    boxes_xyxy: list[tuple[float, float, float, float]],
    out_w: int,
    out_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heat = np.zeros((out_h, out_w), dtype=np.float32)
    size = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    for x1, y1, x2, y2 in boxes_xyxy:
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round((y1 + y2) * 0.5))
        cx = max(0, min(out_w - 1, cx))
        cy = max(0, min(out_h - 1, cy))

        radius = int(max(1.0, min(bw, bh) * 0.18))
        draw_gaussian(heat, cx, cy, radius=radius, k=1.0)

        size[0, cy, cx] = float(bw) / float(out_w)
        size[1, cy, cx] = float(bh) / float(out_h)
        mask[cy, cx] = 1.0

    return heat, size, mask


def conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CharCenterDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu(1, 32), conv_bn_relu(32, 32))
        self.down1 = nn.Sequential(conv_bn_relu(32, 64, stride=2), conv_bn_relu(64, 64))
        self.down2 = nn.Sequential(conv_bn_relu(64, 128, stride=2), conv_bn_relu(128, 128))
        self.down3 = nn.Sequential(conv_bn_relu(128, 256, stride=2), conv_bn_relu(256, 256))

        self.up2 = nn.Sequential(conv_bn_relu(256 + 128, 128), conv_bn_relu(128, 128))
        self.up1 = nn.Sequential(conv_bn_relu(128 + 64, 64), conv_bn_relu(64, 64))
        self.up0 = nn.Sequential(conv_bn_relu(64 + 32, 32), conv_bn_relu(32, 32))

        self.heat_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.stem(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)

        u2 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, f2], dim=1))

        u1 = F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, f1], dim=1))

        u0 = F.interpolate(u1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, f0], dim=1))

        heat_logits = self.heat_head(u0)
        size_norm = torch.sigmoid(self.size_head(u0))
        return heat_logits, size_norm


def heatmap_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    pos_mask = (target >= 0.99).float()
    neg_mask = (target < 0.99).float()
    neg_weight = (1.0 - target) ** 4

    pos_loss = -torch.log(pred) * ((1.0 - pred) ** 2) * pos_mask
    neg_loss = -torch.log(1.0 - pred) * (pred**2) * neg_weight * neg_mask

    pos_count = pos_mask.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / pos_count


def size_regression_loss(pred_size: torch.Tensor, target_size: torch.Tensor, center_mask: torch.Tensor) -> torch.Tensor:
    mask = center_mask.unsqueeze(1)
    diff = (pred_size - target_size).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def compute_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = max(1e-6, area_a + area_b - inter)
    return inter / union


def nms_xyxy(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_thresh: float = 0.35,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    if not boxes:
        return [], []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        suppressed = False
        for k in keep:
            if compute_iou_xyxy(boxes[i], boxes[k]) >= iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def decode_detector_outputs(
    heat_logits: torch.Tensor,
    size_norm: torch.Tensor,
    *,
    score_thresh: float = 0.30,
    topk: int = 300,
    min_box_size: float = 6.0,
    max_box_size_ratio: float = 0.70,
    nms_iou: float = 0.35,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    if heat_logits.ndim != 4 or size_norm.ndim != 4:
        raise ValueError("Expected batched tensors with shape [B,C,H,W].")
    if heat_logits.shape[0] != 1:
        raise ValueError("decode_detector_outputs currently supports batch size 1.")

    heat = torch.sigmoid(heat_logits[0, 0]).detach()
    size = size_norm[0].detach()
    h, w = heat.shape

    pooled = F.max_pool2d(heat.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
    keep_mask = (heat >= pooled) & (heat >= float(score_thresh))
    ys, xs = torch.where(keep_mask)
    if ys.numel() == 0:
        return [], []

    scores = heat[ys, xs]
    if scores.numel() > topk:
        top_scores, idx = torch.topk(scores, k=topk)
        ys = ys[idx]
        xs = xs[idx]
        scores = top_scores

    boxes: list[tuple[float, float, float, float]] = []
    score_list: list[float] = []
    max_bw = float(w) * float(max_box_size_ratio)
    max_bh = float(h) * float(max_box_size_ratio)
    for y, x, s in zip(ys.tolist(), xs.tolist(), scores.tolist()):
        bw = float(size[0, y, x].item()) * float(w)
        bh = float(size[1, y, x].item()) * float(h)
        bw = float(np.clip(bw, float(min_box_size), max_bw))
        bh = float(np.clip(bh, float(min_box_size), max_bh))

        cx = float(x) + 0.5
        cy = float(y) + 0.5
        x1 = max(0.0, cx - bw * 0.5)
        y1 = max(0.0, cy - bh * 0.5)
        x2 = min(float(w), cx + bw * 0.5)
        y2 = min(float(h), cy + bh * 0.5)
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
            continue
        boxes.append((x1, y1, x2, y2))
        score_list.append(float(s))

    return nms_xyxy(boxes, score_list, iou_thresh=nms_iou)


def match_detections(
    pred_boxes: list[tuple[float, float, float, float]],
    pred_scores: list[float],
    gt_boxes: list[tuple[float, float, float, float]],
    iou_thresh: float = 0.3,
) -> tuple[int, int, int]:
    if not pred_boxes:
        return 0, 0, len(gt_boxes)
    if not gt_boxes:
        return 0, len(pred_boxes), 0

    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    used = [False for _ in gt_boxes]
    tp = 0
    fp = 0
    for i in order:
        best_j = -1
        best_iou = 0.0
        for j, gt in enumerate(gt_boxes):
            if used[j]:
                continue
            iou = compute_iou_xyxy(pred_boxes[i], gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = used.count(False)
    return tp, fp, fn

