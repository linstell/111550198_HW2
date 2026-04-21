from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
)


def infer_num_queries_and_labels(
    checkpoint: Dict[str, object],
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[int, int]:
    meta = checkpoint.get("meta", {})
    if "num_queries" in meta and "num_labels" in meta:
        return int(meta["num_queries"]), int(meta["num_labels"])

    # fallback for older checkpoints
    query_candidates = [
        "model.query_position_embeddings.weight",
        "query_position_embeddings.weight",
        "model.query_position_embeddings.layers.0.weight",
    ]
    cls_candidates = [
        "class_labels_classifier.weight",
        "model.class_labels_classifier.weight",
    ]

    num_queries = None
    num_labels = None

    for key in query_candidates:
        if key in state_dict:
            num_queries = int(state_dict[key].shape[0])
            break

    for key in cls_candidates:
        if key in state_dict:
            num_labels_plus_bg = int(state_dict[key].shape[0])
            num_labels = num_labels_plus_bg - 1
            break

    if num_queries is None or num_labels is None:
        raise KeyError("Could not infer num_queries/num_labels. Save meta in checkpoint.")

    return num_queries, num_labels


def summarize_export(all_preds: List[Dict], num_images: int) -> None:
    per_class = Counter(p["category_id"] for p in all_preds)
    per_img = Counter(p["image_id"] for p in all_preds)

    print("Per-class counts:", dict(sorted(per_class.items())))
    print("Images with preds:", len(per_img))
    print("No-pred images:", num_images - len(per_img))
    print("Images >5 preds:", sum(1 for v in per_img.values() if v > 5))
    print("Images >8 preds:", sum(1 for v in per_img.values() if v > 8))
    print("Images >12 preds:", sum(1 for v in per_img.values() if v > 12))


@torch.no_grad()
def run_model(
    model: DeformableDetrForObjectDetection,
    processor: DeformableDetrImageProcessor,
    device: torch.device,
    image: Image.Image,
    orig_w: int,
    orig_h: int,
    threshold: float,
    max_boxes: int,
    min_w: float,
    min_h: float,
    use_amp: bool,
) -> List[Dict]:
    inputs = processor(
        images=image,
        return_tensors="pt",
        size={"shortest_edge": max(orig_h, orig_w), "longest_edge": max(orig_h, orig_w)},
    )
    pixel_values = inputs["pixel_values"].to(device)
    pixel_mask = inputs.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)

    if use_amp:
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    else:
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    target_sizes = torch.tensor([[orig_h, orig_w]], device=device)

    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    scores = results["scores"].detach().cpu().tolist()
    labels = results["labels"].detach().cpu().tolist()
    boxes = results["boxes"].detach().cpu().tolist()

    preds: List[Dict] = []
    for score, label, box in zip(scores, labels, boxes):
        x1, y1, x2, y2 = box

        x1 = max(0.0, min(float(x1), orig_w))
        y1 = max(0.0, min(float(y1), orig_h))
        x2 = max(0.0, min(float(x2), orig_w))
        y2 = max(0.0, min(float(y2), orig_h))

        if x2 <= x1 or y2 <= y1:
            continue

        w = x2 - x1
        h = y2 - y1
        if w < min_w or h < min_h:
            continue

        preds.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": float(score),
                "category_id": int(label) + 1,
            }
        )

    preds = sorted(preds, key=lambda x: x["score"], reverse=True)[:max_boxes]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", type=str, default="test")
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    ap.add_argument("--out", type=str, default="pred.json")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.03)
    ap.add_argument("--max_boxes", type=int, default=30)
    ap.add_argument("--min_w", type=float, default=1.0)
    ap.add_argument("--min_h", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    num_queries, num_labels = infer_num_queries_and_labels(ckpt, state)
    print(f"Inferred from checkpoint: num_queries={num_queries}, num_labels={num_labels}")
    print(
        "Checkpoint epoch:",
        ckpt.get("epoch", "unknown"),
        "val_map:",
        ckpt.get("val_map", "unknown"),
    )

    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {str(i): i for i in range(num_labels)}

    processor = DeformableDetrImageProcessor(
        format="coco_detection",
        do_resize=True,
    )

    config = DeformableDetrConfig()
    config.use_pretrained_backbone = False
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id
    config.auxiliary_loss = True
    config.num_queries = num_queries

    model = DeformableDetrForObjectDetection(config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    test_dir = Path(args.test_dir)
    image_paths = sorted(test_dir.glob("*.png"), key=lambda p: int(p.stem))

    print(f"Images: {len(image_paths)}")
    print(
        f"Settings: image_size={args.image_size}, conf={args.conf}, max_boxes={args.max_boxes}, "
        f"min_w={args.min_w}, min_h={args.min_h}"
    )

    use_amp = torch.cuda.is_available()
    all_preds = []

    for image_path in tqdm(image_paths, desc="Infer"):
        img = Image.open(image_path).convert("RGB")
        image_id = int(image_path.stem)
        orig_w, orig_h = img.size

        # fixed eval size, boxes mapped back by target_sizes
        resized_inputs = processor(
            images=img,
            return_tensors="pt",
            size={"shortest_edge": args.image_size, "longest_edge": args.image_size},
        )
        pixel_values = resized_inputs["pixel_values"].to(device)
        pixel_mask = resized_inputs.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        else:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
        results = processor.post_process_object_detection(
            outputs,
            threshold=args.conf,
            target_sizes=target_sizes,
        )[0]

        scores = results["scores"].detach().cpu().tolist()
        labels = results["labels"].detach().cpu().tolist()
        boxes = results["boxes"].detach().cpu().tolist()

        preds = []
        for score, label, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = box

            x1 = max(0.0, min(float(x1), orig_w))
            y1 = max(0.0, min(float(y1), orig_h))
            x2 = max(0.0, min(float(x2), orig_w))
            y2 = max(0.0, min(float(y2), orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1
            if w < args.min_w or h < args.min_h:
                continue

            preds.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "score": float(score),
                    "category_id": int(label) + 1,
                }
            )

        preds = sorted(preds, key=lambda x: x["score"], reverse=True)[:args.max_boxes]

        for p in preds:
            x1, y1, x2, y2 = p["bbox_xyxy"]
            all_preds.append(
                {
                    "image_id": image_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(p["score"]),
                    "category_id": int(p["category_id"]),
                }
            )

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_preds, f)

    print("Saved:", out_path)
    print("Total preds:", len(all_preds), "Avg/image:", len(all_preds) / max(1, len(image_paths)))
    summarize_export(all_preds, len(image_paths))


if __name__ == "__main__":
    main()
