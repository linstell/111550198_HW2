from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
    DetrForObjectDetection,
)

from dataset_hf import DigitDetectionDataset
from utils import collate_fn


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def has_invalid_boxes(labels):
    for target in labels:
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return True
        if torch.isnan(boxes).any() or torch.isinf(boxes).any():
            return True
    return False


def coco_eval_map(coco_gt, predictions):
    if not predictions:
        return 0.0

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(predictions, tmp)
        tmp_path = tmp.name

    coco_dt = coco_gt.loadRes(tmp_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    train_loss,
    val_loss,
    val_map,
    train_sizes,
    eval_size,
    num_queries,
    num_labels,
    train_subset_size,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_map": val_map,
            "meta": {
                "train_sizes": train_sizes,
                "eval_size": eval_size,
                "num_queries": num_queries,
                "num_labels": num_labels,
                "train_subset_size": train_subset_size,
                "model_type": "deformable_detr",
                "backbone": "resnet50",
            },
        },
        path,
    )


def build_model(num_labels: int, num_queries: int) -> DeformableDetrForObjectDetection:
    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {str(i): i for i in range(num_labels)}

    config = DeformableDetrConfig()
    config.use_pretrained_backbone = False
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id
    config.auxiliary_loss = True
    config.num_queries = num_queries

    model = DeformableDetrForObjectDetection(config)

    # Homework-compliant: pretrained backbone only
    pretrained_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    missing, unexpected = model.model.backbone.load_state_dict(
        pretrained_detr.model.backbone.state_dict(),
        strict=False,
    )
    print("Backbone load:", {"missing": len(missing), "unexpected": len(unexpected)})
    return model


def build_optimizer(model, backbone_lr: float, head_lr: float, weight_decay: float):
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return max(0.10, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model,
    processor,
    val_loader,
    val_dataset,
    device,
    use_amp,
    val_conf,
    val_max_boxes,
):
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    val_predictions = []

    val_bar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc="Validation",
        ascii=True,
        ncols=110,
        file=sys.stdout,
    )

    with torch.no_grad():
        for _, batch in val_bar:
            labels_cpu = batch["labels"]
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels_cpu]

            if has_invalid_boxes(labels):
                continue

            try:
                if use_amp:
                    with autocast(device_type="cuda"):
                        outputs = model(
                            pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels,
                        )
                        loss = outputs.loss
                else:
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
            except Exception as err:
                print(f"\nValidation batch failed: {err}")
                raise

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            val_loss_sum += float(loss.item())
            val_batches += 1
            val_bar.set_postfix(loss=f"{val_loss_sum / max(1, val_batches):.4f}")

            target_sizes = torch.stack([target["orig_size"] for target in labels_cpu])

            results = processor.post_process_object_detection(
                outputs,
                threshold=val_conf,
                target_sizes=target_sizes,
            )

            for target, result in zip(labels_cpu, results):
                image_id = int(target["image_id"].item())
                scores = result["scores"].cpu().tolist()
                labels_pred = result["labels"].cpu().tolist()
                boxes = result["boxes"].cpu().tolist()

                preds = []
                for score, label, box in zip(scores, labels_pred, boxes):
                    x1, y1, x2, y2 = box
                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)
                    x2 = max(0.0, x2)
                    y2 = max(0.0, y2)
                    w = x2 - x1
                    h = y2 - y1
                    if w <= 1 or h <= 1:
                        continue

                    preds.append(
                        {
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "score": float(score),
                            "category_id": int(label) + 1,
                        }
                    )

                preds = sorted(preds, key=lambda x: x["score"], reverse=True)[:val_max_boxes]

                for p in preds:
                    x1, y1, x2, y2 = p["bbox_xyxy"]
                    val_predictions.append(
                        {
                            "image_id": image_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": p["score"],
                            "category_id": p["category_id"],
                        }
                    )

    avg_val_loss = val_loss_sum / max(val_batches, 1)
    val_map = coco_eval_map(val_dataset.coco, val_predictions)
    return avg_val_loss, val_map


def main():
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f}GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    num_labels = 10

    # Faster but still strong config
    train_sizes = [512]
    eval_size = 512
    train_subset_size = 10000      # set None for full training
    batch_size = 2
    grad_accum_steps = 1
    num_epochs = 20
    warmup_epochs = 3
    num_queries = 100
    eval_every = 2

    backbone_lr = 1e-5
    head_lr = 1e-4
    weight_decay = 1e-4
    max_norm = 0.1

    val_conf = 0.03
    val_max_boxes = 100

    processor = DeformableDetrImageProcessor(
        format="coco_detection",
        do_resize=True,
    )

    model = build_model(num_labels=num_labels, num_queries=num_queries)
    model.to(device)

    train_dataset = DigitDetectionDataset(
        image_dir="train",
        annotation_file="train.json",
        processor=processor,
        is_train=True,
        train_sizes=train_sizes,
        eval_size=eval_size,
        subset_size=train_subset_size,
        subset_seed=42,
    )
    val_dataset = DigitDetectionDataset(
        image_dir="valid",
        annotation_file="valid.json",
        processor=processor,
        is_train=False,
        train_sizes=train_sizes,
        eval_size=eval_size,
        subset_size=None,
        subset_seed=42,
    )

    print(
        f"\nConfig: train_sizes={train_sizes}, eval_size={eval_size}, "
        f"subset={train_subset_size}, batch={batch_size}, accum={grad_accum_steps}, "
        f"epochs={num_epochs}, warmup={warmup_epochs}, eval_every={eval_every}, "
        f"backbone_lr={backbone_lr}, head_lr={head_lr}, queries={num_queries}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = build_optimizer(
        model,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
    )

    use_amp = torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None

    subset_tag = "full" if train_subset_size is None else f"subset{train_subset_size}"
    run_name = f"deformable_detr_r50_sz512_q{num_queries}_ep{num_epochs}_{subset_tag}_lr1e4"
    save_dir = Path("checkpoints") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir.resolve()}")

    best_val_map = -1.0

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        optimizer.zero_grad(set_to_none=True)

        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
            ascii=True,
            ncols=110,
            file=sys.stdout,
        )

        for i, batch in train_bar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            if has_invalid_boxes(labels):
                continue

            try:
                if use_amp:
                    with autocast(device_type="cuda"):
                        outputs = model(
                            pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels,
                        )
                        loss = outputs.loss / grad_accum_steps
                else:
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / grad_accum_steps
            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    print(f"\nOOM at batch {i}; skipping batch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = ((i + 1) % grad_accum_steps == 0) or ((i + 1) == len(train_loader))
            if should_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss_sum += float(loss.item()) * grad_accum_steps
            train_batches += 1
            train_bar.set_postfix(loss=f"{train_loss_sum / max(1, train_batches):.4f}")

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        do_eval = ((epoch + 1) % eval_every == 0) or ((epoch + 1) == num_epochs)

        if do_eval:
            avg_val_loss, val_map = evaluate(
                model=model,
                processor=processor,
                val_loader=val_loader,
                val_dataset=val_dataset,
                device=device,
                use_amp=use_amp,
                val_conf=val_conf,
                val_max_boxes=val_max_boxes,
            )

            print(
                f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}, "
                f"Val mAP: {val_map:.4f}, "
                f"backbone_lr={current_lrs[0]:.2e}, head_lr={current_lrs[1]:.2e}"
            )

            save_checkpoint(
                save_dir / "last_model.pth",
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                avg_train_loss,
                avg_val_loss,
                val_map,
                train_sizes=train_sizes,
                eval_size=eval_size,
                num_queries=num_queries,
                num_labels=num_labels,
                train_subset_size=train_subset_size,
            )

            if val_map > best_val_map:
                best_val_map = val_map
                save_checkpoint(
                    save_dir / "best_model.pth",
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    avg_train_loss,
                    avg_val_loss,
                    val_map,
                    train_sizes=train_sizes,
                    eval_size=eval_size,
                    num_queries=num_queries,
                    num_labels=num_labels,
                    train_subset_size=train_subset_size,
                )
                print(f"-> Saved best model by mAP (val_map: {val_map:.4f})")
        else:
            print(
                f"Epoch {epoch + 1}: skipped validation, "
                f"backbone_lr={current_lrs[0]:.2e}, head_lr={current_lrs[1]:.2e}"
            )

        scheduler.step()

    print(f"\nTraining finished. Best val mAP: {best_val_map:.4f}")


if __name__ == "__main__":
    main()
