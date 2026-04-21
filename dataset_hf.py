from pathlib import Path
import random

from PIL import Image, ImageEnhance, ImageFilter
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class DigitDetectionDataset(Dataset):
    def __init__(
        self,
        image_dir,
        annotation_file,
        processor,
        is_train=False,
        train_sizes=None,
        eval_size=512,
        subset_size=None,
        subset_seed=42,
    ):
        self.image_dir = Path(image_dir)
        self.coco = COCO(annotation_file)
        self.image_ids = sorted(self.coco.imgs.keys())
        self.processor = processor
        self.is_train = is_train
        self.train_sizes = train_sizes or [512]
        self.eval_size = eval_size

        valid_image_ids = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid_anns = self._filter_annotations(anns)
            if valid_anns:
                valid_image_ids.append(img_id)

        self.image_ids = valid_image_ids

        if subset_size is not None and subset_size < len(self.image_ids):
            rng = random.Random(subset_seed)
            sampled_ids = self.image_ids[:]
            rng.shuffle(sampled_ids)
            self.image_ids = sampled_ids[:subset_size]
        
        print(
            f"Loaded {len(self.image_ids)} valid images "
            f"out of {len(self.coco.imgs)} total"
        )

    def __len__(self):
        return len(self.image_ids)

    def _filter_annotations(self, anns):
        valid_anns = []
        for ann in anns:
            if "bbox" not in ann or len(ann["bbox"]) != 4:
                continue

            x, y, w, h = ann["bbox"]
            if w is None or h is None:
                continue
            if w <= 1 or h <= 1:
                continue

            category_id = ann["category_id"] - 1
            if category_id < 0 or category_id > 9:
                continue

            valid_anns.append(ann)
        return valid_anns

    def _build_annotations(self, anns):
        converted_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            category_id = ann["category_id"] - 1

            new_ann = ann.copy()
            new_ann["bbox"] = [float(x), float(y), float(w), float(h)]
            new_ann["area"] = float(w * h)
            new_ann["category_id"] = category_id
            converted_anns.append(new_ann)
        return converted_anns

    def _clip_box(self, x, y, w, h, width, height):
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(width), x + w)
        y2 = min(float(height), y + h)
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w <= 1 or new_h <= 1:
            return None
        return [x1, y1, new_w, new_h]

    def _translate(self, image, anns):
        width, height = image.size
        max_dx = int(0.05 * width)
        max_dy = int(0.05 * height)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)

        image = image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            fillcolor=(0, 0, 0),
        )

        new_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            clipped = self._clip_box(x + dx, y + dy, w, h, width, height)
            if clipped is not None:
                ann = ann.copy()
                ann["bbox"] = clipped
                new_anns.append(ann)
        return image, new_anns

    def _center_scale(self, image, anns):
        width, height = image.size
        scale = random.uniform(0.90, 1.10)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))

        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))

        paste_x = (width - new_w) // 2
        paste_y = (height - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))

        new_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x = x * scale + paste_x
            y = y * scale + paste_y
            w = w * scale
            h = h * scale
            clipped = self._clip_box(x, y, w, h, width, height)
            if clipped is not None:
                ann = ann.copy()
                ann["bbox"] = clipped
                new_anns.append(ann)

        return canvas, new_anns

    def _random_crop(self, image, anns):
        width, height = image.size
        crop_ratio = random.uniform(0.94, 1.0)
        crop_w = int(width * crop_ratio)
        crop_h = int(height * crop_ratio)

        if crop_w >= width or crop_h >= height:
            return image, anns

        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        right = left + crop_w
        bottom = top + crop_h

        cropped = image.crop((left, top, right, bottom)).resize(
            (width, height), Image.BILINEAR
        )

        scale_x = width / crop_w
        scale_y = height / crop_h

        new_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            new_x = x - left
            new_y = y - top
            clipped = self._clip_box(new_x, new_y, w, h, crop_w, crop_h)
            if clipped is None:
                continue

            cx, cy, cw, ch = clipped
            ann = ann.copy()
            ann["bbox"] = [
                cx * scale_x,
                cy * scale_y,
                cw * scale_x,
                ch * scale_y,
            ]
            new_anns.append(ann)

        if not new_anns:
            return image, anns
        return cropped, new_anns

    def _photometric(self, image):
        if random.random() < 0.7:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.75, 1.25)
            )
        if random.random() < 0.7:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.75, 1.25)
            )
        if random.random() < 0.20:
            image = ImageEnhance.Color(image).enhance(
                random.uniform(0.90, 1.10)
            )
        if random.random() < 0.15:
            image = ImageEnhance.Sharpness(image).enhance(
                random.uniform(0.90, 1.15)
            )
        if random.random() < 0.15:
            image = image.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8))
            )
        return image

    def _augment_image_and_annotations(self, image, anns):
        anns = [ann.copy() for ann in anns]

        # Never use horizontal flip for digits.
        if random.random() < 0.35:
            image, anns = self._translate(image, anns)

        if random.random() < 0.35:
            image, anns = self._center_scale(image, anns)

        if random.random() < 0.20:
            image, anns = self._random_crop(image, anns)

        image = self._photometric(image)
        return image, anns

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        anns = self._filter_annotations(anns)

        if self.is_train:
            image, anns = self._augment_image_and_annotations(image, anns)

        converted_anns = self._build_annotations(anns)

        if not converted_anns:
            raise RuntimeError(f"No valid annotations for image_id={image_id}")

        annotations = {
            "image_id": image_id,
            "annotations": converted_anns,
        }

        chosen_size = random.choice(self.train_sizes) if self.is_train else self.eval_size
        encoding = self.processor(
            images=image,
            annotations=annotations,
            return_tensors="pt",
            size={"shortest_edge": chosen_size, "longest_edge": chosen_size},
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]
        return pixel_values, labels
