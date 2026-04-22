import torch
import torch.nn.functional as F


def collate_fn(batch, processor=None):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_height = max(img.shape[1] for img in pixel_values)
    max_width = max(img.shape[2] for img in pixel_values)

    padded_images = []
    pixel_masks = []

    for img in pixel_values:
        _, h, w = img.shape

        pad_h = max_height - h
        pad_w = max_width - w

        padded_img = F.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(padded_img)

        mask = torch.ones((h, w), dtype=torch.bool)
        padded_mask = F.pad(mask, (0, pad_w, 0, pad_h), value=False)
        pixel_masks.append(padded_mask)

    return {
        "pixel_values": torch.stack(padded_images),
        "pixel_mask": torch.stack(pixel_masks),
        "labels": labels,
    }