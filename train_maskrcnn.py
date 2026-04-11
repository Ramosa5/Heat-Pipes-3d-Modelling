import os
import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageDraw
import torchvision
from torchvision.transforms import functional as F


IMAGES_DIR = "train"
ANNOTATION_FILE = "_annotations.coco.json"
MODEL_SAVE_PATH = "new_best_maskrcnn_bubble.pth"

BATCH_SIZE = 1
NUM_EPOCHS = 15
LEARNING_RATE = 5e-4
TRAIN_RATIO = 0.8
SEED = 42
NUM_WORKERS = 0


class CocoInstanceDataset(Dataset):
    def __init__(self, images_dir, annotation_file):
        self.images_dir = images_dir

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco["categories"]

        self.anns_by_image = defaultdict(list)
        for ann in self.annotations:
            self.anns_by_image[ann["image_id"]].append(ann)

        valid_categories = []
        for cat in self.categories:
            if cat["name"].lower() != "objects":
                valid_categories.append(cat)

        if len(valid_categories) == 0:
            seen_cat_ids = sorted(set(ann["category_id"] for ann in self.annotations))
            self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(seen_cat_ids)}
        else:
            self.cat_id_to_label = {
                cat["id"]: i + 1
                for i, cat in enumerate(sorted(valid_categories, key=lambda x: x["id"]))
            }

    def __len__(self):
        return len(self.images)

    def polygon_to_mask(self, segmentation, width, height):
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)

        if isinstance(segmentation, list):
            for poly in segmentation:
                if not isinstance(poly, list):
                    continue
                if len(poly) < 6:
                    continue
                points = []
                for i in range(0, len(poly), 2):
                    try:
                        px = float(poly[i])
                        py = float(poly[i + 1])
                        points.append((px, py))
                    except Exception:
                        pass

                if len(points) >= 3:
                    draw.polygon(points, outline=1, fill=1)

        mask = torch.from_numpy(np.array(mask_img, dtype=np.uint8))
        return mask

    def __getitem__(self, idx):
        try:
            img_info = self.images[idx]
            image_id = img_info["id"]
            file_name = img_info["file_name"]

            img_path = os.path.join(self.images_dir, file_name)
            image = Image.open(img_path).convert("RGB")
            width, height = image.size

            anns = self.anns_by_image[image_id]

            boxes = []
            labels = []
            masks = []
            areas = []
            iscrowd = []

            for ann in anns:
                cat_id = ann.get("category_id")
                if cat_id not in self.cat_id_to_label:
                    continue

                bbox = ann.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue

                try:
                    x = float(bbox[0])
                    y = float(bbox[1])
                    w = float(bbox[2])
                    h = float(bbox[3])
                except Exception:
                    continue

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

                if x2 <= x1 or y2 <= y1:
                    continue

                seg = ann.get("segmentation", [])

                if isinstance(seg, list) and len(seg) > 0:
                    mask = self.polygon_to_mask(seg, width, height)
                else:
                    mask = torch.zeros((height, width), dtype=torch.uint8)
                    x1i = max(0, int(x1))
                    y1i = max(0, int(y1))
                    x2i = min(width, int(x2))
                    y2i = min(height, int(y2))
                    if x2i > x1i and y2i > y1i:
                        mask[y1i:y2i, x1i:x2i] = 1

                if mask.sum().item() == 0:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_label[cat_id])
                masks.append(mask)

                area_value = ann.get("area", w * h)
                try:
                    area_value = float(area_value)
                except Exception:
                    area_value = w * h

                areas.append(area_value)

                try:
                    iscrowd_value = int(ann.get("iscrowd", 0))
                except Exception:
                    iscrowd_value = 0
                iscrowd.append(iscrowd_value)

            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                masks = torch.zeros((0, height, width), dtype=torch.uint8)
                areas = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
                masks = torch.stack(masks).to(torch.uint8)
                areas = torch.tensor(areas, dtype=torch.float32)
                iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([image_id]),
                "area": areas,
                "iscrowd": iscrowd,
            }

            image = F.to_tensor(image)
            return image, target

        except Exception as e:
            print(f"[DATASET ERROR] idx={idx}, file={self.images[idx].get('file_name', 'unknown')}, error={e}")
            raise


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise ValueError("Batch jest pusty po odfiltrowaniu None.")
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features_box, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0 or step == 1:
            print(f"  step {step}/{len(loader)} loss={loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())
        total_loss += loss.item()

        if step % 20 == 0 or step == 1:
            print(f"  val step {step}/{len(loader)} loss={loss.item():.4f}")

    return total_loss / max(len(loader), 1)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Urządzenie:", device)

    dataset = CocoInstanceDataset(IMAGES_DIR, ANNOTATION_FILE)

    n_total = len(dataset)
    indices = list(range(n_total))
    random.shuffle(indices)

    n_train = int(TRAIN_RATIO * n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = get_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print(f"Liczba obrazów: {n_total}")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== EPOCH {epoch + 1}/{NUM_EPOCHS} ===")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Zapisano najlepszy model: {MODEL_SAVE_PATH}")

    print("Koniec treningu.")


if __name__ == "__main__":
    main()