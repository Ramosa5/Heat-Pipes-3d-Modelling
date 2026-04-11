import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


MODEL_PATH = "new_best_maskrcnn_bubble.pth"
IMAGE_PATH = "train/frame_0182_f2987_t99-567_jpg.rf.KXikt2igQ1HndbJSd6CJ.jpg"
SCORE_THRESHOLD = 0.3


def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")
    x = F.to_tensor(image).to(device)

    with torch.no_grad():
        pred = model([x])[0]

    img_np = np.array(image)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    masks = pred["masks"].cpu().numpy()  # [N, 1, H, W]

    for box, score, mask in zip(boxes, scores, masks):
        if score < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x1, y1, f"{score:.2f}")

        binary_mask = (mask[0] > 0.3).astype(np.uint8)
        colored = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.float32)
        colored[..., 1] = binary_mask  # kanał G
        colored[..., 3] = binary_mask * 0.35  # alfa
        ax.imshow(colored)

    ax.set_title("Predykcja Mask R-CNN")
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    main()