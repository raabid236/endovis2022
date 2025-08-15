#!/usr/bin/env python3
"""
Example:
    python inference_segmentation_visual.py \
        --model_ckpt ./checkpoints/best_model.pth \
        --input_dir ./dataset/test \
        --output_dir ./predictions \
        --img-size 256 \
        --num-classes 10
"""

import os
import glob
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.models.segmentation as seg_models

# -----------------------------
# Color palette for 10 classes (0=background)
# -----------------------------
PALETTE = np.array([
    [0, 0, 0],        # class 0 - background (black)
    [255, 0, 0],      # class 1 - red:Tool clasper
    [0, 255, 0],      # class 2 - green:Tool wrist
    [0, 0, 255],      # class 3 - blue:Tool shaft
    [255, 255, 0],    # class 4 - yellow:Suturing needle bad
    [255, 0, 255],    # class 5 - magenta:Thread bad
    [0, 255, 255],    # class 6 - cyan:Suction tool
    [255, 128, 0],    # class 7 - orange:Needle Holder
    [128, 0, 255],    # class 8 - purple:Clamps very bad
    [128, 128, 128],  # class 9 - gray:Catheter
], dtype=np.uint8)

def mask_to_color(mask: np.ndarray) -> Image.Image:
    """
    Convert class ID mask (HxW, int) to RGB image using fixed PALETTE.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        color_mask[mask == class_id] = color
    return Image.fromarray(color_mask, mode='RGB')

@torch.no_grad()
def predict_image(model, image_path, device, img_size, num_classes):
    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size

    img_resized = img.resize((img_size, img_size), resample=Image.BILINEAR)
    img_tensor = TF.to_tensor(img_resized)
    img_tensor = TF.normalize(img_tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Forward pass
    output = model(img_tensor)['out']  # (1,C,H,W)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize mask back to original frame size
    pred_mask = Image.fromarray(pred, mode='L').resize(orig_size, resample=Image.NEAREST)
    return pred_mask

def compute_miou(pred_mask, gt_mask, num_classes):
    ious = []
    per_class_ious = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        gt_inds = (gt_mask == cls)
        intersection = np.logical_and(pred_inds, gt_inds).sum()
        union = np.logical_or(pred_inds, gt_inds).sum()
        if union == 0:
            iou = 1 #float('nan')
        else:
            iou = intersection / union
            ious.append(iou)
        per_class_ious.append(iou)
    miou = np.nanmean(per_class_ious[1:])
    return miou, per_class_ious

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to saved model checkpoint (.pth)')
    parser.add_argument('--input_dir', type=str, required=True, help='Root folder with video_xx/frames_original')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save predictions')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.model_ckpt} ...")
    model = seg_models.deeplabv3_resnet50(pretrained=False, num_classes=args.num_classes)
    ckpt = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Process each video
    videos = sorted([d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))])
    total_miou = []
    total_class_ious = [[] for _ in range(args.num_classes)]
    for video in videos:
        frame_dir = os.path.join(args.input_dir, video, "frames_original")
        if not os.path.isdir(frame_dir):
            continue

        out_mask_dir = os.path.join(args.output_dir, video, "segmentation_pred")
        out_overlay_dir = os.path.join(args.output_dir, video, "overlay_pred")
        os.makedirs(out_mask_dir, exist_ok=True)
        os.makedirs(out_overlay_dir, exist_ok=True)

        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        print(f"[{video}] Processing {len(frame_paths)} frames...")
        for frame_path in frame_paths:
            # Predict mask
            pred_mask = predict_image(model, frame_path, device, args.img_size, args.num_classes)
            pred_mask_np = np.array(pred_mask, dtype=np.uint8)

            # Save grayscale mask
            # pred_mask.save(os.path.join(out_mask_dir, os.path.basename(frame_path)))

            # Create overlay
            # color_mask = mask_to_color(pred_mask_np)
            # orig_img = Image.open(frame_path).convert("RGB")
            # overlay = Image.blend(orig_img, color_mask, alpha=0.5)  # alpha=0.5 for 50% transparency

            # Save overlay
            # overlay.save(os.path.join(out_overlay_dir, os.path.basename(frame_path)))

            # Read ground truth mask
            gt_mask_path = os.path.join(args.input_dir, video, "segmentation", os.path.basename(frame_path))
            if os.path.exists(gt_mask_path):
                gt_mask = Image.open(gt_mask_path).convert("L")
                gt_mask_np = np.array(gt_mask, dtype=np.uint8)
                miou, per_class_ious = compute_miou(pred_mask_np, gt_mask_np, args.num_classes)
                total_miou.append(miou)
                for i, iou in enumerate(per_class_ious):
                    if not np.isnan(iou):
                        total_class_ious[i].append(iou)
                # print(f"{video}/{os.path.basename(frame_path)} mIoU: {miou:.4f}")
                # print("  Per-class IoU:", ["{:.4f}".format(iou) if not np.isnan(iou) else "nan" for iou in per_class_ious])
            else:
                print(f"Ground truth mask not found for {video}/{os.path.basename(frame_path)}")

    if total_miou:
        avg_miou = np.mean(total_miou)
        avg_class_ious = [np.mean(cls_ious) if cls_ious else float('nan') for cls_ious in total_class_ious]
        print(f"Average mIoU over all frames: {avg_miou:.4f}")
        print("Average per-class IoU:", ["{:.4f}".format(iou) if not np.isnan(iou) else "nan" for iou in avg_class_ious])
    else:
        print("No ground truth masks found. Cannot compute mIoU.")

if __name__ == "__main__":
    main()
