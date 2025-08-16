#!/usr/bin/env python3
"""
Example usage:
    python inference_segmentation_visual.py \
        --model_ckpt ./models/best_model.pth \
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
import scipy.ndimage as ndi
import cv2

# -----------------------------
# Color palette for 10 classes (0=background)
# -----------------------------
PALETTE = np.array([
    [0, 0, 0],        # class 0 - black:Patient
    [255, 0, 0],      # class 1 - red:Tool clasper
    [0, 255, 0],      # class 2 - green:Tool wrist
    [0, 0, 255],      # class 3 - blue:Tool shaft
    [255, 255, 0],    # class 4 - yellow:Suturing needle
    [255, 0, 255],    # class 5 - magenta:Thread
    [0, 255, 255],    # class 6 - cyan:Suction tool
    [255, 128, 0],    # class 7 - orange:Needle Holder
    [128, 0, 255],    # class 8 - purple:Clamps
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

def join_islands(mask, class_ids, kernel_size=5):
    mask_out = mask.copy()
    for cls in class_ids:
        binary = (mask == cls).astype(np.uint8)
        closed = ndi.binary_closing(binary, structure=np.ones((kernel_size, kernel_size)))
        mask_out[binary == 1] = 0  # remove original islands
        mask_out[closed == 1] = cls  # add joined regions
    return mask_out

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

        # Store per-class IoU for this video
        video_class_ious = [[] for _ in range(args.num_classes)]

        for frame_path in frame_paths:
            # Predict mask
            pred_mask = predict_image(model, frame_path, device, args.img_size, args.num_classes)
            pred_mask_np = np.array(pred_mask, dtype=np.uint8)
            pred_mask_np = join_islands(pred_mask_np, class_ids=[5], kernel_size=7)

            # Save grayscale mask
            pred_mask.save(os.path.join(out_mask_dir, os.path.basename(frame_path)))

            # Create and save overlay
            color_mask = mask_to_color(pred_mask_np)
            orig_img = Image.open(frame_path).convert("RGB")
            overlay = Image.blend(orig_img, color_mask, alpha=0.5)
            overlay.save(os.path.join(out_overlay_dir, os.path.basename(frame_path)))

            # Read ground truth mask
            gt_mask_path = os.path.join(args.input_dir, video, "segmentation", os.path.basename(frame_path))
            if os.path.exists(gt_mask_path):
                gt_mask = Image.open(gt_mask_path).convert("L")
                gt_mask_np = np.array(gt_mask, dtype=np.uint8)
                miou, per_class_ious = compute_miou(pred_mask_np, gt_mask_np, args.num_classes)
                for i, iou in enumerate(per_class_ious):
                    if not np.isnan(iou):
                        video_class_ious[i].append(iou)
            else:
                print(f"Ground truth mask not found for {video}/{os.path.basename(frame_path)}")

        # Compute per-class IoU for this video
        video_avg_class_ious = [np.mean(cls_ious) if cls_ious else float('nan') for cls_ious in video_class_ious]
        total_class_ious = [
            total_class_ious[i] + [video_avg_class_ious[i]] for i in range(args.num_classes)
        ]
        print(f"Per-class IoU for video {video}:", ["{:.4f}".format(iou) if not np.isnan(iou) else "nan" for iou in video_avg_class_ious])

    # After all videos compute the mean IOUs
    print(f"\nMean mIoU across all videos including background: {np.nanmean(total_class_ious):.4f}")
    print(f"\nMean mIoU across all videos excluding background: {np.nanmean(total_class_ious[1:]):.4f}")

if __name__ == "__main__":
    main()
