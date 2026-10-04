#!/usr/bin/env python3
"""
Model Output Image Exporter
---------------------------
Extracts a frame from a provided test video, executes all three vision models
(DeepLabV3 MobileNet, YOLOv8-Nano, Depth Anything V2) along with the full
Assistive SafePath Spatial Fusion, and saves individual images named after each
model used into the 'image_output' folder.

Usage:
    python export_model_images.py
    python export_model_images.py --video "test_11 (1).mp4" --frame 60
    python export_model_images.py --output-dir "image_output"
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2

# Ensure CUDA and cuDNN libraries from PyTorch are available for ONNX Runtime
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + ';' + os.environ.get('PATH', '')
except Exception:
    pass

import onnxruntime as ort

# COCO Class Label Mapping for Hazards
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 15: 'cat', 16: 'dog', 24: 'backpack', 26: 'handbag', 28: 'suitcase'
}

CLASS_COLORS = {
    'person': (0, 255, 128),      # Mint green
    'car': (255, 180, 0),         # Cyan-blue
    'bicycle': (0, 215, 255),     # Gold
    'motorcycle': (0, 165, 255),  # Orange
    'bus': (255, 100, 0),         # Deep blue
    'truck': (200, 80, 0),        # Navy
    'traffic light': (0, 255, 255),
    'dog': (180, 105, 255),       # Pink
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def resolve_file(filename, search_dirs):
    """Searches for a file across multiple directories."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists(filename):
        return os.path.abspath(filename)
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None

def prepare_tensor(image, w, h, apply_norm=True):
    """Resizes, standardizes channel format, and converts image to NCHW float32 tensor."""
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if apply_norm:
        rgb = (rgb - MEAN) / STD
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

def draw_header(canvas, title, subtitle=None, bg_color=(20, 20, 20), text_color=(0, 255, 255)):
    """Draws a standardized header bar on top of an image."""
    h, w = canvas.shape[:2]
    header_h = 42
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0.0, canvas)
    cv2.line(canvas, (0, header_h), (w, header_h), (80, 80, 80), 1)

    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
    if subtitle:
        (sub_w, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(canvas, subtitle, (w - sub_w - 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    search_dirs = [script_dir, parent_dir]

    parser = argparse.ArgumentParser(description="Export individual model output images from a test video")
    parser.add_argument("--video", type=str, default=None, help="Path or name of test video file")
    parser.add_argument("--frame", type=int, default=50, help="Frame index to extract (default: 50)")
    parser.add_argument("--output-dir", type=str, default="image_output", help="Directory to save model output images")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold (default: 0.35)")
    args = parser.parse_args()

    # Determine Output Directory (in test folder)
    if os.path.isabs(args.output_dir):
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Locate Video
    video_path = None
    if args.video:
        video_path = resolve_file(args.video, search_dirs)
    else:
        # Auto-discover video in script_dir, preferring test_11 (1).mp4
        candidates = [
            "test_11 (1).mp4", "test_11 (2).mp4", "test_11 (3).mp4",
            "test_08_market.mp4", "test_01_urban_crowd.mp4","test_05_san_francisco_street.mp4"
        ]
        for c in candidates:
            resolved = resolve_file(c, search_dirs)
            if resolved:
                video_path = resolved
                break
        if not video_path:
            mp4_list = glob.glob(os.path.join(script_dir, "*.mp4")) + glob.glob(os.path.join(parent_dir, "*.mp4"))
            if mp4_list:
                video_path = mp4_list[0]

    if not video_path or not os.path.exists(video_path):
        print(f"[ERROR] Could not find any valid test video in {search_dirs}")
        sys.exit(1)

    print(f"[1/5] Using test video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame_idx = min(max(0, args.frame), max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, raw_frame = cap.read()
    cap.release()

    if not ret or raw_frame is None:
        print(f"[ERROR] Could not read frame {target_frame_idx} from {video_path}")
        sys.exit(1)

    # Standardize working resolution to 640x360 for high-clarity output
    frame = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_AREA)
    H, W = frame.shape[:2]
    print(f"      Extracted frame {target_frame_idx}/{total_frames} (Resolution: {W}x{H})")

    # Save original input image
    input_path = os.path.join(out_dir, "input_frame.jpg")
    cv2.imwrite(input_path, frame)
    print(f"      Saved: {input_path}")

    # Initialize ONNX Runtime Providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # --------------------------------------------------------------------------
    # MODEL 1: DeepLabV3 MobileNet SafePath Segmentation
    # --------------------------------------------------------------------------
    print("\n[2/5] Running Model 1: DeepLabV3 MobileNet...")
    deeplab_file = resolve_file("deeplabv3_mobilenet_safepath.onnx", search_dirs)
    if not deeplab_file:
        deeplab_file = resolve_file("deeplabv3_mobilenet_safepath_fp16.onnx", search_dirs)

    if not deeplab_file:
        print("[ERROR] Could not find DeepLabV3 ONNX model.")
        sys.exit(1)

    seg_sess = ort.InferenceSession(deeplab_file, opts, providers=providers)
    seg_in_name = seg_sess.get_inputs()[0].name

    seg_tensor = prepare_tensor(frame, 384, 256, apply_norm=True)
    seg_out = seg_sess.run(None, {seg_in_name: seg_tensor})[0]
    pred_labels = np.argmax(seg_out[0], axis=0).astype(np.uint8)  # 256x384

    # Class 1 = Walkable Path
    raw_walkable = (pred_labels == 1).astype(np.uint8) * 255
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    safe_walkable = cv2.erode(raw_walkable, erode_kernel, iterations=1)

    # Upsample safe mask to frame dimensions
    full_safe_mask = cv2.resize(safe_walkable, (W, H), interpolation=cv2.INTER_NEAREST)
    walkable_pixels = cv2.countNonZero(full_safe_mask)
    coverage_pct = (walkable_pixels / float(W * H)) * 100.0

    # Compose DeepLab visualization image
    deeplab_img = frame.copy()
    overlay_path = deeplab_img.copy()
    overlay_path[full_safe_mask > 0] = [200, 220, 0]  # Bright Cyan-Green
    deeplab_img = cv2.addWeighted(overlay_path, 0.45, deeplab_img, 0.55, 0.0)

    # Safety buffer boundary outline
    contours, _ = cv2.findContours(full_safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(deeplab_img, contours, -1, (255, 255, 255), 2)

    draw_header(deeplab_img, "DEEPLABV3-MOBILENET SAFEPATH", 
                f"Walkable Coverage: {coverage_pct:.1f}%", 
                bg_color=(15, 30, 20), text_color=(0, 255, 160))

    deeplab_out_path = os.path.join(out_dir, "deeplabv3_mobilenet.jpg")
    cv2.imwrite(deeplab_out_path, deeplab_img)
    print(f"      Saved: {deeplab_out_path}")

    # --------------------------------------------------------------------------
    # MODEL 2: YOLOv8-Nano Dynamic Hazard Detection
    # --------------------------------------------------------------------------
    print("\n[3/5] Running Model 2: YOLOv8 Hazards...")
    yolo_file = resolve_file("yolov8n_hazards.onnx", search_dirs)
    if not yolo_file:
        print("[ERROR] Could not find YOLOv8 ONNX model.")
        sys.exit(1)

    yolo_sess = ort.InferenceSession(yolo_file, opts, providers=providers)
    yolo_in_name = yolo_sess.get_inputs()[0].name

    yolo_tensor = prepare_tensor(frame, 640, 480, apply_norm=False)
    yolo_out = yolo_sess.run(None, {yolo_in_name: yolo_tensor})[0]

    preds = yolo_out[0].T  # (6300, 84)
    scores = preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)

    mask = confs > args.conf
    valid_boxes = preds[mask, :4]
    valid_confs = confs[mask]
    valid_classes = class_ids[mask]

    detected_boxes = []
    if len(valid_confs) > 0:
        boxes_xywh = []
        for b in valid_boxes:
            cx, cy, bw, bh = b
            boxes_xywh.append([int(cx - bw / 2.0), int(cy - bh / 2.0), int(bw), int(bh)])
        indices = cv2.dnn.NMSBoxes(boxes_xywh, valid_confs.tolist(), args.conf, 0.45)
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                cx, cy, bw, bh = valid_boxes[idx]
                norm_x = (cx - bw / 2.0) / 640.0
                norm_y = (cy - bh / 2.0) / 480.0
                norm_w = bw / 640.0
                norm_h = bh / 480.0
                detected_boxes.append((norm_x, norm_y, norm_w, norm_h, int(valid_classes[idx]), float(valid_confs[idx])))

    yolo_img = frame.copy()
    for norm_x, norm_y, norm_w, norm_h, cls_id, conf in detected_boxes:
        bx = max(0, int(norm_x * W))
        by = max(0, int(norm_y * H))
        bw = max(1, int(norm_w * W))
        bh = max(1, int(norm_h * H))
        cname = COCO_CLASSES.get(cls_id, f"ID:{cls_id}")
        color = CLASS_COLORS.get(cname, (0, 215, 255))

        cv2.rectangle(yolo_img, (bx, by), (bx + bw, by + bh), color, 2)
        lbl = f"{cname} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(yolo_img, (bx, max(0, by - 18)), (bx + tw + 6, by), color, -1)
        cv2.putText(yolo_img, lbl, (bx + 3, max(12, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 10, 10), 1, cv2.LINE_AA)

    draw_header(yolo_img, "YOLOV8-NANO HAZARD DETECTION", 
                f"Objects Detected: {len(detected_boxes)}", 
                bg_color=(25, 25, 25), text_color=(0, 255, 255))

    yolo_out_path = os.path.join(out_dir, "yolov8_hazards.jpg")
    cv2.imwrite(yolo_out_path, yolo_img)
    print(f"      Saved: {yolo_out_path}")

    # --------------------------------------------------------------------------
    # MODEL 3: Depth Anything V2 Monocular Metric Depth
    # --------------------------------------------------------------------------
    print("\n[4/5] Running Model 3: Depth Anything V2...")
    depth_file = resolve_file("depth_anything_v2_small.onnx", search_dirs)
    if not depth_file:
        print("[ERROR] Could not find Depth Anything V2 ONNX model.")
        sys.exit(1)

    depth_sess = ort.InferenceSession(depth_file, opts, providers=providers)
    depth_in_name = depth_sess.get_inputs()[0].name

    depth_tensor = prepare_tensor(frame, 518, 518, apply_norm=True)
    depth_out = depth_sess.run(None, {depth_in_name: depth_tensor})[0]
    raw_depth = depth_out[0]  # (518, 518)

    d_min, d_max = raw_depth.min(), raw_depth.max()
    if d_max > d_min:
        depth_u8 = ((raw_depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    else:
        depth_u8 = np.zeros_like(raw_depth, dtype=np.uint8)

    depth_resized = cv2.resize(depth_u8, (W, H), interpolation=cv2.INTER_CUBIC)
    depth_colormap = cv2.applyColorMap(depth_resized, cv2.COLORMAP_INFERNO)

    # Overlay bounding box distance readings on depth map
    depth_img = depth_colormap.copy()
    for norm_x, norm_y, norm_w, norm_h, cls_id, _ in detected_boxes:
        bx = max(0, int(norm_x * W))
        by = max(0, int(norm_y * H))
        bw = max(1, int(norm_w * W))
        bh = max(1, int(norm_h * H))
        cname = COCO_CLASSES.get(cls_id, f"ID:{cls_id}")

        roi_depth = depth_resized[by:by+bh, bx:bx+bw]
        md = float(np.mean(roi_depth)) if roi_depth.size > 0 else 0.0
        est_d = max(0.5, round((255.0 - md) / 255.0 * 4.5 + 0.5, 1)) if md > 0 else 3.5

        cv2.rectangle(depth_img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
        tag = f"{cname} | {est_d:.1f}m"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(depth_img, (bx, max(0, by - 18)), (bx + tw + 6, by), (20, 20, 20), -1)
        cv2.putText(depth_img, tag, (bx + 3, max(12, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

    draw_header(depth_img, "DEPTH ANYTHING V2 METRIC DEPTH", 
                "Inferno Colormap (518x518)", 
                bg_color=(25, 15, 35), text_color=(255, 180, 0))

    depth_out_path = os.path.join(out_dir, "depth_anything_v2.jpg")
    cv2.imwrite(depth_out_path, depth_img)
    print(f"      Saved: {depth_out_path}")

    # --------------------------------------------------------------------------
    # MODEL FUSION: SafePath Spatial Fusion & Navigation HUD
    # --------------------------------------------------------------------------
    print("\n[5/5] Synthesizing Spatial Fusion HUD & Quad Panel...")
    fusion_img = frame.copy()

    # Walkable Path AR Overlay (Green)
    green_overlay = fusion_img.copy()
    green_overlay[full_safe_mask > 0] = [0, 255, 0]
    fusion_img = cv2.addWeighted(green_overlay, 0.35, fusion_img, 0.65, 0.0)

    # Sector analysis on lower corridor (y in [128, 256] of 256x384 mask)
    lower_safe = safe_walkable[128:256, :]
    sec_l = int(np.count_nonzero(lower_safe[:, 0:128]))
    sec_c = int(np.count_nonzero(lower_safe[:, 128:256]))
    sec_r = int(np.count_nonzero(lower_safe[:, 256:384]))

    sectors = {
        'LEFT': {'walkable': sec_l, 'hazards': 0, 'near': False},
        'CENTER': {'walkable': sec_c, 'hazards': 0, 'near': False},
        'RIGHT': {'walkable': sec_r, 'hazards': 0, 'near': False}
    }

    for norm_x, norm_y, norm_w, norm_h, cls_id, _ in detected_boxes:
        bx = max(0, int(norm_x * W))
        by = max(0, int(norm_y * H))
        bw = max(1, int(norm_w * W))
        bh = max(1, int(norm_h * H))
        cname = COCO_CLASSES.get(cls_id, f"ID:{cls_id}")

        # Overlap with DeepLab mask
        x1_dl = max(0, int(norm_x * 384))
        y1_dl = max(0, int(norm_y * 256))
        x2_dl = min(384, int((norm_x + norm_w) * 384))
        y2_dl = min(256, int((norm_y + norm_h) * 256))
        dl_w = max(0, x2_dl - x1_dl)
        dl_h = max(0, y2_dl - y1_dl)

        overlap = 0.0
        if dl_w > 0 and dl_h > 0:
            path_roi = safe_walkable[y1_dl:y2_dl, x1_dl:x2_dl]
            overlap = cv2.countNonZero(path_roi) / float(dl_w * dl_h)

        # Depth sampling
        roi_depth = depth_resized[by:by+bh, bx:bx+bw]
        md = float(np.mean(roi_depth)) if roi_depth.size > 0 else 0.0
        est_d = max(0.5, round((255.0 - md) / 255.0 * 4.5 + 0.5, 1)) if md > 0 else 3.5
        is_near = (est_d <= 1.8) or (md > 175.0)

        cx = norm_x + norm_w / 2.0
        sec_key = 'LEFT' if cx < 0.35 else ('RIGHT' if cx > 0.65 else 'CENTER')
        if is_near:
            sectors[sec_key]['near'] = True

        if overlap > 0.15:
            # On-Path Active Threat (Red)
            box_color = (0, 0, 255)
            tag = f"HAZARD: {cname.upper()} ({est_d:.1f}m)"
            sectors[sec_key]['hazards'] += 1
        else:
            # Off-Path Safe Obstacle (Yellow)
            box_color = (0, 255, 255)
            tag = f"{cname} ({est_d:.1f}m - Clear)"

        cv2.rectangle(fusion_img, (bx, by), (bx + bw, by + bh), box_color, 2)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(fusion_img, (bx, max(0, by - 18)), (bx + tw + 6, by), box_color, -1)
        cv2.putText(fusion_img, tag, (bx + 3, max(12, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 10, 10), 1, cv2.LINE_AA)

    # Assistive Navigation Steering State Machine
    center_blocked = sectors['CENTER']['near'] or sectors['CENTER']['hazards'] > 0
    left_walk = sectors['LEFT']['walkable'] > 200 and not sectors['LEFT']['near']
    right_walk = sectors['RIGHT']['walkable'] > 200 and not sectors['RIGHT']['near']

    if center_blocked:
        if left_walk and not right_walk:
            nav_msg, nav_col = "HAZARD IN CENTER -> VEER LEFT", (0, 220, 255)
        elif right_walk and not left_walk:
            nav_msg, nav_col = "HAZARD IN CENTER -> VEER RIGHT", (0, 220, 255)
        elif right_walk and left_walk:
            nav_msg, nav_col = "HAZARD IN CENTER -> VEER RIGHT", (0, 220, 255)
        else:
            nav_msg, nav_col = "CROWD BLOCKED -> STOP / CAUTION", (0, 0, 255)
    elif sectors['LEFT']['near']:
        nav_msg, nav_col = "HAZARD ON LEFT -> BIAS RIGHT", (0, 200, 255)
    elif sectors['RIGHT']['near']:
        nav_msg, nav_col = "HAZARD ON RIGHT -> BIAS LEFT", (0, 200, 255)
    elif sectors['CENTER']['walkable'] > 300:
        nav_msg, nav_col = "PATH CLEAR - PROCEED FORWARD", (0, 255, 0)
    else:
        nav_msg, nav_col = "SCANNING FOR WALKABLE PATH", (0, 255, 255)

    draw_header(fusion_img, "SAFEPATH PIPELINE SPATIAL FUSION", 
                "Assistive AR Guidance HUD", 
                bg_color=(10, 25, 40), text_color=(0, 255, 255))

    # Bottom Decision Banner
    banner_y = H - 36
    cv2.rectangle(fusion_img, (0, banner_y), (W, H), (15, 15, 15), -1)
    cv2.line(fusion_img, (0, banner_y), (W, banner_y), nav_col, 2)
    cv2.putText(fusion_img, f"NAV >> {nav_msg}", (14, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, nav_col, 2, cv2.LINE_AA)

    fusion_out_path = os.path.join(out_dir, "pipeline_fusion.jpg")
    cv2.imwrite(fusion_out_path, fusion_img)
    print(f"      Saved: {fusion_out_path}")

    # --------------------------------------------------------------------------
    # COMPOSITE: 2x2 Quad Panel Overview
    # --------------------------------------------------------------------------
    top_row = np.hstack((yolo_img, depth_img))
    bottom_row = np.hstack((deeplab_img, fusion_img))
    quad_img = np.vstack((top_row, bottom_row))

    quad_out_path = os.path.join(out_dir, "quad_panel_overview.jpg")
    cv2.imwrite(quad_out_path, quad_img)
    print(f"      Saved: {quad_out_path}")

    print("\n" + "=" * 60)
    print("ALL MODEL OUTPUT IMAGES EXPORTED SUCCESSFULLY!")
    print(f"Destination: {out_dir}")
    print("Files created:")
    for fn in ["input_frame.jpg", "deeplabv3_mobilenet.jpg", "yolov8_hazards.jpg", 
               "depth_anything_v2.jpg", "pipeline_fusion.jpg", "quad_panel_overview.jpg"]:
        p = os.path.join(out_dir, fn)
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"  - {fn:<28} ({sz / 1024:.1f} KB)")
    print("=" * 60)

if __name__ == "__main__":
    main()

