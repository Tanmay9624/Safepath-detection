"""
SafePath AI: 4-Panel Multi-Modal Perception & Assistive Navigation Quad-Stream
==============================================================================
Displays four simultaneous video streams in a single 1280x720 window:
  - Panel 1 (Top-Left):     YOLOv8 Obstacle Detections (boxes, classes, confidences)
  - Panel 2 (Top-Right):    YOLOv8 + Depth Anything V2 (Full-Resolution Depth Colormap + Distances)
  - Panel 3 (Bottom-Left):  DeepLabV3 Walkable Path Segmentation & Safety Buffer
  - Panel 4 (Bottom-Right): Complete Assistive Navigation HUD (Fusion, Steering, TTS Banner)

Features:
  - Multi-threaded GPU pipeline (Thread 1: SafePath Seg, Thread 2: YOLOv8 + Depth V2).
  - Priority-driven SAPI Text-to-Speech audio engine with 1000 Hz earcon beeps.
  - Interactive snapshot capture (press 's') and clean shutdown (press 'q' or ESC).
"""

import os
import sys
import time
import queue
import argparse
import threading
import numpy as np
import cv2

# Ensure CUDA and cuDNN libraries from PyTorch are loaded by ONNX Runtime
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + ';' + os.environ.get('PATH', '')
except Exception:
    pass

import onnxruntime as ort
from audio_engine import PriorityAudioEngine

# COCO Class Label Mapping for Hazards
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 15: 'cat', 16: 'dog', 24: 'backpack', 26: 'handbag', 28: 'suitcase'
}

# Distinct BGR color palette for YOLO classes
CLASS_COLORS = {
    'person': (0, 255, 128),      # Bright mint green
    'car': (255, 180, 0),         # Cyan-blue
    'bicycle': (0, 215, 255),     # Gold
    'motorcycle': (0, 165, 255),  # Orange
    'bus': (255, 100, 0),         # Deep blue
    'truck': (200, 80, 0),        # Navy
    'traffic light': (0, 255, 255),
    'dog': (180, 105, 255),       # Pink
}

# ==============================================================================
# GPU Acceleration Configuration
# ==============================================================================
CUDA_OPTIONS = {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'do_copy_in_default_stream': True,
    'cudnn_conv_algo_search': 'HEURISTIC',
}

PROVIDERS = [('CUDAExecutionProvider', CUDA_OPTIONS), 'CPUExecutionProvider']

def get_session_options():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    return opts

# Precomputed Normalization Constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def resolve_path(filename):
    """Resolves file paths strictly within full_test or relative to current dir."""
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, filename)
    if os.path.exists(local_path):
        return local_path
    return filename

def prepare_tensor(image, w, h, apply_norm=True):
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if apply_norm:
        rgb = (rgb - MEAN) / STD
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

# ==============================================================================
# THREAD 2: Obstacle Detection & Depth Analysis Pipeline (YOLOv8 + Depth V2)
# ==============================================================================
def worker_obstacle_depth(in_q, out_q, depth_cadence=2, conf_thresh=0.40):
    opts = get_session_options()
    
    yolo_model_path = resolve_path("yolov8n_hazards.onnx")
    print(f"  [Thread-2] Loading YOLOv8 Hazards from: {yolo_model_path}")
    yolo_session = ort.InferenceSession(yolo_model_path, opts, providers=PROVIDERS)
    yolo_in_name = yolo_session.get_inputs()[0].name

    depth_model_path = resolve_path("depth_anything_v2_small.onnx")
    print(f"  [Thread-2] Loading Depth Anything V2 from: {depth_model_path}")
    depth_session = ort.InferenceSession(depth_model_path, opts, providers=PROVIDERS)
    depth_in_name = depth_session.get_inputs()[0].name
    
    print("  [Thread-2] Obstacle & Depth Worker initialized on GPU.")
    
    cached_depth_map = None

    while True:
        item = in_q.get()
        if item is None:
            break
        frame_id, frame = item

        # --- A. Step 1: Run YOLOv8 Hazards ---
        yolo_tensor = prepare_tensor(frame, 640, 480, apply_norm=False)
        yolo_out = yolo_session.run(None, {yolo_in_name: yolo_tensor})[0]

        preds = yolo_out[0].T  # (6300, 84)
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        conf_mask = confidences > conf_thresh
        valid_boxes = preds[conf_mask, :4]
        valid_confs = confidences[conf_mask]
        valid_classes = class_ids[conf_mask]

        yolo_boxes = []
        if len(valid_confs) > 0:
            boxes_xywh = []
            for b in valid_boxes:
                cx, cy, w, h = b
                boxes_xywh.append([int(cx - w / 2.0), int(cy - h / 2.0), int(w), int(h)])

            indices = cv2.dnn.NMSBoxes(boxes_xywh, valid_confs.tolist(), conf_thresh, 0.45)
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                for i in indices:
                    cx, cy, w, h = valid_boxes[i]
                    norm_x = (cx - w / 2.0) / 640.0
                    norm_y = (cy - h / 2.0) / 480.0
                    norm_w = w / 640.0
                    norm_h = h / 480.0
                    yolo_boxes.append((norm_x, norm_y, norm_w, norm_h, valid_classes[i], float(valid_confs[i])))

        # --- B. Step 2: Run Depth Anything V2 ---
        run_depth = (frame_id % depth_cadence == 0) or (cached_depth_map is None)
        if run_depth:
            depth_tensor = prepare_tensor(frame, 518, 518, apply_norm=True)
            depth_out = depth_session.run(None, {depth_in_name: depth_tensor})[0]
            raw_depth = depth_out[0]
            
            d_min, d_max = raw_depth.min(), raw_depth.max()
            if d_max > d_min:
                cached_depth_map = ((raw_depth - d_min) / (d_max - d_min) * 255.0).astype(np.float32)
            else:
                cached_depth_map = np.zeros_like(raw_depth, dtype=np.float32)

        out_q.put((frame_id, yolo_boxes, cached_depth_map, run_depth))

# ==============================================================================
# THREAD 1: Semantic SafePath Segmentation (Concurrent on Thread 1)
# ==============================================================================
def worker_deeplab(in_q, out_q):
    opts = get_session_options()
    seg_model_path = resolve_path("deeplabv3_mobilenet_safepath.onnx")
    print(f"  [Thread-1] Loading SafePath DeepLabV3 MobileNet from: {seg_model_path}")
    seg_session = ort.InferenceSession(seg_model_path, opts, providers=PROVIDERS)
    seg_in_name = seg_session.get_inputs()[0].name
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    print("  [Thread-1] SafePath Segmentation Worker initialized on GPU.")

    while True:
        item = in_q.get()
        if item is None:
            break
        frame_id, frame = item

        tensor = prepare_tensor(frame, 384, 256, apply_norm=True)
        out = seg_session.run(None, {seg_in_name: tensor})[0]
        mask = np.argmax(out, axis=1)[0].astype(np.uint8)

        # Class 1 = Walkable Path
        walkable = (mask == 1).astype(np.uint8) * 255
        # Physical safety buffer (shrink path away from edges and curbs)
        safe_mask = cv2.erode(walkable, erode_kernel, iterations=1)

        out_q.put((frame_id, walkable, safe_mask))

# ==============================================================================
# PANEL RENDERING HELPERS
# ==============================================================================
def draw_panel_header(panel, title, status_text="", bg_color=(20, 20, 20), text_color=(0, 255, 255)):
    """Draws a clean top status bar on each quadrant panel."""
    pw = panel.shape[1]
    cv2.rectangle(panel, (0, 0), (pw, 26), bg_color, -1)
    cv2.line(panel, (0, 26), (pw, 26), (60, 60, 60), 1)
    cv2.putText(panel, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_color, 1, cv2.LINE_AA)
    if status_text:
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(panel, status_text, (pw - tw - 10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

# ==============================================================================
# MAIN ENGINE: 4-Panel Quad View Stream
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="SafePath AI: 4-Panel Single-Window Quad Stream")
    parser.add_argument("--source", type=str, default="test_05_san_francisco_street.mp4",
                        help="Video file path or webcam index (default: test_05_san_francisco_street.mp4)")
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO confidence threshold (default: 0.40)")
    parser.add_argument("--depth-cadence", type=int, default=2, help="Depth inference cadence (default: 2)")
    parser.add_argument("--tts", dest="enable_tts", action="store_true", default=True, help="Enable Audio TTS (default)")
    parser.add_argument("--no-tts", dest="enable_tts", action="store_false", help="Disable Audio TTS")
    parser.add_argument("--clear-interval", type=float, default=15.0,
                        help="Reassurance delay interval for 'Path is clear' audio in seconds (default: 15.0, set 0 for alert-only mode)")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 for infinite)")
    parser.add_argument("--save-snapshot", type=str, default="", help="Save a sample snapshot image to specified path")
    args = parser.parse_args()

    print("==================================================================")
    print(" SafePath AI: 4-Panel Single-Window Quad View Pipeline")
    print("  - Panel 1 (Top-Left):     YOLOv8 Obstacle Detections")
    print("  - Panel 2 (Top-Right):    YOLOv8 + Depth V2 (Full-Res Colormap)")
    print("  - Panel 3 (Bottom-Left):  DeepLabV3 Walkable Path Segmentation")
    print("  - Panel 4 (Bottom-Right): Complete Assistive Navigation HUD")
    print(f"  - Audio TTS: {'ENABLED (SAPI SpVoice + Earcon)' if args.enable_tts else 'DISABLED'}")
    print("==================================================================")

    # Initialize Priority Audio Engine if enabled
    audio_engine = PriorityAudioEngine(default_cooldown=3.0) if args.enable_tts else None

    # Bounded communication queues
    obs_in_q = queue.Queue(maxsize=2)
    obs_out_q = queue.Queue(maxsize=2)
    seg_in_q = queue.Queue(maxsize=2)
    seg_out_q = queue.Queue(maxsize=2)

    # Launch worker threads
    t_obs = threading.Thread(target=worker_obstacle_depth, 
                             args=(obs_in_q, obs_out_q, args.depth_cadence, args.conf), 
                             daemon=True)
    t_seg = threading.Thread(target=worker_deeplab, 
                             args=(seg_in_q, seg_out_q), 
                             daemon=True)
    t_obs.start()
    t_seg.start()

    # Open video source
    video_path = int(args.source) if args.source.isdigit() else resolve_path(args.source)
    if not isinstance(video_path, int) and not os.path.exists(str(video_path)):
        video_path = resolve_path("test_01_urban_crowd.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_path}")
        return

    print(f"\n[START] Streaming video: {video_path}")
    print(">> Single 1280x720 Quad-View Window Active.")
    print(">> Controls: Press 'q' or 'ESC' to exit | Press 's' to save snapshot.\n")

    win_name = "SafePath AI - 4-Panel Multi-Modal Perception & Navigation HUD"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_name, 40, 30)

    PW, PH = 640, 360  # Individual panel dimensions
    frame_count = 0
    smooth_fps = 0.0
    blocked_frame_count = 0
    clear_frame_count = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            # Seamless loop for video files
            if not isinstance(video_path, int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame = cv2.resize(raw_frame, (PW, PH), interpolation=cv2.INTER_LINEAR)
        fid = frame_count
        frame_count += 1

        t0 = time.perf_counter()

        # 1. Dispatch frame to worker threads
        if obs_in_q.full():
            try: obs_in_q.get_nowait()
            except queue.Empty: pass
        obs_in_q.put((fid, frame))

        if seg_in_q.full():
            try: seg_in_q.get_nowait()
            except queue.Empty: pass
        seg_in_q.put((fid, frame))

        # 2. Collect synchronized results
        obs_fid, yolo_boxes, depth_map, depth_computed = obs_out_q.get()
        seg_fid, raw_walkable, safe_mask = seg_out_q.get()

        # =============================================================
        # PANEL 1: YOLOv8 OBSTACLE DETECTIONS
        # =============================================================
        p1 = frame.copy()
        for box_item in yolo_boxes:
            norm_x, norm_y, norm_w, norm_h, cls_id, conf = box_item
            bx = max(0, int(norm_x * PW))
            by = max(0, int(norm_y * PH))
            bx2 = min(PW, int((norm_x + norm_w) * PW))
            by2 = min(PH, int((norm_y + norm_h) * PH))
            bw = max(0, bx2 - bx)
            bh = max(0, by2 - by)
            if bw <= 0 or bh <= 0:
                continue
            cname = COCO_CLASSES.get(int(cls_id), f"ID:{cls_id}")
            bcolor = CLASS_COLORS.get(cname, (0, 215, 255))

            cv2.rectangle(p1, (bx, by), (bx + bw, by + bh), bcolor, 2)
            label = f"{cname} {int(conf * 100)}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(p1, (bx, max(0, by - 18)), (bx + lw + 6, by), bcolor, -1)
            cv2.putText(p1, label, (bx + 3, max(12, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 10, 10), 1, cv2.LINE_AA)

        draw_panel_header(p1, "[1] YOLOv8 OBSTACLE DETECTIONS", 
                          f"Detections: {len(yolo_boxes)}", 
                          bg_color=(25, 25, 25), text_color=(0, 255, 255))

        # =============================================================
        # PANEL 2: YOLOv8 + DEPTH ANYTHING V2 (FULL RESOLUTION COLORMAP)
        # =============================================================
        p2 = np.zeros((PH, PW, 3), dtype=np.uint8)
        depth_mode_str = "RT (518x518)" if depth_computed else "CACHED"

        if depth_map is not None:
            # Resize raw depth map (518x518) to FULL panel resolution (640x360)
            depth_u8 = np.clip(depth_map, 0, 255).astype(np.uint8)
            depth_high_res = cv2.resize(depth_u8, (PW, PH), interpolation=cv2.INTER_CUBIC)
            p2 = cv2.applyColorMap(depth_high_res, cv2.COLORMAP_INFERNO)

            # Overlay YOLO bounding boxes with high-contrast metric distance tags
            for box_item in yolo_boxes:
                norm_x, norm_y, norm_w, norm_h, cls_id, _ = box_item
                bx = max(0, int(norm_x * PW))
                by = max(0, int(norm_y * PH))
                bx2 = min(PW, int((norm_x + norm_w) * PW))
                by2 = min(PH, int((norm_y + norm_h) * PH))
                bw = max(0, bx2 - bx)
                bh = max(0, by2 - by)
                if bw <= 0 or bh <= 0:
                    continue
                cname = COCO_CLASSES.get(int(cls_id), f"ID:{cls_id}")

                # Sample metric depth within bounding box on high-res map
                roi_depth = depth_high_res[by:by2, bx:bx2]
                md = float(np.mean(roi_depth)) if roi_depth.size > 0 else 0.0
                est_d = max(0.5, round((255.0 - md) / 255.0 * 4.5 + 0.5, 1)) if md > 0 else 3.5

                # High-contrast cyan/white bounding box over colormap
                cv2.rectangle(p2, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
                tag = f"{cname} | {est_d:.1f}m"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.rectangle(p2, (bx, max(0, by - 18)), (bx + tw + 6, by), (20, 20, 20), -1)
                cv2.putText(p2, tag, (bx + 3, max(12, by - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

        draw_panel_header(p2, "[2] DEPTH ANYTHING V2 + YOLOv8", 
                          f"Mode: {depth_mode_str}", 
                          bg_color=(20, 10, 30), text_color=(255, 180, 0))

        # =============================================================
        # PANEL 3: DEEPLABV3 WALKABLE PATH SEGMENTATION
        # =============================================================
        p3 = frame.copy()
        
        # Upsample masks to panel size (640x360)
        full_walkable = cv2.resize(raw_walkable, (PW, PH), interpolation=cv2.INTER_NEAREST)
        full_safe = cv2.resize(safe_mask, (PW, PH), interpolation=cv2.INTER_NEAREST)

        # A. Walkable area highlight (Cyan overlay)
        path_overlay = p3.copy()
        path_overlay[full_safe > 0] = [200, 220, 0]  # Bright Cyan-Green
        p3 = cv2.addWeighted(path_overlay, 0.40, p3, 0.60, 0.0)

        # B. Safety buffer boundary outline (7x7 erosion contour)
        contours, _ = cv2.findContours(full_safe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p3, contours, -1, (255, 255, 255), 2)

        # C. Telemetry: Walkable area coverage %
        total_pixels = PW * PH
        walkable_pixels = cv2.countNonZero(full_safe)
        walkable_ratio = (walkable_pixels / float(total_pixels)) * 100.0

        # Draw mini sector bars at bottom of Panel 3
        lower_safe = safe_mask[128:256, :]
        sec_l = int(np.count_nonzero(lower_safe[:, 0:128]))
        sec_c = int(np.count_nonzero(lower_safe[:, 128:256]))
        sec_r = int(np.count_nonzero(lower_safe[:, 256:384]))

        stat_str = f"Walkable Area: {walkable_ratio:.1f}% | L:{sec_l} C:{sec_c} R:{sec_r}px"
        cv2.rectangle(p3, (10, PH - 24), (PW - 10, PH - 6), (15, 15, 15), -1)
        cv2.putText(p3, stat_str, (16, PH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1, cv2.LINE_AA)

        draw_panel_header(p3, "[3] DEEPLABV3 SAFEPATH SEGMENTATION", 
                          f"Coverage: {walkable_ratio:.1f}%", 
                          bg_color=(15, 25, 20), text_color=(0, 255, 150))

        # =============================================================
        # PANEL 4: COMPLETE ASSISTIVE NAVIGATION HUD (FULL FUSION)
        # =============================================================
        p4 = frame.copy()

        # A. Walkable Path AR Overlay (Green)
        green_overlay = p4.copy()
        green_overlay[full_safe > 0] = [0, 255, 0]
        p4 = cv2.addWeighted(green_overlay, 0.35, p4, 0.65, 0.0)

        # B. Sector-Based Assistive Navigation Analysis
        sectors = {'LEFT': {'walkable': sec_l, 'hazards': 0, 'near': False},
                   'CENTER': {'walkable': sec_c, 'hazards': 0, 'near': False},
                   'RIGHT': {'walkable': sec_r, 'hazards': 0, 'near': False}}

        hazard_on_path = False
        closest_on_path_dist = 999.0
        closest_on_path_name = "hazard"
        closest_on_path_sector = "CENTER"

        # C. Spatial Overlap Math, Depth Sampling & Hazard Classification
        for box_item in yolo_boxes:
            norm_x, norm_y, norm_w, norm_h, cls_id, _ = box_item
            x1_dl = max(0, int(norm_x * 384))
            y1_dl = max(0, int(norm_y * 256))
            x2_dl = min(384, int((norm_x + norm_w) * 384))
            y2_dl = min(256, int((norm_y + norm_h) * 256))
            dl_w = max(0, x2_dl - x1_dl)
            dl_h = max(0, y2_dl - y1_dl)

            if dl_w <= 0 or dl_h <= 0:
                continue

            path_roi = safe_mask[y1_dl:y2_dl, x1_dl:x2_dl]
            overlap = cv2.countNonZero(path_roi) / float(dl_w * dl_h)

            orig_x = max(0, int(norm_x * PW))
            orig_y = max(0, int(norm_y * PH))
            orig_x2 = min(PW, int((norm_x + norm_w) * PW))
            orig_y2 = min(PH, int((norm_y + norm_h) * PH))
            orig_box_w = max(0, orig_x2 - orig_x)
            orig_box_h = max(0, orig_y2 - orig_y)
            if orig_box_w <= 0 or orig_box_h <= 0:
                continue
            cname = COCO_CLASSES.get(int(cls_id), f"ID:{cls_id}")

            md = 0.0
            if depth_map is not None:
                x1_md = max(0, int(norm_x * 518))
                y1_md = max(0, int(norm_y * 518))
                x2_md = min(518, int((norm_x + norm_w) * 518))
                y2_md = min(518, int((norm_y + norm_h) * 518))
                if x2_md > x1_md and y2_md > y1_md:
                    md = float(np.mean(depth_map[y1_md:y2_md, x1_md:x2_md]))

            est_d = max(0.5, round((255.0 - md) / 255.0 * 4.5 + 0.5, 1)) if md > 0 else 3.5
            is_near = (est_d <= 1.8) or (md > 175.0)

            cx = norm_x + norm_w / 2.0
            sector_key = 'LEFT' if cx < 0.35 else ('RIGHT' if cx > 0.65 else 'CENTER')

            # Flank proximity tracking
            if is_near:
                sectors[sector_key]['near'] = True

            if overlap > 0.15:
                # Hazard encroaching on walkable path
                hazard_on_path = True
                box_color = (0, 0, 255)  # Red: Blocking Walkable Path
                label = f"HAZARD: {cname.upper()} ({est_d:.1f}m)"
                sectors[sector_key]['hazards'] += 1
                if est_d < closest_on_path_dist:
                    closest_on_path_dist = est_d
                    closest_on_path_name = cname
                    closest_on_path_sector = sector_key
            else:
                # Obstacle safely off-path
                box_color = (0, 255, 255)  # Yellow: Off-path
                label = f"{cname} (Off-Path)"

            cv2.rectangle(p4, (orig_x, orig_y), (orig_x + orig_box_w, orig_y + orig_box_h), box_color, 2)
            cv2.putText(p4, label, (orig_x, max(36, orig_y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_color, 1, cv2.LINE_AA)

        # D. Assistive Navigation Steering Decision & Robust State Debounce
        c_blocked = sectors['CENTER']['near'] or sectors['CENTER']['hazards'] > 0
        l_walk = sectors['LEFT']['walkable'] > 200 and not sectors['LEFT']['near']
        r_walk = sectors['RIGHT']['walkable'] > 200 and not sectors['RIGHT']['near']

        if c_blocked:
            blocked_frame_count += 1
            clear_frame_count = 0
        else:
            clear_frame_count += 1

        # Priority 1: Center walking path is obstructed
        if c_blocked:
            if closest_on_path_dist <= 1.5 and audio_engine:
                loc_str = "directly ahead" if closest_on_path_sector == "CENTER" else f"on your {closest_on_path_sector.lower()}"
                audio_engine.add_alert({
                    "object_id": f"hazard_{closest_on_path_name}_{closest_on_path_sector}",
                    "priority": 1,
                    "message": f"Stop, {closest_on_path_name} {loc_str}",
                    "hazard_type": "obstacle",
                    "distance_m": closest_on_path_dist
                })

            if l_walk and not r_walk:
                nav_text = "NAV: HAZARD IN CENTER -> VEER LEFT"
                nav_color = (0, 255, 255)
                if audio_engine:
                    audio_engine.add_alert({
                        "object_id": "nav_veer_left",
                        "priority": 2,
                        "message": "Obstacle ahead. Veer left onto safe path.",
                        "hazard_type": "navigation",
                        "distance_m": closest_on_path_dist if closest_on_path_dist < 900 else 2.5
                    })
            elif r_walk and not l_walk:
                nav_text = "NAV: HAZARD IN CENTER -> VEER RIGHT"
                nav_color = (0, 255, 255)
                if audio_engine:
                    audio_engine.add_alert({
                        "object_id": "nav_veer_right",
                        "priority": 2,
                        "message": "Obstacle ahead. Veer right onto safe path.",
                        "hazard_type": "navigation",
                        "distance_m": closest_on_path_dist if closest_on_path_dist < 900 else 2.5
                    })
            elif l_walk and r_walk:
                nav_text = "NAV: HAZARD IN CENTER -> VEER RIGHT"
                nav_color = (0, 255, 255)
                if audio_engine:
                    audio_engine.add_alert({
                        "object_id": "nav_veer_right",
                        "priority": 2,
                        "message": "Obstacle ahead. Veer right onto safe path.",
                        "hazard_type": "navigation",
                        "distance_m": closest_on_path_dist if closest_on_path_dist < 900 else 2.5
                    })
            else:
                nav_text = "NAV: CROWD BLOCKED -> STOP / CAUTION"
                nav_color = (0, 0, 255)
                if audio_engine:
                    audio_engine.add_alert({
                        "object_id": "nav_stop",
                        "priority": 1,
                        "message": "Caution, path blocked. Please stop.",
                        "hazard_type": "navigation",
                        "distance_m": closest_on_path_dist if closest_on_path_dist < 900 else 1.0
                    })

        # Priority 2: Center is clear, but flank hazard is dangerously near
        elif sectors['LEFT']['near']:
            nav_text = "NAV: HAZARD ON LEFT -> BIAS RIGHT"
            nav_color = (0, 255, 255)
            if audio_engine:
                audio_engine.add_alert({
                    "object_id": "nav_bias_right",
                    "priority": 2,
                    "message": "Hazard on your left. Bias right.",
                    "hazard_type": "navigation",
                    "distance_m": 1.5
                })

        elif sectors['RIGHT']['near']:
            nav_text = "NAV: HAZARD ON RIGHT -> BIAS LEFT"
            nav_color = (0, 255, 255)
            if audio_engine:
                audio_engine.add_alert({
                    "object_id": "nav_bias_left",
                    "priority": 2,
                    "message": "Hazard on your right. Bias left.",
                    "hazard_type": "navigation",
                    "distance_m": 1.5
                })

        # Priority 3: Path is fully open and clear ahead
        elif sectors['CENTER']['walkable'] > 300:
            nav_text = "NAV: PATH CLEAR - PROCEED FORWARD"
            nav_color = (0, 255, 0)

            # If recovering from a sustained blockage (>=5 frames), allow immediate reassurance
            if audio_engine and args.clear_interval > 0:
                if blocked_frame_count >= 5 and clear_frame_count >= 3:
                    audio_engine.reset_alert_state("nav_forward")
                    blocked_frame_count = 0

                audio_engine.add_alert({
                    "object_id": "nav_forward",
                    "priority": 3,
                    "message": "Path is clear.",
                    "hazard_type": "navigation",
                    "distance_m": 5.0,
                    "cooldown": args.clear_interval
                })

        # Priority 4: No obstacles, but walkable path is lost
        else:
            nav_text = "NAV: SCANNING FOR WALKABLE PATH"
            nav_color = (200, 200, 200)
            if audio_engine:
                audio_engine.add_alert({
                    "object_id": "nav_lost_path",
                    "priority": 2,
                    "message": "Caution, no walkable path detected. Please stop.",
                    "hazard_type": "navigation",
                    "distance_m": 0.0,
                    "cooldown": 4.0
                })

        # E. Telemetry & FPS
        dt = time.perf_counter() - t0
        curr_fps = 1.0 / max(dt, 1e-5)
        smooth_fps = curr_fps if smooth_fps == 0.0 else (0.85 * smooth_fps + 0.15 * curr_fps)

        # F. Assistive HUD Overlay Box in Panel 4
        hud_bg = p4.copy()
        cv2.rectangle(hud_bg, (10, 32), (PW - 10, 102), (15, 15, 15), -1)
        p4 = cv2.addWeighted(hud_bg, 0.70, p4, 0.30, 0.0)

        cv2.putText(p4, nav_text, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, nav_color, 2, cv2.LINE_AA)
        telemetry = f"GPU: {smooth_fps:.1f} FPS | Center: {sectors['CENTER']['walkable']}px | Depth: {depth_mode_str}"
        cv2.putText(p4, telemetry, (18, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        last_msg = audio_engine.last_spoken_message if audio_engine else "TTS Audio Disabled"
        cv2.putText(p4, f"TTS: \"{last_msg}\"", (18, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 240, 255), 1, cv2.LINE_AA)

        # Corridor guide markers
        cv2.line(p4, (int(0.35 * PW), PH - 20), (int(0.35 * PW), PH), (255, 255, 255), 1)
        cv2.line(p4, (int(0.65 * PW), PH - 20), (int(0.65 * PW), PH), (255, 255, 255), 1)

        draw_panel_header(p4, "[4] COMPLETE ASSISTIVE NAVIGATION HUD", 
                          f"FPS: {smooth_fps:.1f}", 
                          bg_color=(25, 15, 15), text_color=(0, 255, 0))

        # =============================================================
        # 3. CONSTRUCT 2x2 QUAD-VIEW CANVAS (1280 x 720)
        # =============================================================
        row_top = np.hstack([p1, p2])
        row_bot = np.hstack([p3, p4])
        quad_canvas = np.vstack([row_top, row_bot])

        # Draw high-contrast grid dividing lines between panels
        cv2.line(quad_canvas, (PW, 0), (PW, PH * 2), (50, 50, 50), 2)
        cv2.line(quad_canvas, (0, PH), (PW * 2, PH), (50, 50, 50), 2)

        # Save snapshot if requested
        if args.save_snapshot and fid == 15:
            cv2.imwrite(args.save_snapshot, quad_canvas)
            print(f"  [SNAPSHOT] Saved quad-view frame to: {args.save_snapshot}")

        # Display composite 4-panel stream
        cv2.imshow(win_name, quad_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            snap_path = f"quad_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_path, quad_canvas)
            print(f"  [SAVED] Manual snapshot: {snap_path}")

        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[STOP] Quad-View pipeline stopped cleanly.")

if __name__ == "__main__":
    main()

