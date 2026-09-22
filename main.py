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
    """
    Resolves file paths strictly within the full_test folder:
    1. Current working directory (.)
    2. Directory containing this script (full_test/)
    """
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, filename)
    if os.path.exists(local_path):
        return local_path
    return filename

def resolve_video_path(src):
    """
    Resolves video file path:
    - Checks given path directly (absolute or relative to current working directory).
    - Checks relative to the script's directory.
    Returns canonical path if the file exists on disk, otherwise None.
    """
    if not src:
        return None
    src_str = str(src).strip().strip('"').strip("'")
    
    # 1. Direct path check (current working directory or absolute)
    if os.path.isfile(src_str):
        return os.path.abspath(src_str)
    
    # 2. Check relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, src_str)
    if os.path.isfile(script_path):
        return os.path.abspath(script_path)
        
    return None

def resolve_stream_source(ip_arg=None, source_arg=None, default_cam=0):
    """
    Resolves camera/video input stream source:
    1. If --ip is provided:
       - Auto-normalizes IP webcam URLs (e.g. 192.168.1.15:8080 -> http://192.168.1.15:8080/video).
    2. If --video / --source / positional argument is provided:
       - If it's a valid video file path, resolves to file path.
       - If numeric string (e.g. '0', '1'), converts to integer camera index.
       - If URL or IP address, formats network stream.
       - Otherwise, treats as path directly.
    3. Default:
       - Falls back to local machine webcam (default_cam = 0).
    """
    if ip_arg:
        url = ip_arg.strip()
        if not (url.startswith("http://") or url.startswith("https://") or url.startswith("rtsp://")):
            if "/" not in url:
                url = f"http://{url}/video"
            else:
                url = f"http://{url}"
        return url, f"IP Webcam ({url})"

    if source_arg is not None:
        src = str(source_arg).strip().strip('"').strip("'")
        
        # 1. Check if it's an existing video file path
        resolved_file = resolve_video_path(src)
        if resolved_file:
            return resolved_file, f"Video File ({os.path.basename(resolved_file)})"

        # 2. Check if camera device index (e.g. 0, 1)
        if src.isdigit():
            return int(src), f"Webcam (Device Index {src})"

        # 3. Check if network stream URL
        if src.startswith("http://") or src.startswith("https://") or src.startswith("rtsp://"):
            return src, f"Network Stream ({src})"

        # 4. Check if raw IP webcam format entered directly (e.g. "192.168.1.15:8080")
        if ":" in src and "." in src and not os.path.exists(src):
            url = f"http://{src}/video" if "/" not in src else f"http://{src}"
            return url, f"IP Webcam ({url})"

        # 5. Non-existent path or custom source
        return src, f"Video Path ({src})"

    # Default fallback: Local machine webcam
    return default_cam, f"Local Machine Webcam (Device Index {default_cam})"


def prepare_tensor(image, w, h, apply_norm=True):
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if apply_norm:
        rgb = (rgb - MEAN) / STD
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

# ==============================================================================
# THREAD 2: Obstacle Detection & Depth Analysis Pipeline (YOLOv8 + Depth V2)
# ==============================================================================
def worker_obstacle_depth(in_q, out_q, depth_cadence=2):
    """
    Runs YOLOv8 Hazard Detection AND Depth Anything V2 sequentially on Thread 2.
    - YOLO detects obstacle bounding boxes.
    - Depth Anything generates the 3D depth map (with cadence caching).
    """
    opts = get_session_options()
    
    # 1. Load YOLOv8 Hazards (local to full_test)
    yolo_model_path = resolve_path("yolov8n_hazards.onnx")
    print(f"  [Thread-2] Loading YOLOv8 Hazards from: {yolo_model_path}")
    yolo_session = ort.InferenceSession(yolo_model_path, opts, providers=PROVIDERS)
    yolo_in_name = yolo_session.get_inputs()[0].name

    # 2. Load Depth Anything V2 (local to full_test)
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

        conf_mask = confidences > 0.45
        valid_boxes = preds[conf_mask, :4]
        valid_confs = confidences[conf_mask]
        valid_classes = class_ids[conf_mask]

        yolo_boxes = []
        if len(valid_confs) > 0:
            boxes_xywh = []
            for b in valid_boxes:
                cx, cy, w, h = b
                boxes_xywh.append([int(cx - w / 2.0), int(cy - h / 2.0), int(w), int(h)])

            indices = cv2.dnn.NMSBoxes(boxes_xywh, valid_confs.tolist(), 0.45, 0.45)
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                for i in indices:
                    cx, cy, w, h = valid_boxes[i]
                    norm_x = (cx - w / 2.0) / 640.0
                    norm_y = (cy - h / 2.0) / 480.0
                    norm_w = w / 640.0
                    norm_h = h / 480.0
                    yolo_boxes.append((norm_x, norm_y, norm_w, norm_h, valid_classes[i]))

        # --- B. Step 2: Run Depth Anything V2 (Sequential in Same Thread) ---
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
    """
    Runs DeepLabV3 MobileNet Walkable Path Segmentation and safety erosion in parallel.
    """
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

        out_q.put((frame_id, safe_mask))

# ==============================================================================
# MAIN ENGINE: Frame Ingest, Thread Synchronization, Dual Visuals & TTS Audio
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="SafePath AI: Edge Assistive Navigation Engine (Headless)")
    parser.add_argument("source_pos", nargs="?", default=None, 
                        help="Video file path or camera index (positional, e.g. path/to/video.mp4 or 0)")
    parser.add_argument("--ip", type=str, default=None, 
                        help="IP Webcam address or stream URL (e.g. 192.168.1.15:8080 or http://192.168.1.15:8080/video)")
    parser.add_argument("--source", type=str, default=None, 
                        help="Input video file path, camera index, or network stream URL")
    parser.add_argument("--video", type=str, default=None, 
                        help="Path to input video file (e.g. --video path/to/video.mp4)")
    parser.add_argument("--cam-index", type=int, default=0, 
                        help="Local machine webcam device index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.45, 
                        help="YOLO hazard confidence threshold (default: 0.45)")
    parser.add_argument("--depth-cadence", type=int, default=2, 
                        help="Depth inference cadence (default: 2)")
    parser.add_argument("--clear-interval", type=float, default=15.0,
                        help="Reassurance delay interval for 'Path is clear' audio in seconds (default: 15.0, set 0 for alert-only mode)")
    parser.add_argument("--max-frames", type=int, default=0, 
                        help="Exit after N frames (0 for continuous execution)")
    parser.add_argument("--gui", action="store_true", default=False, 
                        help="Enable visual display windows (default: False, completely headless)")
    parser.add_argument("--tts", dest="enable_tts", action="store_true", default=True, 
                        help="Enable Audio TTS (default: True)")
    parser.add_argument("--no-tts", dest="enable_tts", action="store_false", 
                        help="Disable Audio TTS")
    args = parser.parse_args()

    # Determine input video source (IP Webcam, Local Camera default, or Video)
    video_source, source_desc = resolve_stream_source(
        ip_arg=args.ip,
        source_arg=args.video or args.source or args.source_pos,
        default_cam=args.cam_index
    )

    print("==================================================================")
    print(" SafePath AI: Edge Assistive Navigation Engine")
    print(f"  - Mode:       {'GUI VISUAL STREAM (Dual Windows)' if args.gui else 'HEADLESS (No GUI, Low-Latency Edge)'}")
    print(f"  - Source:     {source_desc}")
    print("  - Thread 1:   DeepLabV3 MobileNet SafePath (Concurrent)")
    print("  - Thread 2:   YOLOv8 Hazards + Depth Anything V2 (Sequential)")
    print(f"  - Audio TTS:  {'ENABLED (SAPI SpVoice + Earcon)' if args.enable_tts else 'DISABLED'}")
    print(f"  - Heartbeat:  {args.clear_interval:.1f}s ('Path is clear' interval)")
    print("==================================================================")

    # Initialize Priority TTS Audio Engine
    audio_engine = PriorityAudioEngine(default_cooldown=3.0) if args.enable_tts else None

    # Bounded communication queues (size=2 to prevent frame latency buildup)
    obs_in_q = queue.Queue(maxsize=2)
    obs_out_q = queue.Queue(maxsize=2)
    seg_in_q = queue.Queue(maxsize=2)
    seg_out_q = queue.Queue(maxsize=2)

    DEPTH_CADENCE = args.depth_cadence

    # Launch worker threads
    t_obs = threading.Thread(target=worker_obstacle_depth, 
                             args=(obs_in_q, obs_out_q, DEPTH_CADENCE), 
                             daemon=True)
    t_seg = threading.Thread(target=worker_deeplab, 
                             args=(seg_in_q, seg_out_q), 
                             daemon=True)
    t_obs.start()
    t_seg.start()

    print(f"\n[INIT] Connecting to camera feed: {source_desc}...")
    if isinstance(video_source, int):
        # Local machine webcam: use DirectShow on Windows for instant startup without MSMF lag
        if sys.platform == "win32":
            cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(video_source)
        if str(video_source).startswith("http") or str(video_source).startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open video source: {video_source}")
        if isinstance(video_source, int):
            print("  [HINT] Could not open local webcam. Ensure no other application is using it.")
            print("  [HINT] To stream from your smartphone via IP Webcam app: python main.py --ip <phone_ip:port>")
            print("  [HINT] To run with a video file: python main.py <path/to/video.mp4> or python main.py --video <path>")
        else:
            if not os.path.exists(str(video_source)) and not str(video_source).startswith("http") and not str(video_source).startswith("rtsp"):
                print(f"  [HINT] File not found at path: '{video_source}'. Please verify the path.")
            else:
                print("  [HINT] Verify WiFi connection, IP address, or video file format.")
        if audio_engine:
            audio_engine.add_system_error("Video source failed to open. Please check connection or file path.")
            time.sleep(2.0)
        return

    print(f"[READY] Camera feed active. Perception pipeline running.")
    if args.gui:
        print(">> Dual GUI Mode Active. Press 'q' or 'ESC' to stop.\n")
    else:
        print(">> Headless Mode Active (No GUI). Real-time audio alerts enabled.")
        print(">> Press Ctrl+C in terminal to stop.\n")

    if args.gui:
        win_perception = "Stream 1: Thread 2 Perception (YOLOv8 + Depth V2)"
        win_stream = "Stream 2: Complete Assistive Navigation Stream"
        cv2.namedWindow(win_perception, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(win_stream, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(win_perception, 30, 40)
        cv2.moveWindow(win_stream, 680, 40)

    frame_count = 0
    smooth_fps = 0.0
    blocked_frame_count = 0
    clear_frame_count = 0
    prev_nav_text = ""

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                if isinstance(video_source, int) or str(video_source).startswith("http") or str(video_source).startswith("rtsp"):
                    # Live camera stream glitch or temporary network delay
                    time.sleep(0.05)
                    continue
                else:
                    # Video file reached end
                    if args.max_frames > 0:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # Downscale raw frame once to 640x360 for high-speed queue handling & processing
            frame = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_LINEAR)
            orig_h, orig_w = frame.shape[:2]
            fid = frame_count
            frame_count += 1

            t0 = time.perf_counter()

            # 1. Dispatch frame concurrently to both worker pipelines
            if obs_in_q.full():
                try: obs_in_q.get_nowait()
                except queue.Empty: pass
            obs_in_q.put((fid, frame))

            if seg_in_q.full():
                try: seg_in_q.get_nowait()
                except queue.Empty: pass
            seg_in_q.put((fid, frame))

            # 2. Collect synchronized results from Thread 1 and Thread 2
            obs_fid, yolo_boxes, depth_map, depth_computed = obs_out_q.get()
            seg_fid, dl_mask = seg_out_q.get()

            # 3. Sector & Spatial Fusion Analysis (Always runs)
            sectors = {'LEFT': {'walkable': 0, 'hazards': 0, 'near': False},
                       'CENTER': {'walkable': 0, 'hazards': 0, 'near': False},
                       'RIGHT': {'walkable': 0, 'hazards': 0, 'near': False}}

            lower_dl = dl_mask[128:256, :]
            sectors['LEFT']['walkable'] = int(np.count_nonzero(lower_dl[:, 0:128]))
            sectors['CENTER']['walkable'] = int(np.count_nonzero(lower_dl[:, 128:256]))
            sectors['RIGHT']['walkable'] = int(np.count_nonzero(lower_dl[:, 256:384]))

            hazard_on_path = False
            closest_on_path_dist = 999.0
            closest_on_path_name = "hazard"
            closest_on_path_sector = "CENTER"

            for (norm_x, norm_y, norm_w, norm_h, cls_id) in yolo_boxes:
                # Proper (x1, y1, x2, y2) corner coordinate clipping for DeepLab mask (384x256)
                x1_dl = max(0, int(norm_x * 384))
                y1_dl = max(0, int(norm_y * 256))
                x2_dl = min(384, int((norm_x + norm_w) * 384))
                y2_dl = min(256, int((norm_y + norm_h) * 256))
                dl_w = max(0, x2_dl - x1_dl)
                dl_h = max(0, y2_dl - y1_dl)

                if dl_w <= 0 or dl_h <= 0:
                    continue

                path_roi = dl_mask[y1_dl:y2_dl, x1_dl:x2_dl]
                overlap = cv2.countNonZero(path_roi) / float(dl_w * dl_h)
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

                # Critical proximity tracking: flank obstacles near the user register as near hazards
                if is_near:
                    sectors[sector_key]['near'] = True

                if overlap > 0.15:
                    hazard_on_path = True
                    sectors[sector_key]['hazards'] += 1
                    if est_d < closest_on_path_dist:
                        closest_on_path_dist = est_d
                        closest_on_path_name = cname
                        closest_on_path_sector = sector_key

            # 4. Steering Decisions & Priority Audio Dispatch
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
                    nav_color = (0, 255, 255)  # Yellow
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
                    nav_color = (0, 255, 255)  # Yellow
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
                    nav_color = (0, 255, 255)  # Yellow
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
                    nav_color = (0, 0, 255)  # Red
                    if audio_engine:
                        audio_engine.add_alert({
                            "object_id": "nav_stop",
                            "priority": 1,
                            "message": "Caution, path blocked. Please stop.",
                            "hazard_type": "navigation",
                            "distance_m": closest_on_path_dist if closest_on_path_dist < 900 else 1.0
                        })

            # Priority 2: Center is clear, but flank obstacle is dangerously close
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

            # Priority 3: Path is fully open and safe ahead
            elif sectors['CENTER']['walkable'] > 300:
                nav_text = "NAV: PATH CLEAR - PROCEED FORWARD"
                nav_color = (0, 255, 0)  # Green

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

            # Priority 4: No obstacles, but walkable path is completely lost (e.g. edge of sidewalk, drop-off, ditch)
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

            # E. FPS Calculation
            dt = time.perf_counter() - t0
            curr_fps = 1.0 / max(dt, 1e-5)
            smooth_fps = curr_fps if smooth_fps == 0.0 else (0.85 * smooth_fps + 0.15 * curr_fps)

            # 5. Output Telemetry: Headless Console Dashboard vs GUI Windows
            if not args.gui:
                # Live Headless Console Output (reported periodically & on navigation change)
                if frame_count % 30 == 0 or nav_text != prev_nav_text:
                    haz_summary = f"{closest_on_path_name} ({closest_on_path_dist:.1f}m)" if hazard_on_path else "None"
                    print(f"[STATUS | Frame {frame_count:05d}] {smooth_fps:.1f} FPS | {nav_text} | Walkable: {sectors['CENTER']['walkable']}px | Nearest: {haz_summary}")
                    prev_nav_text = nav_text
            else:
                # GUI Visual Mode: Render perception and stream HUD windows
                perception_frame = frame.copy()
                if depth_map is not None:
                    depth_u8 = np.clip(depth_map, 0, 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
                    pip_w, pip_h = 160, 90
                    depth_pip = cv2.resize(depth_colored, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
                    x_off = orig_w - pip_w - 10
                    y_off = 32
                    perception_frame[y_off:y_off+pip_h, x_off:x_off+pip_w] = depth_pip
                    cv2.rectangle(perception_frame, (x_off - 1, y_off - 1), (x_off + pip_w + 1, y_off + pip_h + 1), (255, 255, 255), 1)
                    depth_mode_str = "RT" if depth_computed else "CACHE"
                    cv2.putText(perception_frame, f"DEPTH V2 [{depth_mode_str}]", 
                                (x_off + 4, y_off - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)

                for (norm_x, norm_y, norm_w, norm_h, cls_id) in yolo_boxes:
                    bx = max(0, int(norm_x * orig_w))
                    by = max(0, int(norm_y * orig_h))
                    bx2 = min(orig_w, int((norm_x + norm_w) * orig_w))
                    by2 = min(orig_h, int((norm_y + norm_h) * orig_h))
                    bw = max(0, bx2 - bx)
                    bh = max(0, by2 - by)
                    if bw <= 0 or bh <= 0:
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
                    cv2.rectangle(perception_frame, (bx, by), (bx + bw, by + bh), (0, 215, 255), 2)
                    cv2.putText(perception_frame, f"{cname} | {est_d:.1f}m", (bx, max(18, by - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1)

                cv2.rectangle(perception_frame, (0, 0), (orig_w, 26), (20, 20, 20), -1)
                cv2.putText(perception_frame, f"THREAD 2 PERCEPTION: YOLOv8 Hazards + Depth V2 | Detections: {len(yolo_boxes)}", 
                            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

                stream_frame = frame.copy()
                full_mask = cv2.resize(dl_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                green_overlay = stream_frame.copy()
                green_overlay[full_mask > 0] = [0, 255, 0]
                stream_frame = cv2.addWeighted(green_overlay, 0.35, stream_frame, 0.65, 0.0)

                for (norm_x, norm_y, norm_w, norm_h, cls_id) in yolo_boxes:
                    orig_x = max(0, int(norm_x * orig_w))
                    orig_y = max(0, int(norm_y * orig_h))
                    orig_x2 = min(orig_w, int((norm_x + norm_w) * orig_w))
                    orig_y2 = min(orig_h, int((norm_y + norm_h) * orig_h))
                    orig_box_w = max(0, orig_x2 - orig_x)
                    orig_box_h = max(0, orig_y2 - orig_y)
                    if orig_box_w <= 0 or orig_box_h <= 0:
                        continue
                    cname = COCO_CLASSES.get(int(cls_id), f"ID:{cls_id}")
                    box_color = (0, 0, 255) if hazard_on_path else (0, 255, 255)
                    label = f"HAZARD: {cname.upper()}" if hazard_on_path else f"{cname} (Off-Path)"
                    cv2.rectangle(stream_frame, (orig_x, orig_y), (orig_x + orig_box_w, orig_y + orig_box_h), box_color, 2)
                    cv2.putText(stream_frame, label, (orig_x, max(20, orig_y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2)

                hud_bg = stream_frame.copy()
                cv2.rectangle(hud_bg, (10, 8), (630, 96), (15, 15, 15), -1)
                stream_frame = cv2.addWeighted(hud_bg, 0.65, stream_frame, 0.35, 0.0)

                cv2.putText(stream_frame, nav_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, nav_color, 2)
                depth_status = "RT" if depth_computed else "Cached"
                telemetry = f"GPU: {smooth_fps:.1f} FPS | Walkable: {sectors['CENTER']['walkable']}px | Depth: {depth_status}"
                cv2.putText(stream_frame, telemetry, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

                tts_display = f"TTS Audio: \"{audio_engine.last_spoken_message if audio_engine else 'Disabled'}\""
                cv2.putText(stream_frame, tts_display, (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1)

                cv2.line(stream_frame, (int(0.35 * orig_w), orig_h - 25), (int(0.35 * orig_w), orig_h), (255, 255, 255), 1)
                cv2.line(stream_frame, (int(0.65 * orig_w), orig_h - 25), (int(0.65 * orig_w), orig_h), (255, 255, 255), 1)

                cv2.imshow(win_perception, perception_frame)
                cv2.imshow(win_stream, stream_frame)

                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Halting SafePath AI...")
    finally:
        cap.release()
        if args.gui:
            cv2.destroyAllWindows()
        print("[STOP] Camera feed released and pipeline stopped cleanly.")

class SafePathEngine:
    """
    Modular SafePath Engine used for testing, benchmarking, and headless verification.
    """
    def __init__(self):
        opts = get_session_options()
        seg_path = resolve_path("deeplabv3_mobilenet_safepath.onnx")
        self.seg_session = ort.InferenceSession(seg_path, opts, providers=PROVIDERS)
        self.seg_in = self.seg_session.get_inputs()[0].name

        yolo_path = resolve_path("yolov8n_hazards.onnx")
        self.yolo_session = ort.InferenceSession(yolo_path, opts, providers=PROVIDERS)
        self.yolo_in = self.yolo_session.get_inputs()[0].name

        depth_path = resolve_path("depth_anything_v2_small.onnx")
        self.depth_session = ort.InferenceSession(depth_path, opts, providers=PROVIDERS)
        self.depth_in = self.depth_session.get_inputs()[0].name

        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def run_segmentation(self, frame):
        tensor = prepare_tensor(frame, 384, 256, apply_norm=True)
        out = self.seg_session.run(None, {self.seg_in: tensor})[0]
        mask = np.argmax(out, axis=1)[0].astype(np.uint8)
        walkable = (mask == 1).astype(np.uint8) * 255
        return cv2.erode(walkable, self.erode_kernel, iterations=1)

    def run_yolo(self, frame):
        tensor = prepare_tensor(frame, 640, 480, apply_norm=False)
        out = self.yolo_session.run(None, {self.yolo_in: tensor})[0]
        preds = out[0].T
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        mask = confidences > 0.45
        valid_boxes = preds[mask, :4]
        valid_confs = confidences[mask]
        valid_classes = class_ids[mask]

        boxes = []
        if len(valid_confs) > 0:
            boxes_xywh = []
            for b in valid_boxes:
                cx, cy, w, h = b
                boxes_xywh.append([int(cx - w / 2.0), int(cy - h / 2.0), int(w), int(h)])
            indices = cv2.dnn.NMSBoxes(boxes_xywh, valid_confs.tolist(), 0.45, 0.45)
            if len(indices) > 0:
                for i in np.array(indices).flatten():
                    cx, cy, w, h = valid_boxes[i]
                    boxes.append(((cx - w / 2.0) / 640.0, (cy - h / 2.0) / 480.0, w / 640.0, h / 480.0, valid_classes[i]))
        return boxes

    def run_depth(self, frame):
        tensor = prepare_tensor(frame, 518, 518, apply_norm=True)
        out = self.depth_session.run(None, {self.depth_in: tensor})[0]
        raw = out[0]
        d_min, d_max = raw.min(), raw.max()
        if d_max > d_min:
            return ((raw - d_min) / (d_max - d_min) * 255.0).astype(np.float32)
        return np.zeros_like(raw, dtype=np.float32)

if __name__ == '__main__':
    main()

