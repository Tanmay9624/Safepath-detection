import os
import sys
import time
import queue
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

def prepare_tensor(image, w, h, apply_norm=True):
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if apply_norm:
        rgb = (rgb - MEAN) / STD
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

# ==============================================================================
# THREAD 1: Obstacle Detection & Depth Analysis Pipeline (Sequential on 1 Thread)
# ==============================================================================
def worker_obstacle_depth(in_q, out_q, depth_cadence=2):
    """
    Runs YOLOv8 Hazard Detection AND Depth Anything V2 sequentially on a single thread.
    - YOLO detects obstacle bounding boxes.
    - Depth Anything generates the 3D depth map (with cadence caching).
    """
    opts = get_session_options()
    
    # 1. Load YOLOv8 Hazards
    print("  [Thread-1] Loading YOLOv8 Hazards...")
    yolo_session = ort.InferenceSession("../yolov8n_hazards.onnx", opts, providers=PROVIDERS)
    yolo_in_name = yolo_session.get_inputs()[0].name

    # 2. Load Depth Anything V2
    print("  [Thread-1] Loading Depth Anything V2...")
    depth_session = ort.InferenceSession("../new_suite/depth_anything_v2_small.onnx", opts, providers=PROVIDERS)
    depth_in_name = depth_session.get_inputs()[0].name
    
    print("  [Thread-1] Obstacle & Depth Worker initialized on GPU.")
    
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
# THREAD 2: Semantic SafePath Segmentation (Parallel on Another Thread)
# ==============================================================================
def worker_deeplab(in_q, out_q):
    """
    Runs DeepLabV3 MobileNet Walkable Path Segmentation and safety erosion in parallel.
    """
    opts = get_session_options()
    print("  [Thread-2] Loading SafePath DeepLabV3 MobileNet...")
    seg_session = ort.InferenceSession("deeplabv3_mobilenet_safepath.onnx", opts, providers=PROVIDERS)
    seg_in_name = seg_session.get_inputs()[0].name
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    print("  [Thread-2] SafePath Segmentation Worker initialized on GPU.")

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
# MAIN ENGINE: Frame Ingest, Thread Synchronization & Visual Fusion
# ==============================================================================
def main():
    print("==================================================================")
    print(" SafePath AI: Multi-Threaded Dual Pipeline Architecture")
    print("  - Thread 1: YOLOv8 Hazards + Depth Anything V2 (Sequential)")
    print("  - Thread 2: DeepLabV3 MobileNet SafePath (Concurrent)")
    print("==================================================================")

    # Bounded communication queues (size=2 to prevent frame latency buildup)
    obs_in_q = queue.Queue(maxsize=2)
    obs_out_q = queue.Queue(maxsize=2)
    seg_in_q = queue.Queue(maxsize=2)
    seg_out_q = queue.Queue(maxsize=2)

    DEPTH_CADENCE = 2  # Depth runs every 2nd frame on Thread 1

    # Launch worker threads
    t_obs = threading.Thread(target=worker_obstacle_depth, 
                             args=(obs_in_q, obs_out_q, DEPTH_CADENCE), 
                             daemon=True)
    t_seg = threading.Thread(target=worker_deeplab, 
                             args=(seg_in_q, seg_out_q), 
                             daemon=True)
    t_obs.start()
    t_seg.start()

    video_path = "test_01_urban_crowd.mp4"
    if not os.path.exists(video_path):
        video_path = "../test/test_sidewalk.f134.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_path}")
        return

    print(f"\n[START] Streaming video: {video_path}")
    print(">> Press 'q' or 'ESC' in the window to stop.")

    frame_count = 0
    last_time = time.perf_counter()
    smooth_fps = 0.0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            # Seamless loop at end of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Downscale raw frame once to 640x360 for high-speed queue handling & display
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

        # 3. Overlay Walkable Path (Green)
        full_mask = cv2.resize(dl_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        green_overlay = frame.copy()
        green_overlay[full_mask > 0] = [0, 255, 0]
        frame = cv2.addWeighted(green_overlay, 0.35, frame, 0.65, 0.0)

        hazard_on_path = False

        # 4. Spatial Overlap Math & Depth Sampling
        for (norm_x, norm_y, norm_w, norm_h, cls_id) in yolo_boxes:
            dl_x = max(0, int(norm_x * 384))
            dl_y = max(0, int(norm_y * 256))
            dl_w = min(384 - dl_x, int(norm_w * 384))
            dl_h = min(256 - dl_y, int(norm_h * 256))

            if dl_w <= 0 or dl_h <= 0:
                continue

            path_roi = dl_mask[dl_y:dl_y+dl_h, dl_x:dl_x+dl_w]
            overlap = cv2.countNonZero(path_roi) / float(dl_w * dl_h)

            orig_x = max(0, int(norm_x * orig_w))
            orig_y = max(0, int(norm_y * orig_h))
            orig_box_w = min(orig_w - orig_x, int(norm_w * orig_w))
            orig_box_h = min(orig_h - orig_y, int(norm_h * orig_h))

            box_color = (0, 255, 255)  # Yellow: General Obstacle off-path
            label = "Obstacle"

            if overlap > 0.15:
                hazard_on_path = True
                box_color = (0, 0, 255)  # Red: Blocking Walkable Path

                if depth_map is not None:
                    md_x = max(0, int(norm_x * 518))
                    md_y = max(0, int(norm_y * 518))
                    md_w = min(518 - md_x, int(norm_w * 518))
                    md_h = min(518 - md_y, int(norm_h * 518))

                    if md_w > 0 and md_h > 0:
                        depth_roi = depth_map[md_y:md_y+md_h, md_x:md_x+md_w]
                        max_d = float(np.max(depth_roi))
                        label = "HAZARD: NEAR" if max_d > 180.0 else "HAZARD: AHEAD"

            cv2.rectangle(frame, (orig_x, orig_y), (orig_x + orig_box_w, orig_y + orig_box_h), box_color, 2)
            cv2.putText(frame, label, (orig_x, max(20, orig_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # 5. Smooth FPS Calculation
        dt = time.perf_counter() - t0
        curr_fps = 1.0 / max(dt, 1e-5)
        smooth_fps = curr_fps if smooth_fps == 0.0 else (0.85 * smooth_fps + 0.15 * curr_fps)

        # 6. Status Banner
        status_text = "STATUS: HAZARD INTERSECTING PATH" if hazard_on_path else "STATUS: PATH CLEAR"
        status_color = (0, 0, 255) if hazard_on_path else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        depth_status = "RT" if depth_computed else "Cached"
        fps_text = f"Dual-Thread GPU: {smooth_fps:.1f} FPS | Depth: {depth_status}"
        cv2.putText(frame, fps_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow("SafePath AI: Dual-Thread Architecture", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
