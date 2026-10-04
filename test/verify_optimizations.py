import os
import sys
import time
import numpy as np
import cv2

# Ensure parent directory (full_test) is in sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the engine from main.py
import main

def test_pipeline_optimizations(video_path="test_01_urban_crowd.mp4", num_frames=60):
    print("==================================================================")
    print(" [TEST] SafePath AI Optimization & Verification Benchmark")
    print("==================================================================")
    
    video_path = main.resolve_path(video_path)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open test video: {video_path}")
        return

    engine = main.SafePathEngine()
    
    even_frame_times = []  # Full computation (Seg + YOLO + Depth)
    odd_frame_times = []   # Cadence cached (Seg + YOLO + Cached Depth)
    
    cached_depth_map = None
    DEPTH_CADENCE = 2
    
    saved_frames = []
    
    print(f"\n[BENCHMARK] Processing {num_frames} frames from: {video_path}...")
    
    for i in range(num_frames):
        ret, raw_frame = cap.read()
        if not ret:
            break
            
        t_start = time.perf_counter()
        
        # 1. Downscale once
        frame = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        orig_h, orig_w = frame.shape[:2]
        
        # 2. Seg & YOLO (Every frame)
        dl_mask = engine.run_segmentation(frame)
        yolo_boxes = engine.run_yolo(frame)
        
        # 3. Depth (Cadence: every 2nd frame)
        run_depth = (i % DEPTH_CADENCE == 0) or (cached_depth_map is None)
        if run_depth:
            cached_depth_map = engine.run_depth(frame)
        depth_map = cached_depth_map
        
        # 4. Rendering & Fusion
        full_mask = cv2.resize(dl_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        green_overlay = frame.copy()
        green_overlay[full_mask > 0] = [0, 255, 0]
        frame = cv2.addWeighted(green_overlay, 0.35, frame, 0.65, 0.0)
        
        hazard_on_path = False
        hazard_count = 0
        obstacle_count = 0
        
        for (norm_x, norm_y, norm_w, norm_h, cls_id) in yolo_boxes:
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

            orig_x = max(0, int(norm_x * orig_w))
            orig_y = max(0, int(norm_y * orig_h))
            orig_x2 = min(orig_w, int((norm_x + norm_w) * orig_w))
            orig_y2 = min(orig_h, int((norm_y + norm_h) * orig_h))
            orig_box_w = max(0, orig_x2 - orig_x)
            orig_box_h = max(0, orig_y2 - orig_y)
            if orig_box_w <= 0 or orig_box_h <= 0:
                continue

            box_color = (0, 255, 255)
            label = "Obstacle"
            obstacle_count += 1

            if overlap > 0.15:
                hazard_on_path = True
                hazard_count += 1
                box_color = (0, 0, 255)

                if depth_map is not None:
                    x1_md = max(0, int(norm_x * 518))
                    y1_md = max(0, int(norm_y * 518))
                    x2_md = min(518, int((norm_x + norm_w) * 518))
                    y2_md = min(518, int((norm_y + norm_h) * 518))
                    if x2_md > x1_md and y2_md > y1_md:
                        depth_roi = depth_map[y1_md:y2_md, x1_md:x2_md]
                        max_d = float(np.max(depth_roi))
                        label = "HAZARD: NEAR" if max_d > 180.0 else "HAZARD: AHEAD"

            cv2.rectangle(frame, (orig_x, orig_y), (orig_x + orig_box_w, orig_y + orig_box_h), box_color, 2)
            cv2.putText(frame, label, (orig_x, max(20, orig_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        
        t_total = (time.perf_counter() - t_start) * 1000.0  # ms
        
        # Save telemetry
        if run_depth:
            even_frame_times.append(t_total)
        else:
            odd_frame_times.append(t_total)
            
        # Draw HUD
        status_text = "STATUS: HAZARD INTERSECTING PATH" if hazard_on_path else "STATUS: PATH CLEAR"
        status_color = (0, 0, 255) if hazard_on_path else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        
        mode_text = "Depth: Real-Time" if run_depth else "Depth: Cached"
        cv2.putText(frame, f"Frame {i:02d} | {mode_text} | Total: {t_total:.1f}ms", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    
        # Save visual artifacts at milestone frames
        if i in [10, 25, 40]:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_name = os.path.join(script_dir, f"opt_verification_frame{i}.jpg")
            cv2.imwrite(save_name, frame)
            saved_frames.append((os.path.basename(save_name), i, hazard_count, obstacle_count, t_total))
            
    cap.release()
    
    # Quantitative Analysis
    avg_even = np.mean(even_frame_times[2:]) if len(even_frame_times) > 2 else np.mean(even_frame_times)
    avg_odd  = np.mean(odd_frame_times[2:]) if len(odd_frame_times) > 2 else np.mean(odd_frame_times)
    blended_avg = (avg_even + avg_odd) / 2.0
    blended_fps = 1000.0 / blended_avg
    
    print("\n" + "="*66)
    print(" [RESULTS] QUANTITATIVE PERFORMANCE AUDIT RESULTS")
    print("="*66)
    print(f"  1. Heavy Frames (Seg + YOLO + Depth):    {avg_even:.1f} ms (~{1000/avg_even:.1f} FPS)")
    print(f"  2. Cadence Frames (Seg + YOLO + Cached): {avg_odd:.1f} ms (~{1000/avg_odd:.1f} FPS)")
    print(f"  3. Blended Real-World Frame Time:        {blended_avg:.1f} ms")
    print(f"  4. SUSTAINED PIPELINE THROUGHPUT:        {blended_fps:.1f} FPS")
    print("="*66)
    
    print("\n[ARTIFACTS] Visual Verification Images Saved:")
    for name, f_num, h_cnt, o_cnt, lat in saved_frames:
        print(f"  - {name}: Frame #{f_num} | Latency: {lat:.1f}ms | Hazards on path: {h_cnt} | Total obstacles: {o_cnt}")
        
    return blended_fps

if __name__ == '__main__':
    test_pipeline_optimizations()

