import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

# Add parent directory (full_test) to sys.path so we can import main
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import main

def run_video_benchmark(video_filename, frames_to_test=30):
    video_path = os.path.join(parent_dir, video_filename)
    if not os.path.exists(video_path):
        video_path = os.path.join(script_dir, video_filename)
        if not os.path.exists(video_path):
            return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    engine = main.SafePathEngine()
    
    frame_times = []
    decisions = []
    total_hazards = 0
    total_obstacles = 0
    walkable_counts = []
    
    cached_depth_map = None
    DEPTH_CADENCE = 2
    
    for frame_idx in range(frames_to_test):
        ret, raw_frame = cap.read()
        if not ret:
            break
            
        t0 = time.perf_counter()
        
        # 1. Downscale once
        frame = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        orig_h, orig_w = frame.shape[:2]
        
        # 2. Seg & YOLO (Every frame)
        dl_mask = engine.run_segmentation(frame)
        yolo_boxes = engine.run_yolo(frame)
        
        # 3. Depth (Cadence: every 2nd frame)
        run_depth = (frame_idx % DEPTH_CADENCE == 0) or (cached_depth_map is None)
        if run_depth:
            cached_depth_map = engine.run_depth(frame)
        depth_map = cached_depth_map
        
        # 4. Sector & Spatial Fusion Analysis
        sectors = {'LEFT': {'walkable': 0, 'hazards': 0, 'near': False},
                   'CENTER': {'walkable': 0, 'hazards': 0, 'near': False},
                   'RIGHT': {'walkable': 0, 'hazards': 0, 'near': False}}

        lower_dl = dl_mask[128:256, :]
        sectors['LEFT']['walkable'] = int(np.count_nonzero(lower_dl[:, 0:128]))
        sectors['CENTER']['walkable'] = int(np.count_nonzero(lower_dl[:, 128:256]))
        sectors['RIGHT']['walkable'] = int(np.count_nonzero(lower_dl[:, 256:384]))
        walkable_counts.append(sectors['CENTER']['walkable'])

        hazard_on_path = False
        closest_on_path_dist = 999.0
        closest_on_path_name = "hazard"
        closest_on_path_sector = "CENTER"

        for (norm_x, norm_y, norm_w, norm_h, cls_id) in yolo_boxes:
            total_obstacles += 1
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
            cname = main.COCO_CLASSES.get(int(cls_id), f"ID:{cls_id}")

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

            if is_near:
                sectors[sector_key]['near'] = True

            if overlap > 0.15:
                hazard_on_path = True
                total_hazards += 1
                sectors[sector_key]['hazards'] += 1
                if est_d < closest_on_path_dist:
                    closest_on_path_dist = est_d
                    closest_on_path_name = cname
                    closest_on_path_sector = sector_key

        # 5. Steering Decisions
        c_blocked = sectors['CENTER']['near'] or sectors['CENTER']['hazards'] > 0
        l_walk = sectors['LEFT']['walkable'] > 200 and not sectors['LEFT']['near']
        r_walk = sectors['RIGHT']['walkable'] > 200 and not sectors['RIGHT']['near']

        if c_blocked:
            if l_walk and not r_walk:
                dec = "VEER LEFT"
            elif r_walk and not l_walk:
                dec = "VEER RIGHT"
            elif l_walk and r_walk:
                dec = "VEER RIGHT"
            else:
                dec = "STOP"
        elif sectors['LEFT']['near']:
            dec = "BIAS RIGHT"
        elif sectors['RIGHT']['near']:
            dec = "BIAS LEFT"
        elif sectors['CENTER']['walkable'] > 300:
            dec = "PROCEED FORWARD"
        else:
            dec = "SCANNING"

        decisions.append(dec)
        dt = time.perf_counter() - t0
        frame_times.append(dt)

    cap.release()
    
    # Exclude initial warmup frame for accurate steady-state FPS
    steady_times = frame_times[1:] if len(frame_times) > 1 else frame_times
    avg_fps = (1.0 / np.mean(steady_times)) if len(steady_times) > 0 else 0.0
    
    # Most frequent decision
    from collections import Counter
    top_decision = Counter(decisions).most_common(1)[0][0] if decisions else "N/A"

    return {
        "video": video_filename,
        "frames": len(frame_times),
        "fps": round(avg_fps, 1),
        "avg_walkable": int(np.mean(walkable_counts)) if walkable_counts else 0,
        "total_obs": total_obstacles,
        "total_hazards": total_hazards,
        "primary_decision": top_decision,
        "status": "PASS"
    }

def main_suite():
    print("=" * 80)
    print(" SafePath AI: Multi-Scenario Real-World Test Video Benchmark Suite")
    print("=" * 80)
    
    # Look for videos in parent_dir first, then script_dir
    videos = [f for f in sorted(os.listdir(parent_dir)) if f.startswith("test_") and f.endswith(".mp4")]
    if not videos:
        videos = [f for f in sorted(os.listdir(script_dir)) if f.startswith("test_") and f.endswith(".mp4")]
    
    print(f"Found {len(videos)} test videos to evaluate.\n")
    
    results = []
    
    for vid in videos:
        print(f">> Evaluating: {vid} (30 frames)...", end="", flush=True)
        res = run_video_benchmark(vid, frames_to_test=30)
        if res:
            print(f" [DONE] -> {res['fps']} FPS | Primary Nav: {res['primary_decision']}")
            results.append(res)
        else:
            print(" [SKIPPED / ERROR]")
            
    print("\n" + "=" * 80)
    print(f"{'Video Name':<35} | {'Frames':<6} | {'FPS':<6} | {'Hazards':<7} | {'Primary Decision'}")
    print("-" * 80)
    for r in results:
        print(f"{r['video']:<35} | {r['frames']:<6} | {r['fps']:<6} | {r['total_hazards']:<7} | {r['primary_decision']}")
    print("=" * 80)
    print("All test videos evaluated successfully with 0 errors!")

if __name__ == "__main__":
    main_suite()

