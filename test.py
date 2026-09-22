import os
import sys
import time
import cv2
import numpy as np

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

def get_color_mask(pred_mask):
    """Maps 4 Mapillary class indices to distinct BGR colors."""
    color_palette = np.array([
        [0, 0, 0],       # 0: Background (Black/Transparent)
        [0, 255, 0],     # 1: Walkable Path (Green)
        [255, 0, 0],     # 2: Road (Blue)
        [0, 0, 255]      # 3: Hazards (Red)
    ], dtype=np.uint8)
    return color_palette[pred_mask]

def resolve_path(filename):
    """Resolves file paths strictly within the full_test folder."""
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, filename)
    if os.path.exists(local_path):
        return local_path
    return filename

def process_video(video_path="test_01_urban_crowd.mp4", onnx_model_path="deeplabv3_mobilenet_safepath.onnx"):
    video_path = resolve_path(video_path)
    onnx_model_path = resolve_path(onnx_model_path)
        
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(onnx_model_path, session_opts, providers=providers)
        active_provider = ort_session.get_providers()[0]
        print(f"[INIT] Loaded ONNX model using provider: {active_provider}")
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        return

    input_name = ort_session.get_inputs()[0].name
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    # Normalization constants (ImageNet standard)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    target_size = (384, 256)  # (Width, Height)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    print(f"\n[START] Streaming video inference: {video_path}")
    print(">> Press 'q' or 'ESC' to quit.")

    fps_history = []

    while True:
        t0 = time.perf_counter()
        ret, original_frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        original_h, original_w = original_frame.shape[:2]

        # Preprocess frame
        resized_frame = cv2.resize(original_frame, target_size, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - mean) / std
        
        input_tensor = np.transpose(img_norm, (2, 0, 1))[None, ...].astype(np.float32)

        # Run ONNX inference
        outputs = ort_session.run(None, {input_name: input_tensor})[0]
        pred_mask = np.argmax(outputs, axis=1)[0].astype(np.uint8)

        # Apply Safety Erosion Buffer to Walkable Path (Class 1)
        walkable = (pred_mask == 1).astype(np.uint8)
        safe_walkable = cv2.erode(walkable, erode_kernel, iterations=1)
        pred_mask[walkable == 1] = 0
        pred_mask[safe_walkable == 1] = 1

        # Resize mask back to original display size
        pred_mask_full = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        color_mask = get_color_mask(pred_mask_full)

        # Alpha blend overlay (0.35 mask, 0.65 frame)
        overlay = cv2.addWeighted(color_mask, 0.35, original_frame, 0.65, 0.0)

        # FPS computation
        dt = time.perf_counter() - t0
        fps = 1.0 / max(dt, 1e-5)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        smooth_fps = sum(fps_history) / len(fps_history)

        # On-screen HUD
        cv2.putText(overlay, f"SafePath Segmentation | {active_provider} | {smooth_fps:.1f} FPS", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(overlay, "Green: Safe Walkable (Eroded) | Red: Hazards | Blue: Road", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("SafePath Semantic Segmentation Test", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else ("test_05_san_francisco_street.mp4" if os.path.exists(resolve_path("test_05_san_francisco_street.mp4")) else "test_01_urban_crowd.mp4")
    process_video(video_path=vid)

