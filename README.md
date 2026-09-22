# SafePath AI: Multi-Modal Assistive Navigation Pipeline

> **safepath_detection — Assistive Navigation System**  
> An edge-optimized, real-time computer vision system fusing custom semantic segmentation, monocular depth estimation, and dynamic obstacle detection to calculate safe walking paths for visually impaired pedestrians.

---

## 📌 Project Architecture

```
                                  [ Input Video Frame ]
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
   [ DeepLabV3 MobileNetV3 ]      [ Depth Anything V2 ]          [ YOLOv8 Hazards ]
   Custom Mapillary 4-Class        ViT-Small Backbone            Bounding Box Detection
   Input: 384x256 @ CUDA          Input: 518x518 @ CUDA         Input: 640x480 @ CUDA
   Execution: Every Frame (1x)    Execution: Cadence (1/2x)     Execution: Every Frame (1x)
               │                            │                            │
               ▼                            ▼                            ▼
   Walkable Path Mask (Green)      Relative Depth Map            Obstacle Bounding Boxes
               │                            │                            │
       [ Morphological ]                    │                            │
      [ Safety Erosion ]                    │                            │
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            ▼
                           [ Spatial & Depth Fusion Engine ]
                     - Calculates bounding box overlap with walkable path
                     - Samples depth ROI: Near (<2m) vs Ahead (>2m)
                     - Generates real-time audio/visual navigation warnings
```

---

## 🚀 Dual Threading Implementations: Python vs. C++

The project provides two distinct, production-grade threading implementations. Because Python and C++ possess fundamentally different memory and concurrency models, **we designed specialized threading architectures tailored to the strengths and limitations of each language runtime**:

---

### 1. Python Threading Architecture (`main.py`): Dual-Thread Co-Processing

In Python, the **Global Interpreter Lock (CPython GIL)** prevents multiple native threads from executing pure Python bytecode simultaneously. Spawning 3 or 4 fine-grained Python threads creates severe thread-switching latency, lock contention, and queue serialization overhead.

To maximize throughput under the GIL, we engineered a **Dual-Thread Co-Processing Architecture**:

```
                                [ Camera / Video Ingest ]
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          ┌─────────────────────┐                       ┌─────────────────────┐
          │      THREAD 1       │                       │      THREAD 2       │
          │  (Obstacle & Depth) │                       │  (SafePath Nav)     │
          ├─────────────────────┤                       ├─────────────────────┤
          │ 1. YOLOv8 Hazards   │                       │ 1. DeepLabV3        │
          │    (640x480)        │                       │    MobileNet        │
          │         ↓           │                       │    (384x256)        │
          │ 2. Depth Anything   │                       │         ↓           │
          │    (518x518)        │                       │ 2. 7x7 Morphological│
          │    (Cadence 1/2)    │                       │    Safety Erosion   │
          └──────────┬──────────┘                       └──────────┬──────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                                   [ MAIN THREAD / HUD ]
                             - Synchronizes both queues
                             - Evaluates Walkable Path overlap (>15%)
                             - Samples Depth ROI (NEAR vs AHEAD)
                             - Alpha-blends Green corridor & Status Banner
```

#### Why We Chose This Architecture for Python:
1. **Mitigating GIL Lock Contention:**
   * Passing data through Python's `queue.Queue` requires acquiring and releasing the GIL. Reducing the worker count from 3 down to 2 cuts queue synchronization points by **50%**, eliminating thread thrashing.
2. **Coupling Bounding Boxes with Depth Proximity:**
   * In our navigation math, Depth is **only sampled inside YOLO bounding boxes**. By running YOLO and Depth sequentially on Thread 1, the depth map is generated immediately after bounding boxes are decoded on the same thread—with **zero inter-thread transfer latency**.
3. **Balanced Workload Distribution:**
   * Thread 1 runs YOLO (6.9 ms) + Depth (48 ms every 2nd frame) $\rightarrow$ average latency: **~31 ms**.
   * Thread 2 runs DeepLab (6.9 ms) $\rightarrow$ average latency: **~7 ms**.
   * Because both threads run concurrently in parallel, the total cycle time is bounded by Thread 1 (~31 ms), delivering a smooth **~30–32 FPS** under Python.

---

### 2. C++ Threading Architecture (`pipeline.cpp`): Tri-Thread Lock-Free Pipeline

In C++, there is **NO Global Interpreter Lock**. Native C++17 `std::thread` instances run on true operating system hardware threads, allowing full concurrent execution across multiple CPU cores and asynchronous CUDA streams simultaneously.

To exploit raw hardware parallelism, we engineered a **Tri-Thread Producer-Consumer Architecture**:

```
                              [ Camera / Ingest Thread ]
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
          ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
          │  worker_deeplab   │  │    worker_yolo    │  │ worker_depth_v2   │
          ├───────────────────┤  ├───────────────────┤  ├───────────────────┤
          │ DeepLabV3         │  │ YOLOv8 Hazards    │  │ Depth Anything V2 │
          │ 384x256 @ CUDA    │  │ 640x480 @ CUDA    │  │ 518x518 @ CUDA    │
          │ + cv::erode buffer│  │ + NMS Suppression │  │ + Min/Max Norm    │
          └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           ▼
                                [ Main Consumer Thread ]
                          - BoundedQueue synchronization
                          - Spatial Overlap Math & Depth ROI
                          - Visual Rendering & Display
```

#### Why We Chose This Architecture for C++:
1. **True Multi-Core Hardware Parallelism (Zero GIL):**
   * Unlike Python, C++ worker threads run on independent CPU cores with zero lock contention. DeepLab, YOLO, and Depth Anything dispatch their CUDA kernels concurrently without stalling each other.
2. **Bounded Non-Blocking Queues (`BoundedQueue<T>`):**
   * Uses low-level `std::mutex` and `std::condition_variable` with a fixed buffer size of 2. If one model experiences a transient spike in processing time, the queue automatically discards stale frames to guarantee **zero latency accumulation**.
3. **Hardware SIMD Vectorization (`prepare_tensor`):**
   * Replaced slow pixel-by-pixel loops with OpenCV's `cv::split` and contiguous block memory copy (`std::memcpy`), leveraging AVX2 CPU vector extensions for **sub-millisecond tensor preparation**.
4. **Targeted for Embedded Edge Hardware:**
   * This design is specifically portable to resource-constrained edge platforms (such as the NVIDIA Jetson Orin / Nano), where maximizing GPU stream concurrency is essential.

---

### ⚖️ Architectural Comparison: Python vs. C++

| Feature | Python Implementation (`main.py`) | C++ Implementation (`pipeline.cpp`) |
| :--- | :--- | :--- |
| **Worker Threads** | **2 Threads** (Obstacle/Depth + SafePath) | **3 Threads** (DeepLab + YOLO + Depth) |
| **Concurrency Model** | Dual-Thread Co-Processing | Tri-Thread Producer-Consumer |
| **Inter-Thread Sync** | Thread-safe `queue.Queue` | Custom `BoundedQueue` (`std::condition_variable`) |
| **Primary Bottleneck Mitigated** | **Python GIL contention** & queue overhead | **GPU stream serialization** & CPU memory copy |
| **Preprocessing Speed** | ~1.5 ms (Vectorized NumPy) | **~0.3–1.1 ms** (AVX2 SIMD `cv::split` + planar pointers) |
| **Sustained GPU Throughput** | **~20.0–25.0+ FPS** (Cadence 2x) | **~33.4–42.3 FPS** (Cadence 2x / 3x) 🚀 |
| **Optimal Use Case** | Interactive prototyping, fast development | **Zero-overhead edge deployment** (Jetson / x86) |

---

## ⚡ Performance Benchmark (RTX 3050 Laptop GPU)

| Operation | Unoptimized (CPU) | GPU Accelerated | GPU + Cadence Caching |
| :--- | :--- | :--- | :--- |
| **DeepLabV3 MobileNet** | ~75 ms | **6.9 ms** | **6.9 ms** (Runs 100% of frames) |
| **YOLOv8 Hazards** | ~60 ms | **6.9 ms** | **6.9 ms** (Runs 100% of frames) |
| **Depth Anything V2** | ~450 ms | **48.0 ms** | **24.0 ms blended** (Runs every 2nd frame) |
| **Preprocessing & Overlap Math** | ~45 ms (CPU) | ~1.5 ms | **~1.1 ms** (SIMD / pre-downscaling) |
| **Total Frame Latency** | **~500+ ms** | **~74 ms** | **29.9 ms (C++) / 55.8 ms (Python)** |
| **Sustained Pipeline FPS** | **~2.0 FPS** | **~13.5 FPS** | **33.4 FPS (C++) / ~20 FPS (Python)** 🚀 |

*(Detailed telemetry and root-cause analysis documented in [`optimization.md`](./optimization.md)).*

---

## 🔬 Core Innovations

### 1. Custom 4-Class Semantic Segmentation
Trained on the **Mapillary Vistas V2.0** street-scene dataset, compressed from 124 classes down to 4 safety-critical labels:
* **Class 0 (Background):** Sky, buildings, trees. Weighted at `0.5` in loss to permanently prevent false-positive hazard hallucinations in the sky.
* **Class 1 (Walkable Path):** Sidewalks, crosswalks, pedestrian zones. Weighted at `3.0`.
* **Class 2 (Roadway):** Curbs and vehicular road surfaces. Weighted at `1.0`.
* **Class 3 (Hazards):** Vehicles, pedestrians, poles, barriers, fences, animals. Weighted heavily at `4.0` to penalize false negatives (missed collisions).

### 2. Morphological Safety Buffer
Sidewalks have physical drop-offs, curbs, and uneven edges. The pipeline applies a `7x7` elliptical morphological erosion (`cv2.morphologyEx` / `cv::erode`) to the raw walkable mask, shrinking the navigation corridor inward to keep the pedestrian safely centered.

### 3. Spatial & Depth Fusion Math
Rather than computing distance globally, the pipeline calculates the spatial intersection between each YOLO obstacle bounding box and the DeepLab walkable mask:
$$\text{Overlap Ratio} = \frac{\sum_{(x, y) \in \text{ROI}} \mathbf{M}_{\text{walkable}}(x, y)}{\text{Area}(\text{ROI})}$$
* **Overlap $> 0.15$:** Obstacle blocks the user's immediate walking path $\rightarrow$ Box turns **RED**.
  * The system samples $\max(\mathbf{D}_{\text{ROI}})$ on the **Depth Anything V2** map:
    * **`HAZARD: NEAR`** ($\text{Depth} > 180$) $\rightarrow$ Urgent collision warning.
    * **`HAZARD: AHEAD`** ($\text{Depth} \le 180$) $\rightarrow$ Navigational advisory.
* **Overlap $\le 0.15$:** Obstacle is safely outside the path $\rightarrow$ Box remains **YELLOW** (`"Obstacle"`).

### 4. Sector-Based Steering Guidance (Blind Navigation Engine)
To give direct assistive cues to a visually impaired user navigating through crowds, the lower walking corridor is partitioned into three vertical navigation zones (**Left**, **Center**, **Right**):
* **`NAV: PATH CLEAR — PROCEED FORWARD`:** Center path is open and walkable.
* **`NAV: HAZARD IN CENTER $\rightarrow$ VEER RIGHT / VEER LEFT`:** Identifies which side of the sidewalk has clear walkable space and steers the user away from incoming pedestrians.
* **`NAV: CROWD BLOCKED $\rightarrow$ STOP / CAUTION`:** Dense pedestrian obstruction across all sectors.
* **Audible Proximity Cue:** A non-blocking alert tone (`winsound.Beep`) fires when an obstacle enters the near walking zone ($< 2\text{ m}$).

---

## 📁 Directory Structure

```
full_test/
├── build/                        # Compiled C++ binaries and build tree
│   └── Release/
│       ├── safepath.exe          # Fully compiled C++ GPU executable
│       ├── onnxruntime.dll       # Runtime ONNX DLL
│       ├── onnxruntime_providers_cuda.dll  # CUDA GPU execution provider
│       ├── opencv_world4100.dll  # Runtime OpenCV DLL
│       └── *.dll                 # Bundled CUDA 12 / cuDNN 9 runtime libraries
├── deps/                         # Self-contained C++ SDKs (OpenCV & ONNX Runtime)
│   ├── onnxruntime/              # ONNX Runtime C++ GPU SDK (v1.20.0)
│   └── opencv/                   # Pre-compiled OpenCV 4.10.0 Windows SDK
├── test/                         # Dedicated test and benchmark suite
│   ├── __init__.py
│   ├── test_audio.py             # Priority audio engine verification test suite
│   ├── test.py                   # Semantic segmentation standalone visual test
│   ├── verify_optimizations.py   # Latency benchmark and visual artifact verification
│   ├── benchmark_suite.py        # Multi-scenario real-world test video evaluation
│   └── check_onnx.py             # ONNX input/output dimension validator
├── deeplabv3_mobilenet_safepath.onnx       # Trained semantic segmentation model (Graph)
├── deeplabv3_mobilenet_safepath.onnx.data  # Model weight tensors (44 MB)
├── depth_anything_v2_small.onnx            # Monocular depth estimation ONNX model
├── yolov8n_hazards.onnx                    # Real-time obstacle detection ONNX model
├── main.py                       # High-speed GPU Python pipeline (Headless by default)
├── quad_view_main.py             # 4-Panel Single-Window Quad Stream (1280x720)
├── audio_engine.py               # Asynchronous Priority TTS Speech & Audio Engine
├── pipeline.cpp                  # Multi-threaded C++ production engine
├── CMakeLists.txt                # Cross-platform build script (Windows / Linux)
├── optimization.md               # In-depth benchmark telemetry and optimization audit
├── walkthrough.md                # Visual walkthrough, verification report and snapshots
├── requirements.txt              # Unified Python requirements specification
├── requirements-gpu.txt          # GPU-accelerated requirements (NVIDIA CUDA 12)
├── requirements-cpu.txt          # CPU-only lightweight requirements
├── setup_cpp_windows.ps1         # Windows C++ compiler setup guide
└── batch_download.py             # Multi-scenario video test suite downloader
```

---

## 🛠️ Step-by-Step Usage Guide

> [!TIP]
> For a comprehensive installation and setup walkthrough across Windows and Linux / NVIDIA Jetson, see [**`SETUP.md`**](./SETUP.md).

### 1. Python Environment Setup

Install the required dependencies based on your hardware:

* **For NVIDIA GPU Systems (RTX / GTX with CUDA 12):**
  ```bash
  pip install -r requirements-gpu.txt
  ```
* **For CPU-Only Systems (No NVIDIA GPU):**
  ```bash
  pip install -r requirements-cpu.txt
  ```

---

### 2. Download Test Videos (Optional)

If you are cloning this repository fresh and want sample POV pedestrian footage:

```bash
python batch_download.py
```

---

### 3. Running the Python Pipeline (`main.py`)

Navigate to the project directory and run:

```bash
# If running from the parent repository:
cd safepath_detection/full_test

# Or if you are already inside the folder:
cd full_test

# 1. Run Headless with default Local Machine Webcam:
python main.py

# 2. Run with smartphone IP Webcam (auto-normalizes to http://<ip>:<port>/video):
python main.py --ip 192.168.1.100:8080

# 3. Run with video file path (positional or via --video flag):
python main.py path/to/video.mp4
python main.py --video path/to/video.mp4

# 4. Optional GUI visual mode (opens dual OpenCV windows for debugging):
python main.py --gui

# 5. Throughput & Execution Tuning:
python main.py path/to/video.mp4 --depth-cadence 3  # Boosts throughput to ~30-35+ FPS
python main.py path/to/video.mp4 --max-frames 50    # Stops after 50 frames (automated testing)
```

* **Execution Mode:** Completely **Headless** by default (zero GUI windows). Renders all assistive directives through real-time TTS audio speech, 1000 Hz earcon warning beeps, and a live console status dashboard.
* **Continuous Playback vs. Test Limits:** By default (`--max-frames 0`), the pipeline executes continuously in real time (loops video files indefinitely until `Ctrl+C`). Passing `--max-frames N` stops after $N$ frames for testing and benchmarks.
* **Pipeline Frame Rates:** Runs at **~20 to 25+ FPS** in steady state on GPU. Increase `--depth-cadence` to `3` or `4` to push sustained performance to **30–35+ FPS**.
* **Camera Fallback:** Automatically defaults to the local machine camera (`Index 0`) with Windows DirectShow acceleration if no IP webcam or video path is specified.
* **Audio Heartbeat Tuning:** Add `--clear-interval 15.0` (default) or `--clear-interval 0` for pure alert-by-exception.
* **Controls:** Press `Ctrl+C` in the terminal (or `q` / `ESC` if running with `--gui`) to cleanly stop.

---

### 3.1 Running the 4-Panel Quad View Stream (`quad_view_main.py`)

To view all perception and navigation layers simultaneously in a single, unified 1280x720 window:

```bash
# Run 4-panel quad stream (Panel 1: YOLO, Panel 2: Depth V2 full-res, Panel 3: DeepLabV3, Panel 4: HUD):
python quad_view_main.py

# Or specify a custom video / webcam:
python quad_view_main.py --source path/to/video.mp4
python quad_view_main.py --source 0

# Optional flags:
python quad_view_main.py --clear-interval 15.0  # Seconds between 'Path is clear' audio (default: 15.0)
python quad_view_main.py --clear-interval 0     # Pure alert-by-exception (silent when path is clear)
python quad_view_main.py --no-tts               # Run silently without speech audio
python quad_view_main.py --conf 0.35            # Custom YOLO confidence threshold
```
* **Panel 1 (Top-Left):** YOLOv8 Detections & Class Labels.
* **Panel 2 (Top-Right):** Depth Anything V2 High-Resolution Colormap + YOLOv8 Overlays & Metric Distances.
* **Panel 3 (Bottom-Left):** DeepLabV3 Walkable Path Segmentation & Safety Buffer Contour.
* **Panel 4 (Bottom-Right):** Complete Assistive Navigation HUD (Fusion, Steering & Audio Banner).
* **Interactive Snapshot:** Press `s` anytime to save an instant snapshot of the 4-panel view. Press `q` or `ESC` to exit.

---

### 4. Running the Compiled C++ Executable (`safepath.exe`)

The C++ multi-threaded executable has been pre-compiled for Windows with full CUDA GPU acceleration and Cadence Caching:

```powershell
# 1. Run with live camera or default video:
.\build\Release\safepath.exe

# 2. Run with video path:
.\build\Release\safepath.exe path\to\video.mp4

# 3. High-Performance Headless Edge Mode (33.4+ FPS with Cadence 2x):
.\build\Release\safepath.exe path\to\video.mp4 --headless --depth-cadence 2

# 4. Ultra-Fast High-Throughput Mode (42.3+ FPS with Cadence 3x):
.\build\Release\safepath.exe path\to\video.mp4 --headless --depth-cadence 3

# 5. Automated Benchmark Run (e.g. 50 frames limit):
.\build\Release\safepath.exe path\to\video.mp4 50 --headless
```
* **Cadence Scaling:** `--depth-cadence 2` (default, runs Depth on even frames $\rightarrow$ 33.4 FPS) or `--depth-cadence 3` (runs Depth every 3rd frame $\rightarrow$ 42.3 FPS).
* **Execution Options:** Add `--headless` for maximum edge throughput with zero GUI overhead, or run without `--headless` for interactive OpenCV visualization.

---

### 5. Rebuilding the C++ Executable from Source (CMake)

If you modify `pipeline.cpp`, recompile using CMake and MSVC:

```powershell
# 1. Configure the build system
cmake -B build -S .

# 2. Compile in Release mode
cmake --build build --config Release

# 3. Run the updated executable
.\build\Release\safepath.exe
```
*(Note: If `cmake` is not added to your Windows PATH, invoke `& "C:\Program Files\CMake\bin\cmake.exe"`).*

---

### 6. Test Suite & Verification Scripts (`test/`)

All automated benchmarks, unit tests, and model validators reside inside the dedicated [`test/`](./test/) folder:

```bash
# 1. Automated Performance & Latency Benchmark:
python test/verify_optimizations.py

# 2. Multi-Scenario Real-World Test Video Benchmark Suite:
python test/benchmark_suite.py

# 3. Priority Audio Engine & Preemption Unit Tests:
python test/test_audio.py

# 4. Standalone Semantic Segmentation Visual Test:
python test/test.py path/to/video.mp4

# 5. ONNX Model Input/Output Tensor Shape Validator:
python test/check_onnx.py
```
* **Performance Verification (`test/verify_optimizations.py`):** Processes 60 frames, computes heavy vs. cadence frame latency, and writes timestamped verification images (`opt_verification_frame10.jpg`, `frame25.jpg`, `frame40.jpg`) directly to `test/`.
* **Benchmark Suite (`test/benchmark_suite.py`):** Runs SafePath across real-world pedestrian test videos, reporting steady-state FPS, hazard counts, and navigation decisions in a summary table.
* **Audio Unit Test (`test/test_audio.py`):** Validates priority preemption (Priority 1 immediate override over Priority 3 guidance), 1000 Hz proximity warning tones, and debounce cooldown timers.
* **Model Validator (`test/check_onnx.py`):** Inspects input/output dimensions and execution providers for DeepLabV3, Depth Anything V2, and YOLOv8 models.

---

## 👥 Project

* **Project:** `safepath_detection` (SafePath AI)

