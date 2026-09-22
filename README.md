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
| **Preprocessing Speed** | ~1.5 ms (Vectorized NumPy) | **~1.1 ms** (AVX2 SIMD `cv::split` + `memcpy`) |
| **Optimal Use Case** | Interactive prototyping, fast development | **Zero-overhead edge deployment** (Jetson / x86) |

---

## ⚡ Performance Benchmark (RTX 3050 Laptop GPU)

| Operation | Unoptimized (CPU) | GPU Accelerated | GPU + Cadence Caching |
| :--- | :--- | :--- | :--- |
| **DeepLabV3 MobileNet** | ~75 ms | **6.9 ms** | **6.9 ms** (Runs 100% of frames) |
| **YOLOv8 Hazards** | ~60 ms | **6.9 ms** | **6.9 ms** (Runs 100% of frames) |
| **Depth Anything V2** | ~450 ms | **48.0 ms** | **24.0 ms blended** (Runs every 2nd frame) |
| **Preprocessing & Overlap Math** | ~45 ms (CPU) | ~1.5 ms | **~1.5 ms** (SIMD / pre-downscaling) |
| **Total Frame Latency** | **~500+ ms** | **~74 ms** | **55.8 ms blended** (28.1 ms on odd frames) |
| **Sustained Pipeline FPS** | **~2.0 FPS** | **~13.5 FPS** | **~18.0–20.0+ FPS** 🚀 |

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
├── deeplabv3_mobilenet_safepath.onnx       # Trained semantic segmentation model (Graph)
├── deeplabv3_mobilenet_safepath.onnx.data  # Model weight tensors (44 MB)
├── main.py                       # High-speed GPU Python pipeline
├── pipeline.cpp                  # Multi-threaded C++ production engine
├── CMakeLists.txt                # Cross-platform build script (Windows / Linux)
├── optimization.md               # In-depth benchmark telemetry and optimization audit
├── verify_optimizations.py       # Automated benchmark and verification script
├── requirements.txt              # Unified Python requirements specification
├── requirements-gpu.txt          # GPU-accelerated requirements (NVIDIA CUDA 12)
├── requirements-cpu.txt          # CPU-only lightweight requirements
├── setup_cpp_windows.ps1         # Windows C++ compiler setup guide
├── download_videos.py            # YouTube test video downloader
├── batch_download.py             # Multi-scenario video test suite downloader
├── check_onnx.py                 # ONNX input/output dimension validator
├── test_01_urban_crowd.mp4       # Real-world test: Dense pedestrian crowd (Tokyo)
├── test_02_suburban_path.mp4     # Real-world test: Residential sidewalk navigation
├── test_03_rainy_night.mp4       # Real-world test: Reflective streets & harsh weather
└── test_04_chest_mount.mp4       # Real-world test: Downward white-cane walking POV
```

---

## 🛠️ Step-by-Step Usage Guide

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

# Run real-time GPU inference with cadence caching
python main.py
```

* **Video Selection:** Change `video_path = ...` in `main.py` to test different environments (`test_01_urban_crowd.mp4`, `test_02_suburban_path.mp4`, etc.).
* **Controls:** Press `q` or `ESC` in the display window to exit.

---

### 4. Running the Compiled C++ Executable (`safepath.exe`)

The C++ multi-threaded executable has been pre-compiled for Windows with full CUDA GPU support:

```powershell
# From safepath_detection/full_test:
.\build\Release\safepath.exe
```

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

### 6. Running the Automated Performance Verification Benchmark

To verify latency and save visual verification artifacts:

```bash
python verify_optimizations.py
```
This processes 60 frames, computes heavy vs. cadence frame latency, and writes timestamped verification images (`opt_verification_frame10.jpg`, `frame25.jpg`, `frame40.jpg`) directly to `full_test/`.

---

## 👥 Project & Team

* **Project:** `safepath_detection` (SafePath AI)
* **Team:** Group 7

