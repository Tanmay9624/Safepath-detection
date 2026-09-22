# SafePath AI: Multi-Modal Assistive Navigation Pipeline

> **5th Semester Capstone Project — Group 7**  
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

## 🚀 Dual Implementations: Python vs. C++

The project provides two feature-complete, production-grade implementations sharing identical mathematical fusion logic and GPU acceleration:

### 1. Python Implementation (`main.py`)
* **Best For:** High-efficiency Python deployment, algorithmic tuning, and interactive testing.
* **Backend:** ONNX Runtime GPU (`CUDAExecutionProvider`) dynamically linked to PyTorch CUDA 12 binaries.
* **Dual-Thread Execution Architecture:**
  * **Thread 1 (`worker_obstacle_depth`):** Executes **YOLOv8 Hazards** (640×480) and **Depth Anything V2** (518×518) sequentially on a single thread. Eliminates inter-thread synchronization overhead between bounding box generation and depth proximity sampling.
  * **Thread 2 (`worker_deeplab`):** Executes **DeepLabV3 MobileNet** (384×256) and $7\times 7$ morphological safety erosion concurrently in parallel.
  * **Main Thread:** Dispatches frames to both bounded queues (`obs_in_q`, `seg_in_q`), synchronizes outputs, computes walkable path intersection ($>15\%$), and renders the HUD.
```
                                [ Camera / Video Ingest ]
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          ┌─────────────────────┐                       ┌─────────────────────┐
          │      THREAD 1       │                       │      THREAD 2       │
          │   (Obstacle/Depth)  │                       │      (SafePath)     │
          ├─────────────────────┤                       ├─────────────────────┤
          │ 1. YOLOv8 Hazards   │                       │ 1. DeepLabV3        │
          │    (640x480)        │                       │    MobileNet        │
          │         ↓           │                       │    (384x256)        │
          │ 2. Depth Anything   │                       │         ↓           │
          │    (518x518)        │                       │ 2. 7x7 Morphological│
          │    (Sequential)     │                       │    Safety Erosion   │
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
* **Performance on RTX 3050:** **~18–20 FPS sustained** with Cadence Caching (`DEPTH_CADENCE = 2`).

### 2. High-Performance C++ Pipeline (`pipeline.cpp` $\rightarrow$ `safepath.exe`)
* **Best For:** Production deployment on embedded edge devices (NVIDIA Jetson / x86_64).
* **Backend:** Native C++17, OpenCV 4.10.0, ONNX Runtime C++ GPU API (CUDA 12).
* **Architecture:** Bounded, lock-free multi-threading (`std::thread`, `BoundedQueue`) across three dedicated worker threads:
  * `worker_deeplab`: Asynchronous semantic inference + safety erosion.
  * `worker_yolo`: Asynchronous anchor decoding & Non-Maximum Suppression (NMS).
  * `worker_depth_anything`: Asynchronous 518×518 depth tensor processing.
  * `main`: Lock-free synchronization, spatial overlap calculation, and rendering.
* **Optimizations:** SIMD-vectorized tensor preprocessing (`cv::split` & `std::memcpy`), elimination of POSIX `pthread` on Windows, and zero runtime interpreter overhead.

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

*(Detailed telemetry and root-cause analysis documented in [`optimization.md`](file:///D:/5th%20SEM/Project/NFT_Project_Group7/full_test/optimization.md)).*

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

### 1. Running the Python Pipeline (`main.py`)

Activate your Python virtual environment and run:

```bash
cd "D:\5th SEM\Project\NFT_Project_Group7\full_test"

# Run real-time GPU inference with cadence caching
python main.py
```

* **Video Selection:** Change line 113 (`video_path = ...`) in `main.py` to test different environments (`test_01_urban_crowd.mp4`, `test_02_suburban_path.mp4`, etc.).
* **Controls:** Press `q` or `ESC` in the display window to exit.

---

### 2. Running the Compiled C++ Executable (`safepath.exe`)

The C++ multi-threaded executable has been pre-compiled for Windows with full CUDA GPU support:

```powershell
cd "D:\5th SEM\Project\NFT_Project_Group7\full_test"

# Launch the native compiled binary
.\build\Release\safepath.exe
```

---

### 3. Rebuilding the C++ Executable from Source (CMake)

If you modify `pipeline.cpp`, recompile using CMake and MSVC:

```powershell
cd "D:\5th SEM\Project\NFT_Project_Group7\full_test"

# 1. Configure the build system
& "C:\Program Files\CMake\bin\cmake.exe" -B build -S .

# 2. Compile in Release mode
& "C:\Program Files\CMake\bin\cmake.exe" --build build --config Release

# 3. Run the updated executable
.\build\Release\safepath.exe
```

---

### 4. Running the Automated Performance Verification Benchmark

To verify latency and save visual verification artifacts:

```bash
python verify_optimizations.py
```
This processes 60 frames, computes heavy vs. cadence frame latency, and writes timestamped verification images (`opt_verification_frame10.jpg`, `frame25.jpg`, `frame40.jpg`) directly to `full_test/`.

---

## 👥 Contributors

* **Academic Program:** Bachelor of Engineering (Computer Science / AI)
* **Course:** 5th Semester Capstone Engineering Project
* **Project Team:** Group 7
