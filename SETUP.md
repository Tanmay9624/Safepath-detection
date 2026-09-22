# SafePath Detection: Cross-Platform Setup & Execution Guide

> **Assistive Navigation System for Visually Impaired Pedestrians**  
> Complete setup and installation manual for running the Python and C++ SafePath engines on **Windows** and **Linux / NVIDIA Jetson**.

---

## 📋 Table of Contents
1. [Prerequisites & Repository Structure](#1-prerequisites--repository-structure)
2. [Python Setup & Execution](#2-python-setup--execution)
   * [Windows (Python Setup)](#a-windows-python-setup)
   * [Linux / Ubuntu / NVIDIA Jetson (Python Setup)](#b-linux--ubuntu--nvidia-jetson-python-setup)
   * [Running the Python Pipeline](#c-running-the-python-pipeline)
3. [C++ Setup & Native Compilation](#3-c-setup--native-compilation)
   * [Windows (MSVC + CMake)](#a-windows-c-build)
   * [Linux / Ubuntu / NVIDIA Jetson (GCC / Clang + CMake)](#b-linux--ubuntu--nvidia-jetson-c-build)
4. [Input Sources: Live Camera vs Pre-Recorded Footage](#4-input-sources-live-camera-vs-pre-recorded-footage)
5. [Troubleshooting & Common Issues](#5-troubleshooting--common-issues)

---

## 1. Prerequisites & Repository Structure

### Hardware Recommendations
* **GPU Systems (Recommended):** NVIDIA GPU (RTX 20/30/40 series, GTX 16 series, or Jetson Orin/Nano) with CUDA 12 support.
* **CPU-Only Systems:** Intel Core i5/i7 (8th Gen+) or AMD Ryzen 5/7 with AVX2 instruction support.

### Clone the Repository
```bash
git clone https://github.com/Tanmay9624/Safepath-detection.git
cd Safepath-detection
```

> [!NOTE]
> All core models (`deeplabv3_mobilenet_safepath.onnx`, `deeplabv3_mobilenet_safepath.onnx.data`, `yolov8n_hazards.onnx`, and `depth_anything_v2_small.onnx`) are bundled directly inside the repository. No external model downloads are required.

---

## 2. Python Setup & Execution

### A. Windows (Python Setup)

1. **Install Python 3.10 – 3.13** from [python.org](https://www.python.org/downloads/) (ensure **"Add Python to PATH"** is checked during installation).
2. **Open PowerShell or Command Prompt** and create a virtual environment:
   ```powershell
   cd Safepath-detection
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install Dependencies:**
   * **For NVIDIA GPU systems (CUDA 12):**
     ```powershell
     pip install -r requirements-gpu.txt
     ```
   * **For CPU-only machines:**
     ```powershell
     pip install -r requirements-cpu.txt
     ```

---

### B. Linux / Ubuntu / NVIDIA Jetson (Python Setup)

1. **Update system packages and install Python venv tools:**
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv libgl1 libglib2.0-0
   ```
2. **Create and activate a virtual environment:**
   ```bash
   cd Safepath-detection
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install Dependencies:**
   * **For NVIDIA GPU Desktop / Laptop (x86_64 with CUDA 12):**
     ```bash
     pip install --upgrade pip
     pip install -r requirements-gpu.txt
     ```
   * **For NVIDIA Jetson (ARM64 Orin / Nano):**
     ```bash
     pip install --upgrade pip
     pip install numpy opencv-python onnx yt-dlp
     # Install JetPack-compatible ONNX Runtime GPU:
     pip install onnxruntime-gpu
     ```
   * **For standard CPU-only Linux:**
     ```bash
     pip install -r requirements-cpu.txt
     ```

---

### C. Running the Python Pipeline

#### 1. Download Sample Walking Footage (Optional)
If you don't have a webcam or test video files, download the real-world crowd benchmark suite:
```bash
python batch_download.py
```
This fetches sample POV walking clips (`test_01_urban_crowd.mp4`, `test_02_suburban_path.mp4`, etc.).

#### 2. Run Real-Time Assistive Navigation
```bash
python main.py

# Or specify custom test environments or live webcam:
python main.py test_05_san_francisco_street.mp4  # Sidewalk walk with cars & sparse pedestrians (SF)
python main.py test_06_residential_walk.mp4        # Quiet residential street with parked cars
python main.py test_07_argentina_street.mp4        # City sidewalk & zebra crosswalk with traffic
python main.py 0                                   # Live USB / Web Camera
```
* **HUD Features:**
  - **Green Corridor:** Safe walkable path calculated via DeepLabV3 + morphological erosion.
  - **Bounding Boxes:** Yellow (safe obstacle off-path) / Red (blocking walkable path).
  - **Proximity:** Monocular depth tags (`NEAR: <2m` vs `AHEAD`).
  - **Assistive Steering Commands:**
    * `NAV: PATH CLEAR - PROCEED FORWARD`
    * `NAV: HAZARD IN CENTER -> VEER RIGHT / VEER LEFT`
    * `NAV: CROWD BLOCKED -> STOP / CAUTION`
  - **Auditory Alert Tone:** Fires when an obstacle is within immediate proximity ($< 2\text{ m}$).
* **Controls:** Press `q` or `ESC` in the display window to exit.

#### 3. Run Automated Performance Verification Benchmark
```bash
python verify_optimizations.py
```
Computes real-time latency over 60 frames and outputs benchmark telemetry.

---

## 3. C++ Setup & Native Compilation

The C++ pipeline (`pipeline.cpp`) provides ultra-low latency inference targeted for edge micro-computers (e.g. wearable backpacks, smart canes, NVIDIA Jetson).

---

### A. Windows (C++ Build)

#### 1. Prerequisites
* **CMake 3.15+**: Download from [cmake.org](https://cmake.org/download/) or install via winget:
  ```powershell
  winget install -e --id Kitware.CMake
  ```
* **Visual Studio 2022 C++ Build Tools** (MSVC compiler with C++17 support).

#### 2. Automated Dependency Setup
Run the automated downloader to fetch pre-compiled OpenCV 4.10.0 and ONNX Runtime C++ SDK:
```powershell
python get_cpp_deps.py
```
*This extracts dependencies into `deps/opencv` and `deps/onnxruntime` automatically.*

#### 3. Compile the Executable
```powershell
# Configure CMake
cmake -B build -S .

# Build in Release Mode
cmake --build build --config Release
```

#### 4. Run the Binary
```powershell
.\build\Release\safepath.exe
```

---

### B. Linux / Ubuntu / NVIDIA Jetson (C++ Build)

#### 1. Install Build Essentials & OpenCV
```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

#### 2. Install ONNX Runtime C++ SDK
* **For x86_64 Linux:**
  ```bash
  # Download ONNX Runtime C++ (v1.20.0)
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz
  tar -xzvf onnxruntime-linux-x64-gpu-1.20.0.tgz
  mkdir -p deps/onnxruntime
  cp -r onnxruntime-linux-x64-gpu-1.20.0/* deps/onnxruntime/
  ```
* **For NVIDIA Jetson (ARM64):**
  ```bash
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-aarch64-1.20.0.tgz
  tar -xzvf onnxruntime-linux-aarch64-1.20.0.tgz
  mkdir -p deps/onnxruntime
  cp -r onnxruntime-linux-aarch64-1.20.0/* deps/onnxruntime/
  ```

#### 3. Build & Run
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run the native pipeline
./safepath
```

---

## 4. Input Sources: Live Camera vs Pre-Recorded Footage

### Switching to a Live Webcam / Wearable Camera
To use a physical USB camera, chest-mounted action cam, or smart glasses instead of video files:

* **In Python (`main.py`):**
  Change line 204 from a filename to camera index `0` (or `1` for external USB camera):
  ```python
  # video_path = resolve_path("test_01_urban_crowd.mp4")
  video_path = 0  # 0 for default webcam / 1 for external USB chest-cam
  ```

* **In C++ (`pipeline.cpp`):**
  Change line 260 from a filename to integer device index:
  ```cpp
  // cv::VideoCapture cap(video_source);
  cv::VideoCapture cap(0); // 0 for default camera
  ```

---

## 5. Troubleshooting & Common Issues

### 1. `CUDAExecutionProvider` Fallback Warning on Windows
* **Problem:** Terminal prints `[WARN] CUDA provider failed, falling back to CPU`.
* **Solution:** Ensure PyTorch CUDA DLLs or the NVIDIA CUDA 12 toolkit are added to your environment path. In Python, `main.py` automatically registers PyTorch's CUDA directory (`torch/lib`). If running C++, verify that `cublasLt64_12.dll` and `cudnn64_9.dll` reside in `build/Release/` or in your Windows system PATH.

### 2. Camera Access Denied on Linux
* Ensure your user account belongs to the `video` group:
  ```bash
  sudo usermod -a -G video $USER
  ```
  *(Log out and log back in for changes to take effect).*

### 3. Missing `*.mp4` Test Videos
* Run the automated video test downloader:
  ```bash
  python batch_download.py
  ```
  Or place any standard `.mp4` video in the project directory.

---

## 📄 License & Team
* **Project:** SafePath Detection (`safepath_detection`)
* **Purpose:** Capstone Engineering Assistive Navigation Project
* **Team:** Group 7
