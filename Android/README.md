SafePath Detection (Android Edge AI)
SafePath is a real-time, edge-computing assistive navigation system designed for Android. It processes live camera feeds entirely on-device to provide auditory steering cues for visually impaired users or autonomous navigation systems.

The application bridges a modern Kotlin CameraX frontend with a highly optimized, multi-threaded C++ backend utilizing ONNX Runtime and OpenCV via JNI (Java Native Interface).

🧠 System Architecture
The pipeline is designed for high-throughput edge inference, decoupling the UI from heavy AI workloads using a multi-threaded producer-consumer architecture.

1. Frontend (Kotlin + CameraX)
CameraX: Captures live frames in YUV format, extracting the raw byte array.

JNI Bridge: Passes the raw bytes and physical paths of the pre-loaded AI models to the C++ engine.

Text-to-Speech (TTS): Receives asynchronous string outputs from the C++ decision engine and vocalizes them to the user.

2. Backend (C++ + ONNX Runtime)
The C++ engine spins up four concurrent worker threads synchronized via custom BoundedQueue structures:

Thread 1 (DeepLabV3+): Performs semantic segmentation to identify walkable paths (e.g., sidewalks, pavement) and divides the view into left, center, and right sectors.

Thread 2 (YOLOv8): Detects dynamic hazards, pedestrians, and obstacles, outputting normalized bounding boxes.

Thread 3 (Depth Anything V2): Runs at a staggered cadence (every 2nd frame) to estimate spatial proximity and prevent collisions with approaching objects.

Thread 4 (Decision Engine): Aggregates mask overlaps, bounding box coordinates, and depth maps. It calculates spatial logic (e.g., hazard overlapping walkable path) and generates text-based steering commands (e.g., "HAZARD IN CENTER -> VEER RIGHT").

📂 Repository Structure
Plaintext
android/
├── app/src/main/
│   ├── java/com/example/myapplication/ # Kotlin UI, Camera, and TTS logic
│   ├── cpp/native-lib.cpp              # Core C++ JNI bridge and multi-threaded workers
│   └── assets/                         # Model directory (ignored in version control)
├── gradle/                             # Gradle wrapper files
└── build.gradle.kts                    # Project dependencies (OpenCV, ONNX Runtime)
⚙️ Setup & Installation Procedure
Due to GitHub's strict file size limitations, the pre-trained ONNX models are not included in this repository. You must manually add them to your local environment before compiling.

1. Clone the Repository
Bash
git clone https://github.com/Tanmay9624/Safepath-detection.git
cd Safepath-detection/android
2. Add the AI Models
Navigate to app/src/main/ and ensure the assets folder exists.

Place your exported ONNX models into the assets directory. Your folder must look exactly like this to run properly:

app/src/main/assets/yolov8n_hazards.onnx

app/src/main/assets/deeplabv3_mobilenet_safepath.onnx

app/src/main/assets/deeplabv3_mobilenet_safepath.onnx.data (Weights file required for DeepLab)

app/src/main/assets/depth_anything_v2_small.onnx

3. Open in Android Studio
Launch Android Studio and click Open.

Select the android folder inside the cloned repository.

Allow Gradle to sync. This will automatically download the necessary OpenCV and ONNX Runtime dependencies declared in the build.gradle.kts files.

🚀 How to Build and Run
Connect a Physical Device: This application requires significant CPU resources and physical camera hardware. Emulators are heavily discouraged. Enable USB Debugging or Wireless Debugging on your Android device.

Deploy: Click the green Run (Shift + F10) button in Android Studio to build the APK.

Permissions: Upon the first launch, grant the requested Camera permissions.

Testing Outdoors: Unplug your device and point the camera at a physical pathway, campus path, or sidewalk. Ensure your device volume is turned up to hear the TTS navigation instructions.

Testing Indoors (Simulation): If you cannot test outdoors, open a "POV City Walking Tour" video on your computer in full screen. Point your phone's camera at the monitor. The AI will process the video feed and generate steering commands as if you were walking down the street.
