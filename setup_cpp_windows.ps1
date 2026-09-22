<#
================================================================================
C++ DEPENDENCY SETUP FOR WINDOWS
================================================================================

WHY DO WE NEED TO DOWNLOAD ALL THIS JUST FOR C++?

When you use Python (like in main.py), you just type `pip install opencv-python onnxruntime`. 
Python's package manager automatically downloads pre-compiled Windows binaries and 
handles all the messy behind-the-scenes linking for you.

C++ does NOT have a package manager like `pip`. To run `pipeline.cpp`, you must 
manually assemble the building blocks yourself:

1. A Compiler (Visual Studio Build Tools): C++ is a compiled language. You need 
   Microsoft's C++ compiler (MSVC) to translate `pipeline.cpp` into machine code.
2. CMake: A tool that reads `CMakeLists.txt` and tells the Microsoft compiler 
   exactly where all your folders are.
3. ONNX Runtime C++ API: You need to manually download the C++ headers (.h files) 
   and dynamic libraries (.dll) from Microsoft's GitHub so your C++ code knows 
   what `Ort::Session` means.
4. OpenCV C++ API: Same as above. You need the raw C++ libraries for computer vision.

This script automates downloading the ONNX Runtime binaries and installing the 
basic Windows compilers.
================================================================================
#>

Write-Host "Starting C++ Dependency Setup for Windows..." -ForegroundColor Cyan

# 1. Install CMake
Write-Host "`n[1/4] Installing CMake..." -ForegroundColor Yellow
winget install -e --id Kitware.CMake --accept-source-agreements --accept-package-agreements

# 2. Install Visual Studio C++ Build Tools (The Compiler)
# NOTE: This is a massive download (~6GB). It runs silently in the background.
Write-Host "`n[2/4] Installing MSVC C++ Compiler (This may take a while)..." -ForegroundColor Yellow
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait" --accept-source-agreements --accept-package-agreements

# 3. Download ONNX Runtime C++ GPU Binaries for Windows
Write-Host "`n[3/4] Downloading ONNX Runtime C++ Binaries..." -ForegroundColor Yellow
$OnnxUrl = "https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-win-x64-gpu-1.17.1.zip"
$OnnxZip = "onnxruntime-win-x64-gpu.zip"
Invoke-WebRequest -Uri $OnnxUrl -OutFile $OnnxZip
Write-Host "Extracting ONNX Runtime..." -ForegroundColor Yellow
Expand-Archive -Path $OnnxZip -DestinationPath ".\" -Force
Remove-Item $OnnxZip
Write-Host "ONNX Runtime extracted to: $(Get-Location)\onnxruntime-win-x64-gpu-1.17.1" -ForegroundColor Green

# 4. OpenCV Instructions
Write-Host "`n[4/4] OpenCV C++ Setup" -ForegroundColor Yellow
Write-Host "OpenCV requires a manual self-extracting download."
Write-Host "1. Go to: https://opencv.org/releases/"
Write-Host "2. Download the 'Windows' version (an .exe file)."
Write-Host "3. Run it and extract it to C:\opencv"
Write-Host "4. Once done, you must update CMakeLists.txt to point to C:\opencv and your new ONNX Runtime folder." -ForegroundColor Red

Write-Host "`nSetup Script Complete!" -ForegroundColor Cyan

