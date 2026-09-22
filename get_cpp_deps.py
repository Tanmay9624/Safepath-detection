import os
import sys
import urllib.request
import zipfile
import subprocess
import shutil

DEPS_DIR = os.path.abspath("deps")
os.makedirs(DEPS_DIR, exist_ok=True)

def download_with_progress(url, dest_path):
    print(f"[DOWNLOAD] Fetching: {url}")
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  -> Progress: {percent}% [{count*block_size // (1024*1024)} MB / {total_size // (1024*1024)} MB]")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    sys.stdout.write("\n")

def setup_onnxruntime():
    ort_dir = os.path.join(DEPS_DIR, "onnxruntime")
    if os.path.exists(ort_dir):
        print("[SKIP] ONNX Runtime C++ SDK already exists in deps/onnxruntime")
        return

    url = "https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-win-x64-1.20.0.zip"
    zip_path = os.path.join(DEPS_DIR, "ort.zip")
    download_with_progress(url, zip_path)
    
    print("[EXTRACT] Unpacking ONNX Runtime C++ SDK...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DEPS_DIR)
    os.remove(zip_path)
    
    extracted_folder = os.path.join(DEPS_DIR, "onnxruntime-win-x64-1.20.0")
    if os.path.exists(extracted_folder):
        if os.path.exists(ort_dir):
            shutil.rmtree(ort_dir)
        os.rename(extracted_folder, ort_dir)
    print("[SUCCESS] ONNX Runtime ready in deps/onnxruntime\n")

def setup_opencv():
    cv_dir = os.path.join(DEPS_DIR, "opencv")
    if os.path.exists(cv_dir):
        print("[SKIP] OpenCV C++ SDK already exists in deps/opencv")
        return

    url = "https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-windows.exe"
    exe_path = os.path.join(DEPS_DIR, "opencv_setup.exe")
    download_with_progress(url, exe_path)
    
    print("[EXTRACT] Extracting OpenCV Windows SDK (self-extracting archive)...")
    cmd = [exe_path, f"-o{DEPS_DIR}", "-y"]
    subprocess.run(cmd, check=True)
    os.remove(exe_path)
    print("[SUCCESS] OpenCV C++ ready in deps/opencv\n")

if __name__ == "__main__":
    print("==================================================")
    print(" Downloading C++ Dependencies for Windows Build   ")
    print("==================================================")
    setup_onnxruntime()
    setup_opencv()
    print("[DONE] All C++ dependencies downloaded and configured!")
    print("You can now run:")
    print("   mkdir build")
    print("   cd build")
    print("   cmake ..")
    print("   cmake --build . --config Release")

