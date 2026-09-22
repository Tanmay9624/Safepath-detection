import os
import sys
import onnxruntime as ort

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

def resolve_path(filename):
    if os.path.exists(filename):
        return filename
    local_path = os.path.join(script_dir, filename)
    if os.path.exists(local_path):
        return local_path
    parent_path = os.path.join(parent_dir, filename)
    if os.path.exists(parent_path):
        return parent_path
    return filename

models = {
    "Segmentation": resolve_path("deeplabv3_mobilenet_safepath.onnx"),
    "Depth": resolve_path("depth_anything_v2_small.onnx"),
    "YOLO": resolve_path("yolov8n_hazards.onnx")
}

for name, path in models.items():
    try:
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        print(f"--- {name} ({os.path.basename(path)}) ---")
        print(f"Input Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")
        print(f"Output Name: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}\n")
    except Exception as e:
        print(f"Failed to load {name} from '{path}': {e}")

