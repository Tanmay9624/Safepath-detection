import onnxruntime as ort

models = {
    "Segmentation": "deeplabv3_mobilenet_safepath.onnx",
    "Depth": "../new_suite/depth_anything_v2_small.onnx",
    "YOLO": "../yolov8n_hazards.onnx"
}

for name, path in models.items():
    try:
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        print(f"--- {name} ---")
        print(f"Input Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")
        print(f"Output Name: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}\n")
    except Exception as e:
        print(f"Failed to load {name}: {e}")

