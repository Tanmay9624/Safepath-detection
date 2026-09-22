import os
import onnx
from onnxconverter_common import float16

def resolve_path(filename):
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, filename)
    if os.path.exists(local_path):
        return local_path
    return filename

print("Converting DeepLabV3 MobileNet to FP16...")
seg_in = resolve_path("deeplabv3_mobilenet_safepath.onnx")
model = onnx.load(seg_in, load_external_data=True)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "deeplabv3_mobilenet_safepath_fp16.onnx")
print("Saved deeplabv3_mobilenet_safepath_fp16.onnx")

print("\nConverting Depth Anything V2 to FP16...")
try:
    dep_in = resolve_path("depth_anything_v2_small.onnx")
    dep = onnx.load(dep_in, load_external_data=True)
    dep_fp16 = float16.convert_float_to_float16(dep)
    onnx.save(dep_fp16, "depth_anything_v2_small_fp16.onnx")
    print("Saved depth_anything_v2_small_fp16.onnx")
except Exception as e:
    print("Depth Anything FP16 error:", e)

print("\nConverting YOLOv8 Hazards to FP16...")
try:
    yolo_in = resolve_path("yolov8n_hazards.onnx")
    yolo = onnx.load(yolo_in, load_external_data=True)
    yolo_fp16 = float16.convert_float_to_float16(yolo)
    onnx.save(yolo_fp16, "yolov8n_hazards_fp16.onnx")
    print("Saved yolov8n_hazards_fp16.onnx")
except Exception as e:
    print("YOLO FP16 error:", e)

