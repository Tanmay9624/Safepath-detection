import onnx
from onnxconverter_common import float16

print("Converting DeepLabV3 MobileNet to FP16...")
model = onnx.load("deeplabv3_mobilenet_safepath.onnx", load_external_data=True)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "deeplabv3_mobilenet_safepath_fp16.onnx")
print("Saved deeplabv3_mobilenet_safepath_fp16.onnx")

print("\nConverting Depth Anything V2 to FP16...")
try:
    dep = onnx.load("../new_suite/depth_anything_v2_small.onnx", load_external_data=True)
    dep_fp16 = float16.convert_float_to_float16(dep)
    onnx.save(dep_fp16, "../new_suite/depth_anything_v2_small_fp16.onnx")
    print("Saved depth_anything_v2_small_fp16.onnx")
except Exception as e:
    print("Depth Anything FP16 error:", e)

print("\nConverting YOLOv8 Hazards to FP16...")
try:
    yolo = onnx.load("../yolov8n_hazards.onnx", load_external_data=True)
    yolo_fp16 = float16.convert_float_to_float16(yolo)
    onnx.save(yolo_fp16, "../yolov8n_hazards_fp16.onnx")
    print("Saved yolov8n_hazards_fp16.onnx")
except Exception as e:
    print("YOLO FP16 error:", e)

