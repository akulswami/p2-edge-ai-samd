"""
E3 Worker — continuous inference co-tenant stressor.
Launched as a subprocess by e3_jetson.py.
Runs indefinitely until killed by parent.
Usage: python3 e3_worker.py <trt_engine_path> <manifest_path>
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import json, sys, os

TRT_ENGINE = sys.argv[1]
MANIFEST   = sys.argv[2]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1).astype(np.float16)[np.newaxis, :]

logger = trt.Logger(trt.Logger.ERROR)
with open(TRT_ENGINE, "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
input_name  = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)
in_shape    = tuple(engine.get_tensor_shape(input_name))
out_shape   = tuple(engine.get_tensor_shape(output_name))

h_in  = cuda.pagelocked_empty(in_shape,  dtype=np.float16)
h_out = cuda.pagelocked_empty(out_shape, dtype=np.float16)
d_in  = cuda.mem_alloc(h_in.nbytes)
d_out = cuda.mem_alloc(h_out.nbytes)
stream = cuda.Stream()

with open(MANIFEST) as f:
    manifest = json.load(f)

images = [preprocess(e["path"]) for e in manifest[:100]]  # 100-image loop is enough

idx = 0
while True:
    arr = images[idx % len(images)]
    np.copyto(h_in, arr)
    cuda.memcpy_htod_async(d_in, h_in, stream)
    context.set_tensor_address(input_name,  int(d_in))
    context.set_tensor_address(output_name, int(d_out))
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_out, d_out, stream)
    stream.synchronize()
    idx += 1
