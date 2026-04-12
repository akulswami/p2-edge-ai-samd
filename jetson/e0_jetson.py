import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import json, time, os, csv
from PIL import Image

TRT_ENGINE = 'models/mobilenetv2_fp16.trt'
MANIFEST   = 'data/manifest.json'
RESULTS    = 'results/e0_jetson.csv'
N_TRIALS   = 10
T_NOMINAL  = 0.100
T_STAR     = 0.05

print('Loading engine...')
logger = trt.Logger(trt.Logger.WARNING)
with open(TRT_ENGINE, 'rb') as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
input_name  = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

h_input  = cuda.pagelocked_empty((1,3,224,224), dtype=np.float16)
h_output = cuda.pagelocked_empty((1,1000),      dtype=np.float16)
d_input  = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream   = cuda.Stream()
context.set_tensor_address(input_name,  int(d_input))
context.set_tensor_address(output_name, int(d_output))

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((224,224))
    arr = np.array(img).astype(np.float32)/255.0
    arr = (arr-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    return arr.transpose(2,0,1).astype(np.float16)[np.newaxis,:]

def infer(inp):
    np.copyto(h_input, inp)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    out = h_output[0].astype(np.float32)
    out = out - out.max()
    out = np.exp(out) / np.exp(out).sum()
    return out

print('Loading manifest and preprocessing images...')
with open(MANIFEST) as f:
    manifest = json.load(f)
images = [preprocess(ex['path']) for ex in manifest]
labels = [ex['label'] for ex in manifest]

print('Computing nominal baseline outputs...')
nominal = np.array([infer(img) for img in images])  # float32

os.makedirs('results', exist_ok=True)
rows = []
for trial in range(N_TRIALS):
    deltas, correct = [], []
    for i, img in enumerate(images):
        t0 = time.monotonic()
        out = infer(img)
        elapsed = time.monotonic() - t0
        # using L-inf norm on softmax probabilities
        delta = float(np.abs(out - nominal[i]).max())
        deltas.append(delta)
        correct.append(int(np.argmax(out) == labels[i]))
        gap = T_NOMINAL - elapsed
        if gap > 0: time.sleep(gap)
    ster   = float(np.mean([d > T_STAR for d in deltas]))
    acc    = float(np.mean(correct))
    d_mean = float(np.mean(deltas))
    d_p99  = float(np.percentile(deltas, 99))
    rows.append({'trial':trial+1,'ster':ster,'acc':acc,'delta_mean':d_mean,'delta_p99':d_p99})
    print(f'Trial {trial+1:02d}/10 | STER={ster:.4f} Acc={acc:.4f} delta_mean={d_mean:.6f} delta_p99={d_p99:.6f}')

with open(RESULTS,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['trial','ster','acc','delta_mean','delta_p99'])
    w.writeheader()
    w.writerows(rows)

ster_vals = [r['ster'] for r in rows]
acc_vals  = [r['acc']  for r in rows]
print(f'--- E0 SUMMARY ---')
print(f'STER: mean={np.mean(ster_vals):.4f}  std={np.std(ster_vals):.4f}')
print(f'Acc:  mean={np.mean(acc_vals):.4f}   std={np.std(acc_vals):.4f}')
print(f'Results saved to {RESULTS}')