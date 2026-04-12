from datasets import load_dataset
from PIL import Image
import json, os, random
SEED = 42
N = 500
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
print('Loading dataset...')
ds = load_dataset('zh-plus/tiny-imagenet', split='valid', streaming=True)
random.seed(SEED)
pool = []
for i, ex in enumerate(ds):
    if i >= 3000: break
    pool.append(ex)
random.shuffle(pool)
sample = pool[:N]
os.makedirs(OUT_DIR, exist_ok=True)
manifest = []
for i, ex in enumerate(sample):
    img_path = os.path.join(OUT_DIR, f'img_{i:04d}.jpg')
    ex['image'].convert('RGB').save(img_path, 'JPEG', quality=95)
    manifest.append({'idx': i, 'path': img_path, 'label': ex['label']})
    if (i+1) % 50 == 0: print(f'  Saved {i+1}/{N}')
with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Done. {N} images saved.')