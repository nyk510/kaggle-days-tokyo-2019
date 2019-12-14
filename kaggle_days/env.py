import os

DATA_DIR = '/analysis/data'

RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

for d in [RAW_DIR, OUTPUT_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
