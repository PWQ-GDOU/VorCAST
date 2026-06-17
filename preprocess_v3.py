#!/usr/bin/env python3
"""Preprocess tornado data v3 - fixed date filter."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config

t0 = time.time()
c = load_config('trainer_app/config_default.yaml')
pp = DataPreprocessor(c)
dc = c['data']
print(f'Config: in_channels={dc["in_channels"]}, use_storm_motion={dc["use_storm_motion"]}')
result = pp.validate_datasets(dc['dataset1_path'], dc['dataset2_path'])
print(f'Validation: valid={result["valid"]}, nc={result["nc_count"]}, csv={result["csv_count"]}')
print('Starting preprocessing...')
pp.process_events(
    dc['dataset1_path'],
    dc['dataset2_path'],
    progress_callback=lambda cur, tot, msg: print(f'[{cur}/{tot}] {msg}')
)
elapsed = time.time() - t0
import glob
count = len(glob.glob(os.path.join(dc['processed_dir'], '*.npz')))
print(f'ALL DONE in {elapsed/60:.1f} min, {count} samples')
