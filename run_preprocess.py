#!/usr/bin/env python3
"""Server-side preprocessing runner."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config

config = load_config('trainer_app/config_default.yaml')
dc = config['data']
print(f'dataset1: {dc["dataset1_path"]}')
print(f'dataset2: {dc["dataset2_path"]}')
print(f'processed_dir: {dc["processed_dir"]}')
print(f'in_channels: {dc["in_channels"]}')
print(f'use_storm_motion: {dc["use_storm_motion"]}')

pp = DataPreprocessor(config)
result = pp.validate_datasets(dc['dataset1_path'], dc['dataset2_path'])
print(f'Validation valid: {result["valid"]}')
print(f'nc_count: {result["nc_count"]}, csv_count: {result["csv_count"]}')
if not result['valid']:
    print(f'Errors: {result["errors"]}')
    sys.exit(1)

print('Processing events...')
pp.process_events(
    dc['dataset1_path'],
    dc['dataset2_path'],
    progress_callback=lambda cur, tot, msg: print(f'[{cur}/{tot}] {msg}')
)
print('DONE: preprocessing complete')
