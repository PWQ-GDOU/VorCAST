import sys
sys.path.insert(0, '.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config

c = load_config('trainer_app/config_default.yaml')
pp = DataPreprocessor(c)

nc_files = pp._index_nc_files(c['data']['dataset1_path'])
nc_dates = {f.stem.split('_')[-1][:8] for f in nc_files.values()}
print(f'NC dates: {nc_dates}')

storms = pp._parse_all_csvs(c['data']['dataset2_path'], nc_dates)
print(f'Filtered storms: {len(storms)}')
sids = list(storms.keys())[:5]
print(f'Sample IDs: {sids}')

# Try first storm
sid = sids[0]
rows = storms[sid]
print(f'Storm {sid}: {len(rows)} rows, time range: {rows[0]["time"]} to {rows[-1]["time"]}')

entry, samples, skipped = pp._process_one_storm(sid, rows, nc_files)
print(f'Result: status={entry.get("status")}, samples={samples}, skipped={skipped}')
if not skipped:
    print(f'  Windows found: {entry.get("windows_found")}')
    print(f'  Samples saved: {entry.get("samples_saved")}')
