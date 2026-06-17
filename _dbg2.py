import sys; sys.path.insert(0,'.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config
c = load_config('trainer_app/config_default.yaml')
pp = DataPreprocessor(c)
dc = c['data']
nc_files = pp._index_nc_files(dc['dataset1_path'])

# Check NC dates extraction
nc_dates = {f.stem.split("_")[-1][:8] for f in nc_files.values()}
print("NC dates:", nc_dates)

# Check which CSV files exist and would be filtered
from pathlib import Path
csv_dir = Path(dc['dataset2_path'])
csv_stems = {f.stem for f in csv_dir.glob("*.csv")}
print("CSV stems sample:", sorted(list(csv_stems))[:5], "...", sorted(list(csv_stems))[-3:])

# Check which are filtered out
for stem in sorted(csv_stems)[:5]:
    in_nc = stem in nc_dates
    print(f"  {stem}.csv: in_nc={in_nc}")

# Now parse with filter and check for 20230104 storms
storms = pp._parse_all_csvs(dc['dataset2_path'], nc_dates)
dates_20230104 = [k for k in storms if "_20230104" in k]
print(f"\nStorms with 20230104: {len(dates_20230104)}")
if dates_20230104:
    for s in dates_20230104[:3]:
        rows = storms[s]
        times = [r['time'].strftime('%Y%m%dT%H%MZ') for r in rows[:3]]
        print(f"  {s}: {len(rows)} rows, first times: {times}")
