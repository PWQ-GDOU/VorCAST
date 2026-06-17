import sys; sys.path.insert(0,'.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config
c = load_config('trainer_app/config_default.yaml')
pp = DataPreprocessor(c)
dc = c['data']
nc_files = pp._index_nc_files(dc['dataset1_path'])
nc_dates = {f.stem.split("_")[-1][:8] for f in nc_files.values()}
print("NC dates:", nc_dates)
storms = pp._parse_all_csvs(dc['dataset2_path'], nc_dates)
print("Storms:", len(storms))
print("T_in:", c["data"]["T_in"], "T_out:", c["data"]["T_out"])
viable = [(sid,len(rows)) for sid,rows in storms.items() if len(rows)>=48]
viable.sort(key=lambda x:-x[1])
print("Viable (>=48 rows):", len(viable))
for sid,n in viable[:5]:
    print(f"  {sid}: {n} rows")
