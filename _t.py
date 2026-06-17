import sys; sys.path.insert(0,'.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config
c=load_config('trainer_app/config_default.yaml')
pp=DataPreprocessor(c)
nc=pp._index_nc_files(c['data']['dataset1_path'])
nd={f.stem.split('_')[-1][:8] for f in nc.values()}
print('ND:',nd)
s=pp._parse_all_csvs(c['data']['dataset2_path'], nd)
print('Total:',len(s))
ts=[k for k in s if '_2023010' in k]
print('2023010x storms:',len(ts))
for k in sorted(ts)[:5]:
    print(f'  {k}: {len(s[k])}')
a04=[k for k in s if '_20230104' in k]
print('20230104:',len(a04))
if a04:
    for k in a04[:2]:
        rows=s[k]
        print(f'  {k}: {len(rows)} rows, times=[{rows[0]["time"]}]')
