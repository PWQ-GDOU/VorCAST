import sys; sys.path.insert(0,'.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config
from pathlib import Path
c=load_config('trainer_app/config_default.yaml')
pp=DataPreprocessor(c)
nc_files=pp._index_nc_files(c['data']['dataset1_path'])
nd={f.stem.split('_')[-1][:8] for f in nc_files.values()}
print('NC_DATES:',nd)

csv_dir=Path(c['data']['dataset2_path'])
for cf in sorted(csv_dir.glob('*.csv')):
    sd=cf.stem
    m=sd in nd
    print(f'{sd}: match={m}')
    if m: print(f'  -> WOULD PROCESS')
