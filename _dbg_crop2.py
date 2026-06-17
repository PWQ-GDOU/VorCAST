import sys,os; os.chdir(os.path.dirname(__file__)); sys.path.insert(0,'.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config
from datetime import datetime as dt
import numpy as np

c=load_config('trainer_app/config_default.yaml')
pp=DataPreprocessor(c)
dc=c['data']
nc=pp._index_nc_files(dc['dataset1_path'])
nd={f.stem.split('_')[-1][:8] for f in nc.values()}
print(f'NC: {len(nc)} files, dates={nd}')
storms=pp._parse_all_csvs(dc['dataset2_path'], nd)
print(f'Storms: {len(storms)}')

for sid in sorted(storms):
    rows=storms[sid]
    if len(rows)<48: continue
    w=pp._find_all_windows(rows,nc)
    if not w: continue
    windows,err=w
    if not windows: continue
    win=windows[0]
    print(f'\nStorm {sid}: {len(rows)}r, {len(windows)}w')
    print(f'Window: {len(win["matched_rows"])} rows, {len(win["matched_files"])} files')
    print(f'Times: {[r["time"].strftime("%H%MZ") for r in win["matched_rows"][:5]]}...')
    
    nf=win['matched_files'][0]
    row=win['matched_rows'][0]
    print(f'\nTesting crop on {nf.name}')
    print(f'  center=({row["lon"]:.2f},{row["lat"]:.2f}) half={pp.spatial_degree/2}')
    print(f'  lon_arr: {pp.lon_arr.min():.2f}-{pp.lon_arr.max():.2f} ({len(pp.lon_arr)}pts)')
    
    half=pp.spatial_degree/2.0
    lm=(pp.lon_arr>=row['lon']-half)&(pp.lon_arr<=row['lon']+half)
    ltm=(pp.lat_arr>=row['lat']-half)&(pp.lat_arr<=row['lat']+half)
    print(f'  lon_idx: {lm.sum()}, lat_idx: {ltm.sum()}')
    
    if lm.sum()<4 or ltm.sum()<4:
        print('  FAIL: crop too small')
    else:
        vol=pp._extract_radar_crop(nf,row['lon'],row['lat'],half,29)
        if vol is None:
            print('  FAIL: extract returned None')
            from trainer_app.data.preprocess import _reconstruct_crop, _check_nan_excessive
            li=np.where(lm)[0]; lai=np.where(ltm)[0]
            ls=np.round(np.linspace(li[0],li[-1],pp.grid_size)).astype(int)
            las=np.round(np.linspace(lai[0],lai[-1],pp.grid_size)).astype(int)
            dense=_reconstruct_crop(nf,pp.channel_variables,las,ls,29)
            if not dense:
                print('  _reconstruct_crop returned EMPTY!')
            else:
                for vn in pp.channel_variables:
                    if vn in dense:
                        d=dense[vn]; nr=np.isnan(d).sum()/d.size
                        print(f'  {vn}: {d.shape} NaN={nr:.1%}',end='')
                        if nr>pp.nan_max_ratio: print(f' EXCESSIVE!',end='')
                        print()
                    else:
                        print(f'  {vn}: MISSING')
        else:
            print(f'  SUCCESS: shape={vol.shape}')
    break
