import sys,os; os.chdir(os.path.dirname(__file__)); sys.path.insert(0,'.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config
import numpy as np

c=load_config('trainer_app/config_default.yaml')
pp=DataPreprocessor(c)
dc=c['data']
nc=pp._index_nc_files(dc['dataset1_path'])
nd={f.stem.split('_')[-1][:8] for f in nc.values()}
storms=pp._parse_all_csvs(dc['dataset2_path'], nd)
print(f'Storms: {len(storms)} NC: {len(nc)}')

for sid in sorted(storms):
    rows=storms[sid]
    if len(rows)<48: continue
    w=pp._find_all_windows(rows,nc)
    windows,_=w
    if not windows: continue
    # windows[0] = (matched_rows_list, matched_files_list)
    mrows,mfiles=windows[0]
    print(f'\nStorm {sid}: {len(rows)} rows, {len(windows)} windows')
    print(f'  Window rows: {len(mrows)}, files: {len(mfiles)}')
    print(f'  First time: {mrows[0]["time"]}, last: {mrows[-1]["time"]}')
    
    # Test extract on several files
    half=pp.spatial_degree/2.0
    for i,(row,nf) in enumerate(zip(mrows[:5], mfiles[:5])):
        vol=pp._extract_radar_crop(nf,row['lon'],row['lat'],half,29)
        status='OK shape='+str(vol.shape) if vol is not None else 'NONE'
        print(f'  [{i}] {nf.name[:40]} crop={row["lon"]:.1f},{row["lat"]:.1f}: {status}')
    
    # If all None, debug first one
    if all(pp._extract_radar_crop(f,r['lon'],r['lat'],half,29) is None 
           for r,f in zip(mrows[:5],mfiles[:5])):
        print('  ALL NONE! Deep debugging...')
        row=mrows[0]; nf=mfiles[0]
        lm=(pp.lon_arr>=row['lon']-half)&(pp.lon_arr<=row['lon']+half)
        ltm=(pp.lat_arr>=row['lat']-half)&(pp.lat_arr<=row['lat']+half)
        print(f'  lon_idx={lm.sum()} lat_idx={ltm.sum()}')
        if lm.sum()>=4 and ltm.sum()>=4:
            li=np.where(lm)[0]; lai=np.where(ltm)[0]
            ls=np.round(np.linspace(li[0],li[-1],pp.grid_size)).astype(int)
            las=np.round(np.linspace(lai[0],lai[-1],pp.grid_size)).astype(int)
            from trainer_app.data.preprocess import _reconstruct_crop
            dense=_reconstruct_crop(nf,pp.channel_variables,las,ls,29)
            if not dense:
                print('  _reconstruct_crop EMPTY!')
            else:
                for vn in pp.channel_variables:
                    if vn in dense:
                        d=dense[vn]; nr=np.isnan(d).sum()/d.size
                        print(f'  {vn}: {d.shape} NaN={nr:.1%}')
        break
    break
