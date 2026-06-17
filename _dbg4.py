import sys,os; os.chdir(os.path.dirname(__file__)); sys.path.insert(0,'.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config
import numpy as np

c=load_config('trainer_app/config_default.yaml')
pp=DataPreprocessor(c)
dc=c['data']
nc=pp._index_nc_files(dc['dataset1_path'])
# Load coords! (process_events does this)
pp._load_coordinates(list(nc.values())[0])
print(f'lon_arr loaded: {pp.lon_arr is not None}')
nd={f.stem.split('_')[-1][:8] for f in nc.values()}
storms=pp._parse_all_csvs(dc['dataset2_path'], nd)
half=pp.spatial_degree/2.0

for sid in sorted(storms):
    rows=storms[sid]
    if len(rows)<48: continue
    w=pp._find_all_windows(rows,nc)
    windows,_=w
    if not windows: continue
    mrows,mfiles=windows[0]
    print(f'Storm {sid}: {len(rows)} rows, {len(windows)} windows, window={len(mrows)} steps')
    
    # Test extract
    ok=0
    for row,nf in zip(mrows[:5],mfiles[:5]):
        vol=pp._extract_radar_crop(nf,row['lon'],row['lat'],half,29)
        if vol is not None: ok+=1
        else:
            # Debug why
            lm=(pp.lon_arr>=row['lon']-half)&(pp.lon_arr<=row['lon']+half)
            ltm=(pp.lat_arr>=row['lat']-half)&(pp.lat_arr<=row['lat']+half)
            if lm.sum()<4 or ltm.sum()<4:
                print(f'  SKIP: crop area too small lon={lm.sum()} lat={ltm.sum()}')
            else:
                from trainer_app.data.preprocess import _reconstruct_crop
                li=np.where(lm)[0]; lai=np.where(ltm)[0]
                ls=np.round(np.linspace(li[0],li[-1],pp.grid_size)).astype(int)
                las=np.round(np.linspace(lai[0],lai[-1],pp.grid_size)).astype(int)
                dense=_reconstruct_crop(nf,pp.channel_variables,las,ls,29)
                if not dense:
                    print(f'  SKIP: reconstruct empty at {row["lon"]:.2f},{row["lat"]:.2f}')
                else:
                    nans=[]
                    for vn in pp.channel_variables:
                        if vn in dense:
                            nr=np.isnan(dense[vn]).sum()/dense[vn].size
                            nans.append(f'{vn}={nr:.1%}')
                    print(f'  SKIP: NaN too high ({" ".join(nans)}) threshold={pp.nan_max_ratio}')
    print(f'  First 5 files: {ok}/5 OK')
    break
