import sys,os,time
os.chdir(os.path.dirname(__file__))  # ensure CWD is trainer_app/
sys.path.insert(0,'.')
from trainer_app.data.preprocess import DataPreprocessor
from trainer_app.utils.config import load_config
from datetime import datetime as dt

c=load_config('trainer_app/config_default.yaml')
pp=DataPreprocessor(c)
dc=c['data']

# Verify data paths
nc=pp._index_nc_files(dc['dataset1_path'])
nd={f.stem.split('_')[-1][:8] for f in nc.values()}
print(f'NC: {len(nc)} files, dates={nd}')

storms=pp._parse_all_csvs(dc['dataset2_path'], nd)
print(f'Storms: {len(storms)}')

# Find first storm with windows
for sid in sorted(storms):
    rows=storms[sid]
    if len(rows)<48:
        continue
    w=pp._find_all_windows(rows, nc)
    if not w:
        continue
    
    win=w[0]
    print(f'\nStorm {sid}: {len(rows)}r, {len(w)}w')
    print(f'Window: {len(win["matched_rows"])} rows, {len(win["matched_files"])} files')
    print(f'Times: {[r["time"].strftime("%H%MZ") for r in win["matched_rows"][:5]]}...')
    print(f'NCs: {[f.name[:30] for f in win["matched_files"][:5]]}...')
    
    # Test extract_radar_crop on first file
    nf=win['matched_files'][0]
    row=win['matched_rows'][0]
    print(f'\nTesting _extract_radar_crop on {nf.name}')
    print(f'  center=({row["lon"]:.2f}, {row["lat"]:.2f}) half_deg={pp.spatial_degree/2}')
    
    # Check coordinate arrays
    print(f'  lon_arr range: [{pp.lon_arr.min():.2f}, {pp.lon_arr.max():.2f}] len={len(pp.lon_arr)}')
    print(f'  lat_arr range: [{pp.lat_arr.min():.2f}, {pp.lat_arr.max():.2f}] len={len(pp.lat_arr)}')
    
    half=pp.spatial_degree/2.0
    lon_mask = (pp.lon_arr >= row['lon']-half) & (pp.lon_arr <= row['lon']+half)
    lat_mask = (pp.lat_arr >= row['lat']-half) & (pp.lat_arr <= row['lat']+half)
    print(f'  lon_idx count: {lon_mask.sum()}, lat_idx count: {lat_mask.sum()}')
    
    if lon_mask.sum()<4 or lat_mask.sum()<4:
        print('  FAIL: crop too small')
    else:
        vol=pp._extract_radar_crop(nf, row['lon'], row['lat'], half, 29)
        if vol is None:
            print('  FAIL: extract returned None')
            # Try calling _reconstruct_crop directly
            from trainer_app.data.preprocess import _reconstruct_crop, _check_nan_excessive
            lon_idx = np.where(lon_mask)[0]
            lat_idx = np.where(lat_mask)[0]
            import numpy as np
            lon_sub=np.round(np.linspace(lon_idx[0], lon_idx[-1], pp.grid_size)).astype(int)
            lat_sub=np.round(np.linspace(lat_idx[0], lat_idx[-1], pp.grid_size)).astype(int)
            dense=_reconstruct_crop(nf, pp.channel_variables, lat_sub, lon_sub, 29)
            if not dense:
                print('  _reconstruct_crop returned empty dict!')
            else:
                print(f'  _reconstruct_crop returned {len(dense)} channels')
                for vn in pp.channel_variables:
                    if vn in dense:
                        data=dense[vn]
                        nan_ratio=np.isnan(data).sum()/data.size
                        print(f'    {vn}: shape={data.shape}, NaN={nan_ratio:.1%}')
                        if _check_nan_excessive(data, pp.nan_max_ratio):
                            print(f'      -> EXCESSIVE NaN (>{pp.nan_max_ratio:.0%})')
                    else:
                        print(f'    {vn}: MISSING')
        else:
            print(f'  SUCCESS: shape={vol.shape}')
    break
