import sys,os,time,glob; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from trainer_app.data.preprocess import DataPreprocessor; from trainer_app.utils.config import load_config
t0=time.time(); c=load_config('trainer_app/config_default.yaml'); pp=DataPreprocessor(c); dc=c['data']
print('ch=%d sm=%s nan=%.2f'%(dc['in_channels'],dc['use_storm_motion'],c['data']['nan_max_ratio']))
r=pp.validate_datasets(dc['dataset1_path'],dc['dataset2_path'])
print('ok=%s nc=%d csv=%d'%(r['valid'],r['nc_count'],r['csv_count']))
pp.process_events(dc['dataset1_path'],dc['dataset2_path'],
    progress_callback=lambda cur,tot,msg: print('[%d/%d]%s'%(cur,tot,msg)))
t=time.time()-t0; n=len(glob.glob(os.path.join(dc['processed_dir'],'*.npz')))
print('DONE %.1fmin %d samples'%(t/60,n))
