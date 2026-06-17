import numpy as np,glob
fs=sorted(glob.glob('../processed_v3/*.npz'))
d=np.load(fs[0],allow_pickle=True)
ks=list(d.keys())
print('Keys:',ks)
print('input:',d['input'].shape)
print('target:',d['target'].shape)
print('channels:',d['channel_names'].tolist())
# Check storm motion channels
ch=d['channel_names'].tolist()
print('Has storm_u:','storm_u' in ch)
print('Has storm_v:','storm_v' in ch)
print('Files:',len(fs))
