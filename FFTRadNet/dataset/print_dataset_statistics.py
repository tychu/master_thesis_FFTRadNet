import sys
sys.path.insert(0, '../')

from dataset.dataset import RADIal
from dataset.encoder import ra_encoder

from dataset.matlab_dataset import MATLAB

import numpy as np


geometry = {    "ranges": [512,896,1],
                "resolution": [0.673828125,0.2],
                "size": 3}

statistics = {  "input_mean":np.zeros(32),
                "input_std":np.ones(32),
                "reg_mean":np.zeros(3),
                "reg_std":np.ones(3)}
    
enc = ra_encoder(geometry = geometry, 
                    statistics = statistics,
                    regression_layer = 2)

# dataset = RADIal(root_dir = '/media/julien/shared_data/radar_v1/RADIal/',
#                        statistics=None,
#                        encoder=enc.encode,
#                        difficult=True)
dataset = MATLAB(root_dir = '/imec/other/dl4ms/chu06/public/data/', #replace with the correct directory
                 folder_dir = "simulation_sequential_rw_data_DDA_3targets_rx2_10000_doppler32/",
                 statistics= None, 
                 encoder=enc.encode)


reg = []
m=0
s=0
for i in range(len(dataset)):
    print(i,len(dataset))
    #radar_FFT, segmap,out_label,box_labels,image= dataset.__getitem__(i)
    radar_FFT,out_label,box_labels= dataset.__getitem__(i)

    print('radar_FFT : ', radar_FFT.shape)
    
    #data = np.reshape(radar_FFT,(512*256,32))
    data = np.reshape(radar_FFT,(512*256,4))
    
    m += data.mean(axis=0)
    s += data.std(axis=0)

    idy,idx = np.where(out_label[0]>0)
    
    reg.append(out_label[1:,idy,idx])

reg = np.concatenate(reg,axis=1)
    
print('===  INPUT  ====')
print('mean',m/len(dataset))
print('std',s/len(dataset))
# Print mean and std separated by commas
print('mean:', ', '.join(map(str, m/len(dataset))))
print('std:', ', '.join(map(str, s/len(dataset))))

print('===  Regression  ====')
print('mean',np.mean(reg,axis=1))
print('std',np.std(reg,axis=1))
# Print mean and std separated by commas
print('mean:', ', '.join(map(str, np.mean(reg,axis=1))))
print('std:', ', '.join(map(str, np.std(reg,axis=1))))
