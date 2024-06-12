import sys
sys.path.insert(0, '../')

from dataset.dataset import RADIal
from dataset.encoder import ra_encoder

from dataset.matlab_dataset import MATLAB

import numpy as np


geometry = {    "ranges": [512,896,1],
                "resolution": [0.6,0.2],
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
dataset = MATLAB(root_dir = '/imec/other/dl4ms/chu06/public/data/', 
                 statistics= None, 
                 encoder=enc.encode)


reg = []
m=0
s=0
for i in range(len(dataset)):
    print(i,len(dataset))
    radar_FFT, segmap,out_label,box_labels,image= dataset.__getitem__(i)

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

print('===  Regression  ====')
print('mean',np.mean(reg,axis=1))
print('std',np.std(reg,axis=1))

# ===  INPUT  ====
# mean [ 1.40113858e-09 -1.53072174e-08  1.22109185e-08  2.58594150e-10]
# std [2.72596013e+10 2.75136996e+10 2.75559747e+10 2.73023984e+10]
# ===  Regression  ====
# mean [1.13859902 0.32137931]
# std [2.06553848 0.69512452]