import sys
sys.path.insert(0, '../')

from dataset.dataset import RADIal
from dataset.encoder import ra_encoder

from dataset.radial_mat_dataset import RADIalMAT
from dataset.matlab_dataset import MATLAB

import numpy as np


geometry = {    "ranges": [512,896,1],
                "resolution": [0.201171875,0.2],
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
dataset = MATLAB(root_dir = '/imec/other/dl4ms/chu06/data/', 
                 statistics= None, 
                 encoder=enc.encode)


reg = []
m=0
s=0
for i in range(len(dataset)):
    print(i,len(dataset))
    radar_FFT, segmap,out_label,box_labels,image= dataset.__getitem__(i)
    
    
    # List to store indices of files that have errors
    # complete_indices = []
    
    # try:
    #     radar_FFT, out_label, index = dataset.__getitem__(i)
    #     complete_indices.append(index)
    # except Exception as e:
    #     print(f'An unexpected error occurred while loading {i}: {e}')
    # print('radar_FFT : ', radar_FFT.shape)
    
    
    data = np.reshape(radar_FFT,(512*256,32))
    #data = np.reshape(radar_FFT,(512*256,4))

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

# Save the error_indices list to a .npy file
#np.save('/imec/other/ruoyumsc/users/chu/data/complete_radial_mat_indices.npy', complete_indices)


# ===  INPUT  ====
# mean [ 7.06430047e-11  1.24442498e-11 -1.77583329e-10 -2.10502022e-10
#   2.53292142e-10 -9.00746062e-11  1.89241358e-10  3.47757342e-11
#  -2.41579764e-10 -1.48665962e-10 -2.56584702e-11 -4.01409430e-10
#   3.70359463e-10  1.46297076e-10  2.17522825e-10  3.12860754e-11
#  -9.49378829e-11 -3.58693901e-10  2.87186400e-10  2.45252031e-10
#  -3.51019528e-11 -1.41795214e-10  3.06161702e-10 -3.36731208e-10
#  -3.38266623e-12  3.67160507e-10 -2.35719019e-10  1.19181197e-10
#   6.82791067e-11  2.81842940e-10 -6.16829635e-11  4.20524180e-10]
# std [7.86189883e+08 7.94907570e+08 7.81678977e+08 7.58260704e+08
#  7.36273896e+08 7.14339462e+08 7.17958061e+08 7.36860310e+08
#  7.57831686e+08 7.83628901e+08 7.91257755e+08 7.83877930e+08
#  7.67160704e+08 7.40062059e+08 7.17983772e+08 7.12081940e+08
#  7.28441112e+08 7.17650741e+08 7.34563869e+08 7.61189445e+08
#  7.80980634e+08 7.98688596e+08 7.95602689e+08 7.80848717e+08
#  7.61336662e+08 7.31970972e+08 7.21163579e+08 7.31012584e+08
#  7.51764771e+08 7.78175229e+08 7.95990660e+08 8.00725665e+08]
# ===  Regression  ====
# mean [2.3188366  0.27666667]
# std [14.61668122  0.69657242]



# ===  INPUT  ====
# mean [-8.12419793e-18 -2.99906584e-17 -3.15557042e-17  2.21559424e-17]
# std [208.89973699 208.62195008 203.17029074 203.7470282 ]
# ===  Regression  ====
# mean [0.4137569  0.28762542]
# std [0.69949276 0.69366993]