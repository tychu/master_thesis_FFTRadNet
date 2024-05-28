import scipy.io as sio
import numpy as np
import os
import pandas as pd 

####
# loading matlab simulation data
#####

variable_name = 'radar_data_cube_4d'
root_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data_DDA/'

# path to directory containing MATLAB files
#mat_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data_DDA/RD_cube/'
mat_dir = os.path.join(root_dir, 'RD_cube/')





filename = os.path.join(mat_dir, "RD_cube_1.mat")  # Assuming the files are named data_1.npy to data_100.npy
mat_data = sio.loadmat(filename)[variable_name]
print(mat_data.shape) # (512, 256, 1, 16)




num_sample = 100
num_tx = 2 
# compute the statistic
####
# Initialize an array to store the cube data
cube_data = np.zeros((num_sample, 512, 256, 2*num_tx)) # stack real and imaginary part 16*2=32

# Read the cube data from files
for i in range(num_sample):
    filename = os.path.join(mat_dir, f"RD_cube_{i+1}.mat")  # Assuming the files are named data_1.npy to data_100.npy
    mat_data = sio.loadmat(filename)[variable_name]
    combine_data_list = []

    for cube_idx in range(mat_data.shape[-1]):
        cube = mat_data[:, :, :, cube_idx]
        # combined the column of each matrix
        combined_data_cube = np.concatenate([cube[:, i, :] for i in range(cube.shape[1])], axis=1)
    
        # append the combined data to the list
        combine_data_list.append(combined_data_cube)

    final_data_3d = np.stack(combine_data_list, axis=-1)


    real_part = np.real(final_data_3d)
    imag_part = np.imag(final_data_3d)
    cube_data[i] = np.concatenate((real_part, imag_part), axis=-1)


# Calculate the mean for each cube along the last axis (axis=3)
cube_means = np.mean(cube_data, axis=(1, 2))
print('the mean for each cube along the last axis')
print(cube_means.shape)

# Calculate the mean value array across all cubes
overall_mean = np.mean(cube_means, axis=0)
overall_std = np.std(cube_means, axis=0)

# Print the mean value array
print("Overall mean value array:")
print(overall_mean)
print("Overall std value array:")
print(overall_std)  

# "input_mean":[ -8.46274006e-18, -5.41727036e-17, -7.39615617e-18,  5.01942238e-17,
#               -6.20152801e-17, -8.62591248e-18,  1.01465060e-16, -4.86264674e-17,
#               -4.51814150e-17, -6.57189147e-18,  4.08259039e-17, -5.26827388e-18,
#                8.77580343e-18, -2.32452946e-17, -1.11561151e-16,  6.38811920e-18,
#                2.46189787e-17, -5.85301122e-17, -1.28177904e-16,  2.94442205e-17,
#               -2.37451118e-17, -4.50876315e-17, -2.55969291e-17, -3.92099005e-17,
#                9.72401955e-17, -3.36127068e-17,  3.87788624e-17, -8.15591084e-19,
#               -3.04335550e-17, -4.58453533e-17, -1.67659940e-16, -3.54685899e-17],
#             "input_std":[1.76095659e-16, 4.00123426e-16, 3.55906949e-16, 6.10328838e-16,
#               5.20128158e-16, 8.49959918e-16, 6.57551127e-16, 6.31617451e-16,
#               5.40698390e-16, 4.31418320e-16, 3.90096184e-16, 3.00393421e-16,
#               2.70512522e-16, 3.09686380e-16, 1.16917808e-15, 2.66722735e-16,
#               7.33508754e-16, 6.12496627e-16, 7.81264407e-16, 2.57367173e-16,
#               3.60027903e-16, 8.45567290e-16, 4.57870189e-16, 6.46305960e-16,
#               6.16999406e-16, 3.58844855e-16, 7.15989759e-16, 3.90876416e-16,
#               4.29911861e-16, 9.91307111e-16, 8.17456481e-16, 3.31921791e-16],


## Ntx = 2
# Overall mean value array:
# [ 1.29290567e-17 -7.21034560e-17 -5.68744813e-17 -1.22157059e-18]
# Overall std value array:
# [7.88506969e-16 1.31454054e-15 5.46832158e-16 6.73096496e-16]

 ######################
 # convert the labels #
 ######################

# path to directory containing CVS files
#csv_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data_DDA/ground_truth/'
csv_dir = os.path.join(root_dir, 'ground_truth/')
# List MATLAB files in directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
print(len(csv_files))
# load and combine arrarys from MATLAB files
combined_label_df = pd.concat([pd.read_csv(os.path.join(csv_dir, f)) for f in csv_files], ignore_index=True)
# calculate mean and std
#columns_means = combined_label_df.mean()
#columns_stds = combined_label_df.std()



resolution= [0.201171875,0.2]
ranges = [512,896,1]
regression_layer = 2
INPUT_DIM = (ranges[0], ranges[1], ranges[2])
OUTPUT_DIM = (regression_layer + 1,INPUT_DIM[0] // 4 , INPUT_DIM[1] // 4 )


# range
#range_bin = int(np.clip(combined_label_df['range'].to_numpy()/resolution[0]/4,0, OUTPUT_DIM[1]-1))
#range_mod = combined_label_df['range'].to_numpy() - range_bin*resolution[0]*4
#combined_label_df['range_processed'] = range_mod

# Compute the 'range_bin' for each element  512//4=128=OUTPUT_DIM[1]=INPUT_DIM[0] // 4
combined_label_df['range_bin'] = (combined_label_df['range'] / resolution[0] / 4).clip(0, OUTPUT_DIM[1] - 1).astype(int)
combined_label_df['range_inter_compute'] = combined_label_df['range_bin'] * resolution[0] * 4
# Compute 'range_mod' for each element
combined_label_df['range_mod'] = combined_label_df['range'] - combined_label_df['range_bin'] * resolution[0] * 4


# ANgle and deg
# Compute the 'angle_bin' for each element 896//4=224=OUTPUT_DIM[2]=INPUT_DIM[1] // 4
#combined_label_df['angle_bin'] = (np.floor(combined_label_df['azimuth'].to_numpy() / resolution[1] / 4 + OUTPUT_DIM[2] / 2)).clip(0, OUTPUT_DIM[2] - 1).astype(int)
combined_label_df['angle_bin'] = (np.floor(combined_label_df['azimuth'].to_numpy() / resolution[1] / 4 + OUTPUT_DIM[2]/2)).clip(0, OUTPUT_DIM[2] - 1).astype(int)
combined_label_df['angle_inter_compute'] = (combined_label_df['angle_bin'] - OUTPUT_DIM[2]/2) * resolution[1] * 4

# Compute 'angle_mod' for each element
#combined_label_df['angle_mod'] = combined_label_df['azimuth'].to_numpy() - (combined_label_df['angle_bin'] - OUTPUT_DIM[2] / 2) * resolution[1] * 4
combined_label_df['angle_mod'] = combined_label_df['azimuth'].to_numpy() - (combined_label_df['angle_bin'] - OUTPUT_DIM[2] / 2) * resolution[1] * 4


columns_means = combined_label_df.mean()
columns_stds = combined_label_df.std()

print("labels overall mean value array:")
print(columns_means)
print("labels overall std value array:")
print(columns_stds) 

print(combined_label_df.iloc[:10, :])

# labels overall mean value array:
# range                   54.088002
# azimuth                 -0.197520
# doppler                128.501800
# range_bin               66.714600
# range_inter_compute     53.684405
# range_mod                0.403597
# angle_bin              111.382800
# angle_inter_compute     -0.493760
# angle_mod                0.296240
# dtype: float64
# labels overall std value array:
# range                  20.228495
# azimuth                51.349577
# doppler                 0.500047
# range_bin              25.144948
# range_inter_compute    20.233825
# range_mod               0.232915
# angle_bin              64.183357
# angle_inter_compute    51.346686
# angle_mod               0.226690
# dtype: float64
