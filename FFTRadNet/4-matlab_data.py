import scipy.io as sio
import numpy as np
import os
import pandas as pd 

####
# loading matlab simulation data
#####

variable_name = 'radar_data_cube_4d'

# path to directory containing MATLAB files
mat_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data/RD_cube/'
# List MATLAB files in directory
#mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]




filename = os.path.join(mat_dir, "RD_cube_1.mat")  # Assuming the files are named data_1.npy to data_100.npy
mat_data = sio.loadmat(filename)[variable_name]
print(mat_data.shape) # (512, 64, 4, 16)





 ###
# compute the statistic
####
# Initialize an array to store the cube data
cube_data = np.zeros((100, 512, 64*4, 32))

# Read the cube data from files
for i in range(100):
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

##

# Overall mean value array:
# [-3.20070034e-18  6.28803379e-19  1.45567694e-18 -6.02545357e-19
#  -4.00843096e-18  2.36938832e-18 -3.72287921e-19 -3.03427530e-18
#  -4.27799072e-18 -3.18619913e-19  1.06414443e-18 -1.06536416e-18
#   2.49481696e-18  1.79449012e-18  4.55839251e-18  6.71527721e-19
#  -2.62241400e-19 -1.64934255e-19  6.81963166e-19 -9.10729825e-20
#   5.78692910e-19  3.52907807e-19 -7.71274320e-19  3.02627931e-19
#   7.48099499e-20  7.99599102e-21 -4.92905413e-19  6.46455545e-20
#   2.37440276e-19 -1.23666810e-18  5.59922659e-19  1.47315970e-18]
# Overall std value array:
# [2.61215742e-17 1.61202613e-17 1.62644270e-17 2.79939372e-17
#  3.71814903e-17 3.08226268e-17 1.36271104e-17 3.20436755e-17
#  3.74390487e-17 1.36662445e-17 1.80110335e-17 3.57482383e-17
#  1.68523141e-17 3.43132773e-17 2.54901448e-17 1.68597672e-17
#  4.42054991e-18 4.45661829e-18 3.84662020e-18 5.27715641e-18
#  5.10650861e-18 6.38152684e-18 8.73501397e-18 5.85429708e-18
#  6.31119188e-18 4.56110461e-18 8.32217987e-18 6.55442188e-18
#  4.69238888e-18 8.91440370e-18 4.31984934e-18 8.27720249e-18]



 ######################
 # convert the labels #
 ######################

 # path to directory containing CVS files
csv_dir = '/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data/ground_truth/'
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

# Compute the 'range_bin' for each element
combined_label_df['range_bin'] = (combined_label_df['range'] / resolution[0] / 4).clip(0, OUTPUT_DIM[1] - 1).astype(int)

# Compute 'range_mod' for each element
combined_label_df['range_mod'] = combined_label_df['range'] - combined_label_df['range_bin'] * resolution[0] * 4


# ANgle and deg
# Compute the 'angle_bin' for each element
combined_label_df['angle_bin'] = (np.floor(combined_label_df['azimuth'].to_numpy() / resolution[1] / 4 + OUTPUT_DIM[2] / 2)).clip(0, OUTPUT_DIM[2] - 1).astype(int)

# Compute 'angle_mod' for each element
combined_label_df['angle_mod'] = combined_label_df['azimuth'].to_numpy() - (combined_label_df['angle_bin'] - OUTPUT_DIM[2] / 2) * resolution[1] * 4


columns_means = combined_label_df.mean()
columns_stds = combined_label_df.std()

print("labels overall mean value array:")
print(columns_means)
print("labels overall std value array:")
print(columns_stds) 
 
# labels overall mean value array:
# range              513.00000
# azimuth             63.10198
# doppler            128.50500
# range_processed      5.00000
# angle_processed    279.12198
# dtype: float64
# labels overall std value array:
# range               0.000000
# azimuth            19.397043
# doppler             0.501230
# range_processed     0.000000
# angle_processed    53.276413

