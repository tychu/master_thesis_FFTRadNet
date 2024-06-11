import os
import numpy as np
import pandas as pd
import random
import json
from dataset.dataset import RADIal
from dataset.matlab_dataset import MATLAB
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
import scipy.io

class DataProcessor:
    def __init__(self, root_dir):
        self.labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
        self.root_dir = root_dir
        
        # Extract unique IDs
        #self.unique_ids = np.unique(self.labels[:, 0])
        self.unique_ids, self.counts = np.unique(self.labels[:, 0], return_counts=True)
        
        # Initialize label dictionary
        self.label_dict = {}
        valid_ids = []
        missing_ids = []
        
        # Check each label for corresponding file
        for sample_id in self.unique_ids:
            radar_name = os.path.join(self.root_dir, 'radar_FFT', f"fft_{sample_id:06d}.npz")
            
            if os.path.exists(radar_name):
                if sample_id not in self.label_dict:
                    self.label_dict[sample_id] = []
                self.label_dict[sample_id].append(sample_id)
                if sample_id not in valid_ids:
                    valid_ids.append(sample_id)
            else:
                if sample_id not in missing_ids:
                    missing_ids.append(sample_id)
        
        # Update sample keys
        self.sample_keys = list(self.label_dict.keys())

        # Save missing IDs as a numpy array for later use
        self.missing_ids = np.array(missing_ids)
        np.save(os.path.join(self.root_dir, 'labels_missing_ids.npy'), self.missing_ids)
        
        # Print or log missing IDs
        if missing_ids:
            print(f"There are IDs do not have corresponding 'fft' npz files, already save those IDs to ", root_dir)
        else:
            print("All IDs have corresponding 'fft' npz files.")
    
    def get_sample(self, index):
        # Get the sample ID
        sample_id = self.sample_keys[index]
        
        # Load the corresponding file
        radar_name = os.path.join(self.root_dir, 'radar_FFT', f"fft_{sample_id:06d}.npz")
        input_data = np.load(radar_name, allow_pickle=True)
        
        return input_data
    
    def get_length(self):
        
        return len(self.sample_keys)
    
    def compare_npz_files(self, idx1, idx2):
        # Load the .npz files
        # Get the sample ID
        sample_id1 = self.sample_keys[idx1]
        sample_id2 = self.sample_keys[idx2]
        
        # Load the corresponding file
        radar_name1 = os.path.join(self.root_dir, 'radar_FFT', f"fft_{sample_id1:06d}.npz")
        input_data1 = np.load(radar_name1, allow_pickle=True)
        radar_name2 = os.path.join(self.root_dir, 'radar_FFT', f"fft_{sample_id2:06d}.npz")
        input_data2 = np.load(radar_name2, allow_pickle=True)
        
        # Check if both files have the same keys
        keys1 = set(input_data1.files)
        keys2 = set(input_data2.files)
        
        if keys1 != keys2:
            return False, "Keys do not match"
        
        # Compare the arrays for each key
        for key in keys1:
            array1 = input_data1[key]
            array2 = input_data2[key]

            #print('file 1: ', array1.shape)
            #print('file 2: ', array2.shape)
            

            
            if not np.array_equal(array1, array2):
                return False, f"Arrays for key '{key}' do not match"
        
        return True, "Files are identical"
    
    def save_images_for_unique_ids(self):
            # Extract IDs that appear exactly two times
            ids_with_two_occurrences = self.unique_ids[self.counts == 2]
            #print(ids_with_two_occurrences.shape) #1330

            # Directory to save the images
            plot_dir = os.path.join(self.root_dir, 'plot')
            os.makedirs(plot_dir, exist_ok=True)

            # # Loop through IDs that appear exactly two times and save their images
            # for sample_id in ids_with_two_occurrences:
            #     img_name = os.path.join(self.root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
            #     try:
            #         image = np.asarray(Image.open(img_name))
            #         save_path = os.path.join(plot_dir, "image_{:06d}.jpg".format(sample_id))
            #         Image.fromarray(image).save(save_path)
            #         print(f"Saved image for ID {sample_id} to {save_path}")
            #     except FileNotFoundError:
            #         print(f"Image for ID {sample_id} not found at {img_name}")

            print("Processing completed.")


class DataComparator:
    def __init__(self, sample_id):

        self.radar_name = os.path.join('/imec/other/ruoyumsc/users/chu/data/radar_FFT/',"fft_{:06d}.npz".format(sample_id))

        self.mat_id_file = os.path.join('/imec/other/ruoyumsc/users/chu/data/radial_targets.mat')
        self.mat_name = os.path.join('/imec/other/ruoyumsc/users/chu/matlab-radar-automotive/simulation_data_radial/RD_cube/',"RD_cube_{:06d}.mat".format(sample_id))
        self.data1 = self.read_npz_data(sample_id)
        #self.data2 = self.read_matlab_data(sample_id)

    def read_npz_data(self, sample_id):
        input_mean = [-2.6244e-03, -2.1335e-01,  1.8789e-02, -1.4427e+00, -3.7618e-01,
                1.3594e+00, -2.2987e-01,  1.2244e-01,  1.7359e+00, -6.5345e-01,
                3.7976e-01,  5.5521e+00,  7.7462e-01, -1.5589e+00, -7.2473e-01,
                1.5182e+00, -3.7189e-01, -8.8332e-02, -1.6194e-01,  1.0984e+00,
                9.9929e-01, -1.0495e+00,  1.9972e+00,  9.2869e-01,  1.8991e+00,
               -2.3772e-01,  2.0000e+00,  7.7737e-01,  1.3239e+00,  1.1817e+00,
               -6.9696e-01,  4.4288e-01]
        input_std = [20775.3809, 23085.5000, 23017.6387, 14548.6357, 32133.5547, 28838.8047,
                27195.8945, 33103.7148, 32181.5273, 35022.1797, 31259.1895, 36684.6133,
                33552.9258, 25958.7539, 29532.6230, 32646.8984, 20728.3320, 23160.8828,
                23069.0449, 14915.9053, 32149.6172, 28958.5840, 27210.8652, 33005.6602,
                31905.9336, 35124.9180, 31258.4316, 31086.0273, 33628.5352, 25950.2363,
                29445.2598, 32885.7422]
        try:
            
            npzdata = np.load(self.radar_name,allow_pickle=True)

            # Iterate over each array in the .npz file 
            for array_name in npzdata.files:
                # Extract the array
                data = npzdata[array_name]
                radar_FFT = np.concatenate([data.real,data.imag],axis=2)

                for i in range(len(input_mean)):
                    radar_FFT[...,i] -= input_mean[i]
                    radar_FFT[...,i] /= input_std[i] 

            print(f"Successfully read data from {self.radar_name}")
            return radar_FFT
        except Exception as e:
            print(f"Error reading {self.radar_name}: {e}")
            return None
        
    def find_matdata_id_by_numSample(self, sample_id):
        try:
            # Load the .mat file
            
            mat_data = scipy.io.loadmat(self.mat_id_file)
            data_dict = mat_data.get('dataDictionary', {})
            field_names = list(data_dict.keys())

            nameS = f'numSample_{sample_id}'
            j = field_names.index(nameS)
            retrieved_data = data_dict[nameS]
            return j, retrieved_data
                    
        except Exception as e:
            print(f"Error reading {self.mat_name}: {e}")
            return None
        

    def read_matlab_data(self, sample_id):
        input_mean = [ 1.11419230e-17,  3.06145276e-18, -1.24560895e-18,  7.19768561e-19,
                4.51463008e-18,  1.00182712e-18,  6.77008740e-19,  4.65800904e-18,
                -2.52098826e-18, -2.42829490e-18, -4.26560142e-18, -9.57608009e-19,
                -1.21016352e-17,  9.12583069e-18, -9.91278908e-19,-1.13778882e-18,
                -3.78170450e-18,  1.52190633e-18,  6.37910205e-18,  9.05723027e-18,
                2.86451638e-18, -5.05282089e-18, -5.73682605e-18, -9.56477982e-19,
                7.37387625e-18,  1.91439772e-18,  1.36774925e-18,  1.81696604e-19,
                -4.63171449e-18, -7.42305385e-18,  8.35017256e-18, -2.69604888e-18]
        input_std = [232.72035485, 233.15903648, 233.23005056, 232.5184981,  233.0206286,
                232.56749406, 232.094457,   232.56056963, 232.98804308, 232.97788545,
                233.09726566, 233.14234567, 232.58432258, 233.18640591, 232.27603896,
                232.58394276, 232.6052493,  232.62227468, 232.35970884, 233.00847115,
                232.54399026, 233.0886895,  233.45127264, 232.97472146, 232.36188323,
                232.60535444, 232.62475202, 232.45358907, 232.6120579, 232.2788462,
                233.18009515, 233.00165888]
        try:
                
            # RD cube
            data = sio.loadmat(self.mat_name)['radar_data_cube_4d']
            mat_input_3d = data[:, :, 0, :]
            radar_FFT = np.concatenate([mat_input_3d.real,mat_input_3d.imag],axis=2)

            for i in range(len(input_mean)):
                radar_FFT[...,i] -= input_mean[i]
                radar_FFT[...,i] /= input_std[i]

            print(f"Successfully read data from {self.mat_name}")
            return radar_FFT
        except Exception as e:
            print(f"Error reading {self.mat_name}: {e}")
            return None



    def rd_plot(self): 
        #rd_plot(inputs[0, 0, :, :]+inputs[0, 16, :, :]*1j)
        directory = './plot/'
        plot_input1 = self.data1[0, 0, :, :]+self.data1[0, 16, :, :]*1j
        plot_input2 = self.data2[0, 0, :, :]+self.data2[0, 16, :, :]*1j
        plot_1 = 20* np.log10(np.abs(plot_input1))
        plot_2 = 20* np.log10(np.abs(plot_input2))
        

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display the first image in the first subplot
        axs[0].imshow(plot_1, cmap='magma', interpolation='none')
        axs[0].set_title('Image 1')
        axs[0].axis('off')  # Turn off axis
        
        # Display the second image in the second subplot
        axs[1].imshow(plot_2, cmap='magma', interpolation='none')
        axs[1].set_title('Image 2')
        axs[1].axis('off')  # Turn off axis
        
        # Save the plot with the specified filename in the specified directory
        filepath = os.path.join(directory, 'RD_input_comparison.png')
        plt.savefig(filepath)
        print(f'Plot saved to {filepath}')
        
        # Close the plot to free up memory
        plt.close()




            


    

# Example usage
root_dir = '/imec/other/ruoyumsc/users/chu/data/'  # Your root directory
processor = DataProcessor(root_dir)

# # To get a sample
# index = 0  # Example index
# sample_data = processor.get_sample(index)
# print(sample_data)


#processor.save_images_for_unique_ids()
# Example usage


# # randomly select 50 files and check
# file_indices = list(range(1, processor.get_length()))  
# num_samples = 50
# random_pairs = random.sample([(i, j) for i in file_indices for j in file_indices if i != j], num_samples)

# results = []
# for idx1, idx2 in random_pairs:
#     are_equal, message = processor.compare_npz_files(idx1, idx2)
#     results.append((idx1, idx2, are_equal, message))
#     print(f'Comparing file{idx1:06d}.npz with file{idx2:06d}.npz: {message}')




# ## for comparing RD input
comparator = DataComparator(sample_id=870)
idx, data = comparator.find_matdata_id_by_numSample(870)
print('the RD idx: ', idx)
#comparator.rd_plot()