import os
import numpy as np
import pandas as pd
import random

class DataProcessor:
    def __init__(self, root_dir):
        self.labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
        self.root_dir = root_dir
        
        # Extract unique IDs
        self.unique_ids = np.unique(self.labels[:, 0])
        
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



# Example usage
root_dir = '/imec/other/ruoyumsc/users/chu/data/'  # Your root directory
processor = DataProcessor(root_dir)

# # To get a sample
# index = 0  # Example index
# sample_data = processor.get_sample(index)
# print(sample_data)

# Example usage


# randomly select 50 files and check
file_indices = list(range(1, processor.get_length()))  
num_samples = 50
random_pairs = random.sample([(i, j) for i in file_indices for j in file_indices if i != j], num_samples)

results = []
for idx1, idx2 in random_pairs:
    are_equal, message = processor.compare_npz_files(idx1, idx2)
    results.append((idx1, idx2, are_equal, message))
    print(f'Comparing file{idx1:06d}.npz with file{idx2:06d}.npz: {message}')