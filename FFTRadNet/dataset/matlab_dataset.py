from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image
import scipy.io as sio

class MATLAB(Dataset):

    def __init__(self, root_dir,statistics=None,encoder=None,difficult=False):

        self.root_dir = root_dir
        self.statistics = statistics
        self.encoder = encoder
        
        #self.labels = pd.read_csv(os.path.join(root_dir,'ground_truth/ground_truth.csv')).to_numpy()
        #csv_file = os.path.join(self.root_dir, 'ground_truth/ground_truth.csv')
        #self.labels = pd.read_csv(csv_file).to_numpy()
        #print('check labels length')
        #print(len(self.labels))

        self.mat_files = [f for f in os.listdir(os.path.join(self.root_dir, 'simulation_data_DDA/RD_cube/')) if f.endswith('.mat')]
        self.csv_files = [f for f in os.listdir(os.path.join(self.root_dir, 'simulation_data_DDA/ground_truth/')) if f.endswith('.csv')]
        
       
        # Keeps only easy samples
        #if(difficult==False):
        #    ids_filters=[]
        #    ids = np.where( self.labels[:, -1] == 0)[0]
        #    ids_filters.append(ids)
        #    ids_filters = np.unique(np.concatenate(ids_filters))
        #    self.labels = self.labels[ids_filters]


        # Gather each input entries by their sample id
        #self.unique_ids = np.unique(self.labels[:,0])
        #self.label_dict = {}
        #for i,ids in enumerate(self.unique_ids):
        #    sample_ids = np.where(self.labels[:,0]==ids)[0]
        #    self.label_dict[ids]=sample_ids
        #self.sample_keys = list(self.label_dict.keys())
        
        # for segmap
        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))


    def __len__(self):
        #return len(self.label_dict)
        return len(self.mat_files)
        #return len(self.labels)

    #def __getitem__(self, index):
    def __getitem__(self, idx):
        
        # Get the sample id
        #sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        #entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        #box_labels = self.labels[entries_indexes]
        
        #box_labels = self.labels

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        #box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 

        
        

        ######################
        #  Encode the labels #
        ######################
        
        # load lables
        csv_file = os.path.join(self.root_dir, 'simulation_data_DDA/ground_truth/', self.csv_files[idx])
        box_labels = pd.read_csv(csv_file).to_numpy()
        #print(box_labels.shape)

        
        out_label=[]
        if(self.encoder!=None):
            out_label = self.encoder(box_labels).copy()
        #out_label = box_labels      

        # Read the Radar FFT data
        #radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
        mat_file = os.path.join(self.root_dir, 'simulation_data_DDA/RD_cube/', self.mat_files[idx])
        mat_input_4d = sio.loadmat(mat_file)['radar_data_cube_4d']
        mat_input_3d = mat_input_4d[:, :, 0, :]

        # #### re-arrange the RD spectrum, one RD specturm per Rx
        # ## combine the Tx
        
        # combine_data_list = []

        # for cube_idx in range(mat_input_4d.shape[-1]):
        #     cube_data = mat_input_4d[:, :, :, cube_idx]
        #     #print('cube_data shape', cube_data.shape)
        #     # combined the column of each matrix
        #     combined_data_cube = np.concatenate([cube_data[:, i, :] for i in range(cube_data.shape[1])], axis=1)
        #     #rint('combined data cube', combined_data_cube.shape)
        #     # append the combined data to the list
        #     combine_data_list.append(combined_data_cube)

        # mat_input_3d = np.stack(combine_data_list, axis=-1)
        

        radar_FFT = np.concatenate([mat_input_3d.real,mat_input_3d.imag],axis=2)
        #print("Shape of concatenated array:", radar_FFT.shape)

        if(self.statistics is not None):
            for i in range(len(self.statistics['input_mean'])):
                radar_FFT[...,i] -= self.statistics['input_mean'][i]
                radar_FFT[...,i] /= self.statistics['input_std'][i]
        #if(sta_mean is not None):
        #    for i in range(len(sta_mean)):
        #        radar_FFT[...,i] -= sta_mean[i]
        #        radar_FFT[...,i] /= sta_std[i]

        # Read the segmentation map
        #segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
        segmap_name = os.path.join(self.root_dir,'radar_Freespace/',"freespace.png")
        segmap = Image.open(segmap_name) # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap))==255

        # Read the camera image
        #img_name = os.path.join(self.root_dir,'camera',"image_{:06d}.jpg".format(sample_id))
        img_name = os.path.join(self.root_dir,'camera/',"image_001010.jpg")
        image = np.asarray(Image.open(img_name))

        return radar_FFT, segmap,out_label,box_labels,image