import os
import os.path
import numpy as np
import h5py
import torch
import cv2
import torch.utils.data as udata
import random
from utils import augment


class Dataset(udata.Dataset):
    #Dataset to load pairs of noisy/target(400) images
    def __init__(self, img_avg=1, patch_size=64, stride=32):
        ''' img_avg is the number of raw images averaged to create this image
            patch_size and stride determine the patching strategy
        '''
        
        super(Dataset, self).__init__()

        self.noisy_file_name = f'../../net_data/avg{img_avg}_{patch_size}_{stride}.h5'
        self.target_file_name = f'../../net_data/avg{400}_{patch_size}_{stride}.h5'


    def __len__(self):
        h5f_noisy = h5py.File(self.noisy_file_name, 'r')
        h5f_target = h5py.File(self.target_file_name, 'r')
        if h5f_noisy['data'].shape[0] != h5f_target['data'].shape[0]:
            raise NotImplemented('The noisy and the target files have different length.')
        else:
            return h5f_noisy['data'].shape[0]

    def __getitem__(self, index):
        h5f_noisy = h5py.File(self.noisy_file_name, 'r')
        h5f_target = h5py.File(self.target_file_name, 'r')

        data_noisy = np.array(h5f_noisy['data'][index,:])
        data_target = np.array(h5f_target['data'][index,:])
        
        h5f_noisy.close()
        h5f_target.close()
        
        if random.random() < 0.5:
            [data_noisy, data_target] = augment([data_noisy, data_target], True, True)
            data_noisy = data_noisy.copy()
            data_target = data_target.copy()
            
        return [torch.from_numpy(data_noisy), torch.from_numpy(data_target)]
