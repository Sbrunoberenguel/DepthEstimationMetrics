import os
import cv2
import numpy as np
import scipy.io as sio

import torch
import torch.utils.data as data

class NYUDataset(data.Dataset):
    def __init__(self, root_dir, set='train',
                 flip=False, rotate=False, color=False,
                 scale =1, prop = 0.8):

        np.random.seed(1994)
        self.d_max = 10.0
        self.scale = scale
        if set=='valid' or set=='train':
            set_folder = 'train'
        else:
            set_folder='test'
        self.img_dir = os.path.join(root_dir,set_folder,'img')
        self.dep_dir = os.path.join(root_dir,set_folder,'depth')

        self.ref_img = os.listdir(self.img_dir)
        length = len(self.ref_img)

        if set=='train':
            self.img_fnames = self.ref_img[:int(prop*length)]
        elif set=='valid':
            self.img_fnames = self.ref_img[int((prop-1)*length):]
        elif set=='test':
            self.img_fnames = self.ref_img
        
        self.dep_fnames = ['%s_d.mat' % fname[:-4] for fname in self.img_fnames]
        
        self.flip = flip
        self.color = color

    def __len__(self):
        return len(self.img_fnames) 


    def __getitem__(self, idx):
        H, W = int(480//self.scale), int(640//self.scale)

        crop_x,crop_y = np.random.randint(4),np.random.randint(10)

        # # Path to RGB image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])

        # # Path to ground truth depth map
        dep_path = os.path.join(self.dep_dir,
                                self.dep_fnames[idx])
        #OpenCv opens in BGR by default
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)/255.
        img = cv2.resize(img,(W,H),cv2.INTER_CUBIC)
        
        dep = sio.loadmat(dep_path)['depth']
        dep = cv2.resize(dep.astype(np.float32),(W,H))

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            dep = np.flip(dep, axis=1)

        #Image cropping -> center crop
        crop_x,crop_y = 2,5
        img = img[crop_y:crop_y+630,crop_x:crop_x+476] #Crop for 14 divisible matrix for Dino/ViT encoder
        dep = dep[crop_y:crop_y+630,crop_x:crop_x+476] #Crop for 14 divisible matrix for Dino/ViT encoder

        #To allow only data in the range of the dataset [0,d:max]
        #  We set a mask that defines this data and the clamp the
        #  ground truth depth map to fairly compare with the prediction
        dep_mask = np.logical_and(dep <= self.d_max, dep > 0.)

        #Convert to Tensor
        x = torch.FloatTensor(img.transpose([2,0,1]).copy())  #channels,height,width
        depth = torch.FloatTensor(dep.copy())
        depth = torch.clamp(depth,0,self.d_max) 
        mask = torch.FloatTensor(dep_mask.copy())

        # #Save ground truth into a dict
        gt = {}
        gt['name'] = self.img_fnames[idx]
        gt['Depth'] = depth.unsqueeze(0)
        gt['Mask'] = mask
        return x, gt
        