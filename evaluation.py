import os
import cv2
import argparse
import warnings
import numpy as np
import scipy.io as sio
from tqdm import trange
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


import loader_NYU as dataset
import metrics

warnings.filterwarnings('ignore')


class VanillaDataset(data.Dataset):
    def __init__(self, root_dir,):

        np.random.seed(1994)
        self.d_max = 16.0
        self.img_dir = os.path.join(root_dir,'img')
        self.dep_dir = os.path.join(root_dir,'depth')
        self.img_fnames = os.listdir(self.img_dir)        
        self.dep_fnames = ['%s_d.mat' % fname[:-4] for fname in self.img_fnames]

    def __len__(self):
        return len(self.img_fnames) 

    def __getitem__(self, idx):
        H, W = int(480), int(640)
        
        # # Path to RGB image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])

        # # Path to ground truth depth map
        dep_path = os.path.join(self.dep_dir,
                                self.dep_fnames[idx])
        
        #OpenCV opens in BGR by default
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)/255.
        img = cv2.resize(img,(W,H),cv2.INTER_CUBIC)
        #Ground truth depth maps stored as *.mat. Change here with your data
        dep = sio.loadmat(dep_path)['depth']
        dep = cv2.resize(dep.astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)

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
        


def main(args):

    device = torch.device('cpu' if args.no_cuda else 'cuda')
    
    if 'NYU' in args.root_dir:
        #Example with a custom dataset, in this case our Dataloader for the NYU dataset v2
        dataset_test = dataset.NYUDataset(
                                root_dir=args.root_dir, scale=1,
                                flip=False, rotate=False, color=False,
                                set='test')        
    else:
        print('Load your own dataloader\n')
        print('DEFAULT: Vanilla dataloader')
        dataset_test = VanillaDataset(root_dir = args.root_dir)


    loader_test = DataLoader(dataset_test, batch_size=1,
                            shuffle=False, drop_last=False,
                            num_workers=2)

    '''
    Load your own model HERE such as:
    import your_model as model
    And load the pre-trained weights
    '''

    import main_model as model
    net,_ = model.load_weigths(args)
    
    ####################
    net.to(device)
    net.eval()

    # Set up depth metrics
    depth_results = metrics.DepthMetric()
    
    # Inferencing   
    iterator_test = iter(loader_test)

    for i in trange(len(iterator_test)):
        x, gt = next(iterator_test)
        mask = gt['Mask']
        bs,ch,H,W = x.shape

        with torch.no_grad():
            output = net(x.to(device))
            output = torch.clamp(output,0,dataset_test.d_max) #We clamp the output data to the maximum of the dataset
            gt_dep = gt['Depth'].reshape(bs,H,W)
            
            #Compute the metrics for the current image and save them in a list
            depth_results.update_metrics(output.squeeze(1),gt_dep.squeeze(1).to(device),mask.to(device))

        #Save the outpur depth map
        if args.save_output:
            os.makedirs(os.path.join(args.out_dir,'depth'),exist_ok=True)
            os.makedirs(os.path.join(args.out_dir,'depthmap'),exist_ok=True)
            #Output management
            depth = output.squeeze(1).cpu().numpy()*mask.numpy()

            #Save coded data
            name = gt['name'][0]
            np.save(os.path.join(args.out_dir,'depth',name[:-4]+'.npy'),depth)
            plt.figure(0)
            plt.imshow(depth.transpose([1,2,0]),vmin=0.,vmax=10.)
            plt.savefig(os.path.join(args.out_dir,'depthmap',name[:-4]+'_dep.png'))
            plt.figure(1)
            plt.imshow(gt_dep.numpy().transpose([1,2,0]),vmin=0.,vmax=10.)
            plt.savefig(os.path.join(args.out_dir,'depthmap',name[:-4]+'_gt.png'))


    #Compute the average metrics over all the dataset and plot them
    out_mse,out_mae,out_rmse,out_rmselog,out_mare,out_d1,out_d2,out_d3 = depth_results.compute_metrics()
    print('Depth Estimation:')
    print('MAE: %.4f' %(out_mae))
    print('MSE: %.4f' %(out_mse))
    print('RMSE_linear: %.4f' %(out_rmse))
    print('RMSE_log: %.4f' %(out_rmselog))
    print('MARE: %.4f' %(out_mare))
    print('Threshold_1_25_1: %.4f' %(out_d1))
    print('Threshold_1_25_2: %.4f' %(out_d2))
    print('Threshold_1_25_3: %.4f' %(out_d3))
    print('-'*20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=False, default=None,
                        help='Path to load saved checkpoint.')
    parser.add_argument('--root_dir', required=False, default='./data',
                        help='Path to the dataset OR to your set of data.')
    parser.add_argument('--out_dir',  required=False, default='./Results',
                        help='Path to the output directory to save depth predictions.')
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    #PARSER END#
    main(args)
