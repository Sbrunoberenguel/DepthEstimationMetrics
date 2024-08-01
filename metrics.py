import torch
import numpy as np

'''
Depth metrics definition.
All metrics include a 'mask'. This mask is used to compute the metric
  in those cases where ground truth data is available and leave out
  pixels where we do not have information to comapre with.
The computation of the mask is made in the evaluation code, when
  loading the input image and ground truth depth map.
'''



class DepthMetric():
    def __init__(self):
        self.mse = []
        self.mae = []
        self.rmse = []
        self.rmse_log = []
        self.mare = []
        self.deltas = {'d1':[],'d2':[],'d3':[]}

    def update_metrics(self,pred_depth,gt_depth,mask):
        self.deltas['d1'].append(self.delta_inlier_ratio(pred_depth,gt_depth,mask,1))
        self.deltas['d2'].append(self.delta_inlier_ratio(pred_depth,gt_depth,mask,2))
        self.deltas['d3'].append(self.delta_inlier_ratio(pred_depth,gt_depth,mask,3))
        self.mare.append(self.abs_rel_error(pred_depth,gt_depth,mask))
        self.rmse.append(self.lin_rms_sq_error(pred_depth,gt_depth,mask))
        self.rmse_log.append(self.log_rms_sq_error(pred_depth,gt_depth,mask))
        self.mse.append(self.sq_error(pred_depth,gt_depth,mask))
        self.mae.append(self.abs_error(pred_depth,gt_depth,mask))
        
    def compute_metrics(self):
        out_mse = np.mean(torch.FloatTensor(self.mse).cpu().numpy())
        out_mae = np.mean(torch.FloatTensor(self.mae).cpu().numpy())
        out_rmse = np.mean(torch.FloatTensor(self.rmse).cpu().numpy())
        out_rmselog = np.mean(torch.FloatTensor(self.rmse_log).cpu().numpy())
        out_mare = np.mean(torch.FloatTensor(self.mare).cpu().numpy())
        out_d1 = np.mean(torch.FloatTensor(self.deltas['d1']).cpu().numpy())
        out_d2 = np.mean(torch.FloatTensor(self.deltas['d2']).cpu().numpy())
        out_d3 = np.mean(torch.FloatTensor(self.deltas['d3']).cpu().numpy())
        return out_mse,out_mae,out_rmse,out_rmselog,out_mare,out_d1,out_d2,out_d3

    def abs_error(self, pred, gt, mask):
        '''Compute mean absolute error'''
        return ((pred[mask>1e-7] - gt[mask>1e-7]).abs()).mean()

    def sq_error(self,pred, gt, mask):
        '''Compute mean squared error'''
        return ((pred[mask>1e-7] - gt[mask>1e-7])**2).mean()

    def lin_rms_sq_error(self, pred, gt, mask):
        '''Compute root mean squared error'''
        return torch.sqrt(((pred[mask>1e-7] - gt[mask>1e-7]) ** 2).mean())

    def log_rms_sq_error(self, pred, gt, mask):
        mask = (mask > 1e-7) & (pred > 1e-7) & (gt > 1e-7) # Compute a mask to avoid Log(0) = NaN
        '''Compute root mean squared error of logarithmic values'''
        return torch.sqrt(((pred[mask].log() - gt[mask].log()) ** 2).mean())

    def abs_rel_error(self, pred, gt, mask):
        '''Compute mean absolute relative error'''
        return ((pred[mask>1e-7] - gt[mask>1e-7]).abs()/gt[mask>1e-7]).mean()

    def delta_inlier_ratio(self, pred, gt, mask, degree=1):
        '''Compute the delta inlier rate to a specified degree (def: 1)'''
        return (torch.max(pred[mask>1e-7] / gt[mask>1e-7], gt[mask>1e-7] / pred[mask>1e-7]) < (1.25 ** degree)).float().mean()
