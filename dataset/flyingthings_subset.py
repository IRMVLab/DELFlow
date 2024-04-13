import os
import numpy as np
import glob
import torch
import cv2
from .augmentation import joint_augmentation, random_downsample, scale_crop

__all__ = ['FlyingThings']

def load_flow_png(filepath, scale=64.0):
    # for KITTI which uses 16bit PNG images
    # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flow_img = cv2.imread(filepath, -1)
    flow = flow_img[:, :, 2:0:-1].astype(np.float32)
    mask = flow_img[:, :, 0] > 0
    flow = flow - 32768.0
    flow = flow / scale
    return flow, mask

class FlyingThings3D(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.root_dir = str(cfgs.root_dir)
        self.split = str(cfgs.split)
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.cfgs = cfgs

        self.indices = []
        for filename in os.listdir(os.path.join(self.root_dir, self.split, 'flow_2d')):
            self.indices.append(int(filename.split('.')[0]))

        if not cfgs.full:
            self.indices = self.indices[::4]
        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(0)

        idx1 = self.indices[i]
        idx2 = idx1 + 1
        data_dict = {'index': idx1}

        # camera intrinsics
        f, cx, cy = 1050, 479.5, 269.5

        # load data
        pcs = np.load(os.path.join(self.split_dir, 'pc', '%07d.npz' % idx1))
        pc1, pc2 = pcs['pc1'], pcs['pc2']

        flow_2d, flow_mask_2d = load_flow_png(os.path.join(self.split_dir, 'flow_2d', '%07d.png' % idx1))
        flow_3d = np.load(os.path.join(self.split_dir, 'flow_3d', '%07d.npy' % idx1))

        occ_mask_3d = np.load(os.path.join(self.split_dir, 'occ_mask_3d', '%07d.npy' % idx1))
        occ_mask_3d = np.unpackbits(occ_mask_3d, count=len(pc1))
 
        image1 = cv2.imread(os.path.join(self.split_dir, 'image', '%07d.png' % idx1))[..., ::-1].copy().astype(np.float32)
        image2 = cv2.imread(os.path.join(self.split_dir, 'image', '%07d.png' % idx2))[..., ::-1].copy().astype(np.float32)

        H, W = image1.shape[:2]

        pc1_mask = np.load(os.path.join(self.split_dir, 'pc1_mask', '%07d.npy' % idx1))
        pc1_mask = np.unpackbits(pc1_mask).reshape(H, W).astype(np.bool_)

        pc2_mask = np.load(os.path.join(self.split_dir, 'pc2_mask', '%07d.npy' % idx1))
        pc2_mask = np.unpackbits(pc2_mask).reshape(H, W).astype(np.bool_)
        
        
        # ignore fast moving objects
        flow_mask_2d = np.logical_and(flow_mask_2d, np.linalg.norm(flow_2d, axis=-1) < 250.0)
        flow_2d = np.concatenate([flow_2d, flow_mask_2d[..., None].astype(np.float32)], axis=2)


        pc1_hw3 = np.zeros((H, W, 3), dtype = np.float32)
        pc2_hw3 = np.zeros((H, W, 3), dtype = np.float32)
        flow_hw3 = np.zeros((H, W, 3), dtype = np.float32)
        valid_mask = np.zeros((H, W), dtype = np.bool_)


        pc1_hw3[pc1_mask] = pc1
        pc2_hw3[pc2_mask] = pc2
        flow_hw3[pc1_mask] = flow_3d
        valid_mask[pc1_mask] = np.logical_not(occ_mask_3d)
        # valid_mask = pc1_mask

        image1, image2, pc1_hw3, pc2_hw3, flow_2d, flow_hw3, valid_mask, f, cx, cy = joint_augmentation(
            image1, image2, pc1_hw3, pc2_hw3, flow_2d, flow_hw3, valid_mask, f, cx, cy, self.cfgs.augmentation
        )
 
        # image1, image2, pc1_hw3, pc2_hw3, flow_2d, flow_hw3, valid_mask = scale_crop(image1, image2, pc1_hw3, pc2_hw3, flow_2d, flow_hw3, valid_mask)
         
        intrinsics = np.float32([f, f, cx, cy])
        
        image1 = np.ascontiguousarray(image1.transpose([2, 0, 1]))
        image2 = np.ascontiguousarray(image2.transpose([2, 0, 1]))
        pc1_3hw = np.ascontiguousarray(pc1_hw3.transpose([2, 0, 1]))
        pc2_3hw = np.ascontiguousarray(pc2_hw3.transpose([2, 0, 1]))
        flow_3hw = np.ascontiguousarray(flow_hw3.transpose([2, 0, 1]))
        valid_mask = np.ascontiguousarray(valid_mask)
        pc1_mask = np.ascontiguousarray(pc1_mask)
        data_dict['image1'] = image1 # 3 x H x W
        data_dict['image2'] = image2 # 3 x H x W
        data_dict['pc1'] = pc1_3hw # 3 x H x W
        data_dict['pc2'] = pc2_3hw # 3 x H x W
        data_dict['flow_3d'] = flow_3hw # 3 x H x W
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1]) # 3 x H x W
        data_dict['mask'] = valid_mask # H x W
        
        non_zero_mask = np.linalg.norm(pc1_3hw, axis = 0) > 1e-8
        data_dict['nonzero_mask'] = non_zero_mask # H x W
        data_dict['intrinsics'] = intrinsics # 4

        return data_dict
