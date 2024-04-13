import os
import cv2
import numpy as np
import torch.utils.data
import cv2
import numpy as np
import torch.utils.data


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

def load_disp_png(filepath):
    array = cv2.imread(filepath, -1)
    valid_mask = array > 0
    disp = array.astype(np.float32) / 256.0
    disp[np.logical_not(valid_mask)] = -1.0
    return disp, valid_mask

def load_calib(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('P_rect_02'):
                proj_mat = line.split()[1:]
                proj_mat = [float(param) for param in proj_mat]
                proj_mat = np.array(proj_mat, dtype=np.float32).reshape(3, 4)
                assert proj_mat[0, 1] == proj_mat[1, 0] == 0
                assert proj_mat[2, 0] == proj_mat[2, 1] == 0
                assert proj_mat[0, 0] == proj_mat[1, 1]
                assert proj_mat[2, 2] == 1

    return proj_mat


def disp2pc(disp, baseline, f, cx, cy, flow=None):
    h, w = disp.shape
    depth = baseline * f / (disp + 1e-5)

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc



def zero_padding(inputs, pad_h, pad_w):
    input_dim = len(inputs.shape)
    assert input_dim in [2, 3]

    if input_dim == 2:
        inputs = inputs[..., None]

    h, w, c = inputs.shape
    assert h <= pad_h and w <= pad_w

    result = np.zeros([pad_h, pad_w, c], dtype=inputs.dtype)
    result[:h, :w, :c] = inputs

    if input_dim == 2:
        result = result[..., 0]

    return result


class KITTI(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)
        assert cfgs.split in ['training200', 'training160', 'training40']

        self.root_dir = os.path.join(cfgs.root_dir, 'training')
        self.split = cfgs.split
        self.cfgs = cfgs
        self.crop = 80

        if self.split == 'training200':
            self.indices = np.arange(200)
        elif self.split == 'training160':
            self.indices = [i for i in range(200) if i % 5 != 0]
        elif self.split == 'training40':
            self.indices = [i for i in range(200) if i % 5 == 0]
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        np.random.seed(23333)

        index = self.indices[i]
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]

        image1 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_10.png' % index))[..., ::-1].copy().astype(np.float32)
        image2 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_11.png' % index))[..., ::-1].copy().astype(np.float32)
        flow_2d, flow_2d_mask = load_flow_png(os.path.join(self.root_dir, 'flow_occ', '%06d_10.png' % index))

        data_dict['input_h'] = image1.shape[0]
        data_dict['input_w'] = image1.shape[1]

        disp1, mask1 = load_disp_png(os.path.join(self.root_dir, 'disp_occ_0', '%06d_10.png' % index))
        disp2, mask2 = load_disp_png(os.path.join(self.root_dir, 'disp_occ_1', '%06d_10.png' % index))
        valid = np.logical_and(np.logical_and(mask1, mask2), flow_2d_mask)
        
        valid = np.logical_and(valid, disp2 > 0.0)
        
        disp1_dense, mask1_dense = load_disp_png(os.path.join(self.root_dir, 'disp_ganet_training', '%06d_10.png' % index))
        disp2_dense, mask2_dense = load_disp_png(os.path.join(self.root_dir, 'disp_ganet_training', '%06d_11.png' % index))


        flow_3d = disp2pc(disp2, baseline=0.54, f=f, cx=cx, cy=cy, flow=flow_2d) - disp2pc(disp1, baseline=0.54, f=f, cx=cx, cy=cy)

        pc1 = disp2pc(disp1_dense, 0.54, f=f, cx=cx, cy=cy)
        pc2 = disp2pc(disp2_dense, 0.54, f=f, cx=cx, cy=cy)
  

        pc1 = pc1[self.crop:]
        pc2 = pc2[self.crop:]


        # limit max depth
        pc1[pc1[..., -1] < self.cfgs.max_depth] = 0.0
        pc2[pc2[..., -1] < self.cfgs.max_depth] = 0.0

        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        intrinsics = np.array([f, f, cx, cy - self.crop])   
        flow_3d = flow_3d[self.crop:]
        flow_2d = flow_2d[self.crop:]
        valid = valid[self.crop:]


        padding_h, padding_w = 320, 1280
        image1 = zero_padding(image1, padding_h, padding_w)
        image2 = zero_padding(image2, padding_h, padding_w)
        pc1 = zero_padding(pc1, padding_h, padding_w)
        pc2 = zero_padding(pc2, padding_h, padding_w)
        valid = zero_padding(valid, padding_h, padding_w)
        flow_3d = zero_padding(flow_3d, padding_h, padding_w)
        flow_2d = zero_padding(flow_2d, padding_h, padding_w)
        
        data_dict['image1'] = np.ascontiguousarray(image1.transpose([2, 0, 1])) # 3 x H x W
        data_dict['image2'] = np.ascontiguousarray(image2.transpose([2, 0, 1])) # 3 x H x W
        data_dict['pc1'] = np.ascontiguousarray(pc1.transpose([2, 0, 1])) # 3 x H x W
        data_dict['pc2'] = np.ascontiguousarray(pc2.transpose([2, 0, 1])) # 3 x H x W
        non_zero_mask = np.linalg.norm(data_dict['pc1'], axis = 0) > 1e-8
        data_dict['nonzero_mask'] = non_zero_mask # H x W
        data_dict['intrinsics'] = intrinsics # 4
        flow_2d = np.concatenate([flow_2d, flow_2d_mask[..., None].astype(np.float32)], axis=-1) # H x W x 3
        data_dict['flow_3d'] = np.ascontiguousarray(flow_3d.transpose([2, 0, 1])) # 3 x H x W
        data_dict['flow_2d'] = np.ascontiguousarray(flow_2d.transpose([2, 0, 1])) # 3 x H x W
        data_dict['mask'] = valid & non_zero_mask # H x W
        

        return data_dict