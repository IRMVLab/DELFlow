import torch.nn as nn
from .camlipwc_core import CamLiPWC_Core
from .base import FlowModel
from .losses2d import calc_supervised_loss_2d
from .losses3d import calc_supervised_loss_3d
from .utils import stride_sample, stride_sample_pc,  normalize_image, resize_to_64x, resize_flow2d, build_pc_pyramid, build_labels


class CamLiPWC(FlowModel):
    def __init__(self, cfgs):
        super(CamLiPWC, self).__init__()
        self.cfgs = cfgs
        self.dense = cfgs.dense
        self.core = CamLiPWC_Core(cfgs.pwc2d, cfgs.pwc3d, self.dense)
        self.stride_h_list = cfgs.stride_h
        self.stride_w_list = cfgs.stride_w
        self.loss = None

    def train(self, mode=True):
        
        self.training = mode

        for module in self.children():
            module.train(mode)

        if self.cfgs.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

        return self

    def eval(self):
        return self.train(False)

    def forward(self, inputs):
        image1 = inputs['image1'].float() / 255.0
        image2 = inputs['image2'].float() / 255.0
        pc1, pc2 = inputs['pc1'].float(), inputs['pc2'].float()
        intrinsics = inputs['intrinsics'].float()
       

        # assert images.shape[2] % 64 == 0 and images.shape[3] % 64 == 0
        origin_h, origin_w = image1.shape[2:]    
        cam_info = {'sensor_h': origin_h, 'sensor_w': origin_w, 'intrinsics': intrinsics, 'type': "dense" if self.dense else "sparse"}

        # encode features
        if self.dense:
            xyzs1, xyzs2 = stride_sample_pc(pc1, pc2, self.stride_h_list, self.stride_w_list)
        else:
            xyzs1, xyzs2, sample_indices1, _ = build_pc_pyramid(
                pc1, pc2, [8192, 4096, 2048, 1024, 512, 256]  # 1/4
            )
            
        feats1_2d, feats1_3d = self.core.encode(image1, xyzs1)
        feats2_2d, feats2_3d = self.core.encode(image2, xyzs2)


        # predict flows (1->2)
        flows_2d, flows_3d = self.core.decode(xyzs1[1:], xyzs2[1:], feats1_2d, feats2_2d, feats1_3d, feats2_3d, pc1, pc2, cam_info)
        
        # final_flow_2d = resize_flow2d(flows_2d[0], origin_h, origin_w)
        final_flow_2d = flows_2d[0]
        final_flow_3d = flows_3d[0]
        
        if 'flow_2d' not in inputs or 'flow_3d' not in inputs:
            return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d}
        
        target_2d = inputs['flow_2d'].float()
        target_3d = inputs['flow_3d'].float()
        valid_mask = inputs['nonzero_mask']
        if self.dense:     
            labels_3d = stride_sample(target_3d, self.stride_h_list[:-2], self.stride_w_list[:-2])
            masks_3d = stride_sample(valid_mask, self.stride_h_list[:-2], self.stride_w_list[:-2])
        else:
            labels_3d, masks_3d = build_labels(target_3d, sample_indices1)
               
        # calculate losses
        loss_2d = calc_supervised_loss_2d(flows_2d, target_2d, self.cfgs.loss2d)
        loss_3d = calc_supervised_loss_3d(flows_3d, labels_3d, self.cfgs.loss3d, masks_3d)
        self.loss = loss_2d + loss_3d

        # prepare scalar summary
        self.update_metrics('loss', self.loss)
        self.update_metrics('loss2d', loss_2d)
        self.update_metrics('loss3d', loss_3d)
        self.update_2d_metrics(final_flow_2d, target_2d)
        self.update_3d_metrics(final_flow_3d, target_3d, valid_mask)

        if 'mask' in inputs:
            self.update_3d_metrics(final_flow_3d, target_3d, valid_mask, inputs['mask'])

        return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d}

    @staticmethod
    def is_better(curr_summary, best_summary):
        if best_summary is None:
            return True
        return curr_summary['epe2d'] < best_summary['epe2d']


