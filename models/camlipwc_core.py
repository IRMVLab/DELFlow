import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu, interpolate
from .camlipwc2d_core import FeaturePyramid2D, FlowEstimatorDense2D, ContextNetwork2D
from .camlipwc3d_core import FeaturePyramid3D, FeaturePyramid3DS, Costvolume3D, Costvolume3DS, FlowPredictor3D, FlowPredictor3DS, KnnUpsampler3D
from .utils import Conv1dNormRelu, Conv2dNormRelu
from .utils import backwarp_2d, backwarp_3d, mesh_grid, forwarp_3d, knn_interpolation, convex_upsample, project_3d_to_2d, knn_grouping_2d, mask_batch_selecting
from .csrc import correlation2d, k_nearest_neighbor
from .fusion_module import GlobalFuser, GlobalFuserS

class CamLiPWC_Core(nn.Module):
    def __init__(self, cfgs2d, cfgs3d, dense=False, debug=False):
        super().__init__()
        
        self.cfgs2d, self.cfgs3d, self.debug, self.dense = cfgs2d, cfgs3d, debug, dense
        corr_channels_2d = (2 * cfgs2d.max_displacement + 1) ** 2

        ## PWC-Net 2D (IRR-PWC)
        self.branch_2d_fnet = FeaturePyramid2D(
            [3, 16, 32, 64, 96, 128, 192],
            norm=cfgs2d.norm.feature_pyramid
        )
        self.branch_2d_fnet_aligners = nn.ModuleList([
            nn.Identity(),
            Conv2dNormRelu(32, 64),
            Conv2dNormRelu(64, 64),
            Conv2dNormRelu(96, 64),
            Conv2dNormRelu(128, 64),
            Conv2dNormRelu(192, 64),
        ])
        self.branch_2d_flow_estimator = FlowEstimatorDense2D(
            [64 + corr_channels_2d + 2 + 32, 128, 128, 96, 64, 32],
            norm=cfgs2d.norm.flow_estimator,
            conv_last=False,
        )
        self.branch_2d_context_network = ContextNetwork2D(
            [self.branch_2d_flow_estimator.flow_feat_dim + 2, 128, 128, 128, 96, 64, 32],
            dilations=[1, 2, 4, 8, 16, 1],
            norm=cfgs2d.norm.context_network
        )
        self.branch_2d_up_mask_head = nn.Sequential(  # for convex upsampling (see RAFT)
            nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 4 * 9, kernel_size=1, stride=1, padding=0),
        )
        self.branch_2d_conv_last = nn.Conv2d(self.branch_2d_flow_estimator.flow_feat_dim, 2, kernel_size=3, stride=1, padding=1)

        if dense:
            ## PWC-Net 3D (Point-PWC)
            self.branch_3d_fnet = FeaturePyramid3D(
                [16, 32, 64, 96, 128, 192],
                norm=cfgs3d.norm.feature_pyramid,
                k=cfgs3d.k,
                ks=cfgs3d.kernel_size
            )
            self.branch_3d_fnet_aligners = nn.ModuleList([
                nn.Identity(),
                Conv2dNormRelu(32, 64),
                Conv2dNormRelu(64, 64),
                Conv2dNormRelu(96, 64),
                Conv2dNormRelu(128, 64),
                Conv2dNormRelu(192, 64),
            ])
            self.branch_3d_correlations = nn.ModuleList([
                nn.Identity(),
                Costvolume3D(32, 32, k=self.cfgs3d.k),
                Costvolume3D(64, 64, k=self.cfgs3d.k),
                Costvolume3D(96, 96, k=self.cfgs3d.k),
                Costvolume3D(128, 128, k=self.cfgs3d.k),
                Costvolume3D(192, 192, k=self.cfgs3d.k),
            ])
            self.branch_3d_correlation_aligners = nn.ModuleList([
                nn.Identity(),
                Conv2dNormRelu(32, 64),
                Conv2dNormRelu(64, 64),
                Conv2dNormRelu(96, 64),
                Conv2dNormRelu(128, 64),
                Conv2dNormRelu(192, 64),
            ])
            self.branch_3d_flow_estimator = FlowPredictor3D(
                [64 + 64 + 3 + 64, 128, 128, 64],
                cfgs3d.norm.flow_estimator,
                conv_last=False,
                k=self.cfgs3d.k,
            )

            ## Bi-CLFM for pyramid features
            self.pyramid_feat_fusers = nn.ModuleList([
                nn.Identity(),
                GlobalFuser(32, 32, norm=cfgs2d.norm.feature_pyramid),
                GlobalFuser(64, 64, norm=cfgs2d.norm.feature_pyramid),
                GlobalFuser(96, 96, norm=cfgs2d.norm.feature_pyramid),
                GlobalFuser(128, 128, norm=cfgs2d.norm.feature_pyramid),
                GlobalFuser(192, 192, norm=cfgs2d.norm.feature_pyramid),
            ])

            ## Bi-CLFM for correlation features
            self.corr_feat_fusers = nn.ModuleList([
                nn.Identity(),
                GlobalFuser(corr_channels_2d, 32),
                GlobalFuser(corr_channels_2d, 64),
                GlobalFuser(corr_channels_2d, 96),
                GlobalFuser(corr_channels_2d, 128),
                GlobalFuser(corr_channels_2d, 192),
            ])

            self.intepolate_3d = KnnUpsampler3D(stride_h = 2, stride_w = 2, k = 3)
            self.knn_upconv = KnnUpsampler3D(stride_h = 4, stride_w = 4, k = 3)

            ## Bi-CLFM for decoder features
            self.estimator_feat_fuser = GlobalFuser(self.branch_2d_flow_estimator.flow_feat_dim, self.branch_3d_flow_estimator.flow_feat_dim)
        
            self.branch_3d_conv_last = nn.Conv2d(self.branch_3d_flow_estimator.flow_feat_dim, 3, kernel_size=1)

        else:
            self.branch_3d_fnet = FeaturePyramid3DS(
                n_channels=[16, 32, 64, 96, 128, 192],  # 1/4
                norm=cfgs3d.norm.feature_pyramid,
                k=cfgs3d.k,
            )
            self.branch_3d_fnet_aligners = nn.ModuleList([
                nn.Identity(),
                Conv1dNormRelu(32, 64),  # 1/4
                Conv1dNormRelu(64, 64),
                Conv1dNormRelu(96, 64),
                Conv1dNormRelu(128, 64),
                Conv1dNormRelu(192, 64),
            ])
            self.branch_3d_correlations = nn.ModuleList([
                nn.Identity(),
                Costvolume3DS(32, 32, k=self.cfgs3d.k),  # 1/4
                Costvolume3DS(64, 64, k=self.cfgs3d.k),
                Costvolume3DS(96, 96, k=self.cfgs3d.k),
                Costvolume3DS(128, 128, k=self.cfgs3d.k),
                Costvolume3DS(192, 192, k=self.cfgs3d.k),
            ])
            self.branch_3d_correlation_aligners = nn.ModuleList([
                nn.Identity(),
                Conv1dNormRelu(32, 64),  # 1/4
                Conv1dNormRelu(64, 64),
                Conv1dNormRelu(96, 64),
                Conv1dNormRelu(128, 64),
                Conv1dNormRelu(192, 64),
            ])
            self.branch_3d_flow_estimator = FlowPredictor3DS(
                [64 + 64 + 3 + 64, 128, 128, 64],
                cfgs3d.norm.flow_estimator,
                k=self.cfgs3d.k,
            )


            self.pyramid_feat_fusers = nn.ModuleList([
                nn.Identity(),
                GlobalFuserS(32, 32, norm=cfgs2d.norm.feature_pyramid),  # 1/4
                GlobalFuserS(64, 64, norm=cfgs2d.norm.feature_pyramid),
                GlobalFuserS(96, 96, norm=cfgs2d.norm.feature_pyramid),
                GlobalFuserS(128, 128, norm=cfgs2d.norm.feature_pyramid),
                GlobalFuserS(192, 192, norm=cfgs2d.norm.feature_pyramid),
            ])

            self.corr_feat_fusers = nn.ModuleList([
                nn.Identity(),
                GlobalFuserS(corr_channels_2d, 32),  # 1/4
                GlobalFuserS(corr_channels_2d, 64),
                GlobalFuserS(corr_channels_2d, 96),
                GlobalFuserS(corr_channels_2d, 128),
                GlobalFuserS(corr_channels_2d, 192),
            ])
            
            self.estimator_feat_fuser = GlobalFuserS(self.branch_2d_flow_estimator.flow_feat_dim, self.branch_3d_flow_estimator.flow_feat_dim)
            self.branch_3d_conv_last = nn.Conv1d(self.branch_3d_flow_estimator.flow_feat_dim, 3, kernel_size=1)


    def encode(self, image, xyzs):
        feats_2d = self.branch_2d_fnet(image)
        feats_3d = self.branch_3d_fnet(xyzs)
        return feats_2d, feats_3d

    def decode(self, xyzs1, xyzs2, feats1_2d, feats2_2d, feats1_3d, feats2_3d, raw_pc1, raw_pc2, camera_info):
        assert len(xyzs1) == len(xyzs2) == len(feats1_2d) == len(feats2_2d) == len(feats1_3d) == len(feats2_3d)
        
        flows_2d, flows_3d, flow_feats_2d, flow_feats_3d = [], [], [], []
        for level in range(len(xyzs1) - 1, 0, -1):
            xyz1, feat1_2d, feat1_3d = xyzs1[level], feats1_2d[level], feats1_3d[level]
            xyz2, feat2_2d, feat2_3d = xyzs2[level], feats2_2d[level], feats2_3d[level]

            batch_size, image_h, image_w = feat1_2d.shape[0], feat1_2d.shape[2], feat1_2d.shape[3]
            if not self.dense:
                n_points = xyz1.shape[-1]

            # project point cloud to image
            uv1, uv_mask1 = project_3d_to_2d(xyz1, camera_info, image_h, image_w)
            uv2, uv_mask2 = project_3d_to_2d(xyz2, camera_info, image_h, image_w)

            # pre-compute knn indices
            if self.dense:
                knn_1in1, valid_mask_1in1 = knn_grouping_2d(xyz1, xyz1, k=self.cfgs3d.k)  # [bs, n_points, k]
            else:
                grid = mesh_grid(batch_size, image_h, image_w, uv1.device)  # [B, 2, H, W]
                grid = grid.reshape([batch_size, 2, -1])  # [B, 2, HW]
                knn_1in1 = k_nearest_neighbor(xyz1, xyz1, k=self.cfgs3d.k)  # [bs, n_points, k]
                valid_mask_1in1 = None
                
            # fuse pyramid features
            feat1_2d, feat1_3d = self.pyramid_feat_fusers[level](uv1, uv_mask1, feat1_2d, feat1_3d)
            feat2_2d, feat2_3d = self.pyramid_feat_fusers[level](uv2, uv_mask2, feat2_2d, feat2_3d)
        

            if level == len(xyzs1) - 1:
                last_flow_2d = torch.zeros([batch_size, 2, image_h, image_w], dtype=xyz1.dtype, device=xyz1.device)
                last_flow_feat_2d = torch.zeros([batch_size, 32, image_h, image_w], dtype=xyz1.dtype, device=xyz1.device)
                
                if self.dense:
                    last_flow_3d = torch.zeros([batch_size, 3, image_h, image_w], dtype=xyz1.dtype, device=xyz1.device)
                    last_flow_feat_3d = torch.zeros([batch_size, 64, image_h, image_w], dtype=xyz1.dtype, device=xyz1.device)
                    xyz1_warp, feat2_2d_warp = xyz1, feat2_2d
                    warping_idx = None
                else:
                    last_flow_3d = torch.zeros([batch_size, 3, n_points], dtype=xyz1.dtype, device=xyz1.device)
                    last_flow_feat_3d = torch.zeros([batch_size, 64, n_points], dtype=xyz1.dtype, device=xyz1.device)
                    xyz2_warp, feat2_2d_warp = xyz2, feat2_2d                    
            else:
                # upsample 2d flow and backwarp
                last_flow_2d = interpolate(flows_2d[-1] * 2, scale_factor=2, mode='bilinear', align_corners=True)
                last_flow_feat_2d = interpolate(flow_feats_2d[-1], scale_factor=2, mode='bilinear', align_corners=True)
                feat2_2d_warp = backwarp_2d(feat2_2d, last_flow_2d, padding_mode='border')

                if self.dense:
                    # upsample 3d flow and backwarp
                    flow_with_feat_3d = torch.cat([flows_3d[-1], flow_feats_3d[-1]], dim=1)
                    flow_with_feat_upsampled_3d = self.intepolate_3d(xyz1, xyzs1[level + 1], flow_with_feat_3d)
                    last_flow_3d, last_flow_feat_3d = torch.split(flow_with_feat_upsampled_3d, [3, 64], dim=1)
                    xyz1_warp, warping_idx = forwarp_3d(xyz1, xyz2, last_flow_3d, camera_info)
                else:
                    last_flow_3d, last_flow_feat_3d = torch.split(
                        knn_interpolation(
                            xyzs1[level + 1],
                            torch.cat([flows_3d[-1], flow_feats_3d[-1]], dim=1),
                            xyz1
                        ), [3, 64], dim=1)      
                    xyz2_warp = backwarp_3d(xyz1, xyz2, last_flow_3d)      
                
            # correlation (2D & 3D)
            if self.dense:
                feat_corr_3d = self.branch_3d_correlations[level](xyz1_warp, feat1_3d, xyz2, feat2_3d, warping_idx, knn_1in1, valid_mask_1in1)
            else:
                feat_corr_3d = self.branch_3d_correlations[level](xyz1, feat1_3d, xyz2_warp, feat2_3d, knn_1in1)
            feat_corr_2d = leaky_relu(correlation2d(feat1_2d, feat2_2d_warp, self.cfgs2d.max_displacement), 0.1)

            # fuse correlation features
            feat_corr_2d, feat_corr_3d = self.corr_feat_fusers[level](uv1, uv_mask1, feat_corr_2d, feat_corr_3d)

            # align features using 1x1 convolution
            feat1_2d = self.branch_2d_fnet_aligners[level](feat1_2d)
            feat1_3d = self.branch_3d_fnet_aligners[level](feat1_3d)
            feat_corr_3d = self.branch_3d_correlation_aligners[level](feat_corr_3d)

            # flow decoder (or estimator)
            x_2d = torch.cat([feat_corr_2d, feat1_2d, last_flow_2d, last_flow_feat_2d], dim=1)
            x_3d = torch.cat([feat_corr_3d, feat1_3d, last_flow_3d, last_flow_feat_3d], dim=1)
            flow_feat_2d = self.branch_2d_flow_estimator(x_2d)  # [bs, 96, image_h, image_w]
            flow_feat_3d = self.branch_3d_flow_estimator(xyz1, x_3d, knn_1in1, valid_mask_1in1)  # [bs, 64, n_points]

            # fuse decoder features
            flow_feat_2d, flow_feat_3d = self.estimator_feat_fuser(uv1, uv_mask1, flow_feat_2d, flow_feat_3d)

            # flow prediction
            flow_delta_2d = self.branch_2d_conv_last(flow_feat_2d)
            flow_delta_3d = self.branch_3d_conv_last(flow_feat_3d)

            # residual connection
            flow_2d = last_flow_2d + flow_delta_2d
            flow_3d = last_flow_3d + flow_delta_3d

            # context network (2D only)
            flow_feat_2d, flow_delta_2d = self.branch_2d_context_network(torch.cat([flow_feat_2d, flow_2d], dim=1))
            flow_2d = flow_delta_2d + flow_2d

            # clip
            flow_2d = torch.clip(flow_2d, min=-1000, max=1000)
            flow_3d = torch.clip(flow_3d, min=-100, max=100)

            # save results
            flows_2d.append(flow_2d)
            flows_3d.append(flow_3d)
            flow_feats_2d.append(flow_feat_2d)
            flow_feats_3d.append(flow_feat_3d)

        flows_2d = [f.float() for f in flows_2d][::-1]
        flows_3d = [f.float() for f in flows_3d][::-1]

        # convex upsamling module, from RAFT
        flows_2d[0] = convex_upsample(flows_2d[0], self.branch_2d_up_mask_head(flow_feats_2d[-1]), scale_factor=4)

        for i in range(1, len(flows_2d)):
            flows_2d[i] = interpolate(flows_2d[i] * 4, scale_factor=4, mode='bilinear', align_corners=True)

        for i in range(len(flows_3d)):
            if self.dense:
                if i == 0:
                    flows_3d[i] = self.knn_upconv(raw_pc1, xyzs1[i + 1], flows_3d[i])
                else:
                    flows_3d[i] = self.knn_upconv(xyzs1[i - 1], xyzs1[i + 1], flows_3d[i])
            else:
                flows_3d[i] = knn_interpolation(xyzs1[i + 1], flows_3d[i], xyzs1[i])
        
        return flows_2d, flows_3d
