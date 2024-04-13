import torch
import torch.nn as nn
from .pointconv import PointConv, PointConvS
from .utils import MLP1d, MLP2d, Conv1dNormRelu, k_nearest_neighbor, batch_indexing, knn_grouping_2d, get_hw_idx, mask_batch_selecting
from ops_pytorch.gpu_threenn_sample.no_sort_knn import no_sort_knn


def get_selected_idx(batch_size: int, out_H: int, out_W: int, stride_H: int, stride_W: int):

    select_h_idx = torch.arange(0, out_H * stride_H, stride_H, device = "cuda") # [out_H]
    select_w_idx = torch.arange(0, out_W * stride_W, stride_W, device = "cuda") # [out_W]
    height_indices = torch.reshape(select_h_idx, (1, 1, -1, 1)).expand(batch_size, 1, out_H, out_W)  # b out_H out_W
    width_indices = torch.reshape(select_w_idx, (1, 1, 1, -1)).expand(batch_size, 1, out_H, out_W)  # b out_H out_W
    select_idx = torch.cat([height_indices, width_indices], dim = 1)
    return select_idx

def fast_index(inputs, idx):
    """
    Input:
        inputs: input points data, [B, 3, H, W]
        idx: sample index data, [B, 2, h, w]
    Return:
        outputs:, indexed points data, [B, 3, h, w]
    """
    if len(inputs.shape) == 4:
        B, C, H, W = inputs.shape
        _, _, h, w = idx.shape
        neighbor_idx = idx[:, 0] * W + idx[:, 1]  # （B, h, w)
        neighbor_idx = neighbor_idx.reshape(B, 1, h*w)  # （B, h*w)
        inputs_bcn = inputs.reshape(B, C, H*W)  # （B, C, H*W)
        gather_feat = torch.gather(inputs_bcn, 2, neighbor_idx.expand(-1, C, -1)) # [B, C, h*w] 
        outputs = torch.reshape(gather_feat, [B, C, h, w])
    else:
        B, H, W = inputs.shape
        _, _, h, w = idx.shape
        neighbor_idx = idx[:, 0] * W + idx[:, 1]  # （B, h, w)
        neighbor_idx = neighbor_idx.reshape(B, h*w)  # （B, h*w)
        inputs_bcn = inputs.reshape(B, H*W)  # （B, H*W)
        gather_feat = torch.gather(inputs, 1, neighbor_idx) # [B, h*w] 
        outputs = torch.reshape(gather_feat, [B, h, w])

    return outputs

def stride_sample_gather(pc1, pc2, label, mask, stride_H_list, stride_W_list):
    """
    Input:
        pc1: input points data, [B, 3, H, W]
        pc2: input points data, [B, 2, H, H]
        mask: [B, H, W]
        label: [B, 3, H, H]
        stride_list: list of sampling strides
    Return:
        xyzs:, list of sampled pc
    """    
    B = pc1.shape[0]
    H_list = [pc1.shape[2]]; W_list = [pc1.shape[3]]

    xyzs1 = [pc1]; xyzs2 = [pc2]; labels = [label]; masks = [mask]

    for s_h, s_w in zip(stride_H_list, stride_W_list):
        H_list.append(H_list[-1] // s_h)
        W_list.append(W_list[-1] // s_w)
        idx = get_selected_idx(B, H_list[-1], W_list[-1], s_h, s_w)
        xyzs1.append(fast_index(xyzs1[-1], idx))
        xyzs2.append(fast_index(xyzs2[-1], idx))
        labels.append(fast_index(labels[-1], idx))
        masks.append(fast_index(masks[-1], idx))

    return xyzs1, xyzs2, labels[1:], masks[1:]

class KnnUpsampler3D(nn.Module):
    def __init__(self, stride_h, stride_w, ks = [10, 20], dist = 100.0, k=3) -> None:
        super().__init__()
        self.k = k
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dist = dist
        self.ks = ks

    @torch.no_grad()
    def knn_grouping(self, query_xyz, input_xyz):
        """
        :param query_xyz: [batch_size, 3, H, W]
        :param input_xyz: [batch_size, 3, h, w]
        :return grouped idx: [batch_size, H*W, k]
        """
        B, C, h, w = input_xyz.shape
        _, _, H, W = query_xyz.shape
        n_sampled = H * W

        assert H // h == self.stride_h and W // w == self.stride_w, "size mismatch"
        
        idx_hw = get_hw_idx(B, H, W).contiguous()
        random_HW = torch.arange(0, self.ks[0] * self.ks[1], device = "cuda", dtype = torch.int)
        input_xyz_hw3 = input_xyz.permute(0, 2, 3, 1).contiguous()    # [B, H, W, 3]
        query_xyz_hw3 = query_xyz.permute(0, 2, 3, 1).contiguous()    # [B, H, W, 3]  

        # Initialize
        select_b_idx = torch.zeros(B, n_sampled, self.k, 1, device = 'cuda').long().detach()             # (B N nsample_q 1)
        select_h_idx = torch.zeros(B, n_sampled, self.k, 1, device = 'cuda').long().detach()
        select_w_idx = torch.zeros(B, n_sampled, self.k, 1, device = 'cuda').long().detach()
        valid_mask = torch.zeros(B, n_sampled, self.k, 1, device = 'cuda').float().detach()
        
        # with torch.no_grad():
        # Sample QNN of (M neighbour points from sampled n points in PC1) in PC2
        select_b_idx, select_h_idx, select_w_idx, valid_mask = no_sort_knn\
            (query_xyz_hw3, input_xyz_hw3, idx_hw, random_HW, H, W, n_sampled, self.ks[0], self.ks[1],\
                self.k, 1, self.dist, self.stride_h, self.stride_w, select_b_idx, select_h_idx, select_w_idx, valid_mask)  
                
        neighbor_idx = select_h_idx * w + select_w_idx  # [B, H*W, k, 1]
       
        return neighbor_idx.squeeze(-1), valid_mask.squeeze(-1)
        # return neighbor_idx.squeeze(-1), valid_mask.squeeze(-1)
       
    def forward(self, query_xyz, input_xyz, input_features):
        """
        :param input_xyz: 3D locations of input points, [B, 3, h, w]
        :param input_features: features of input points, [B, C, h, w]
        :param query_xyz: 3D locations of query points, [B, 3, H, W]
        :param k: k-nearest neighbor, int
        :return interpolated features: [B, C, H, W]
        """
        B, _, H, W = query_xyz.shape
        

        # knn_indices: [B, H*W, 3]
        knn_indices, valid_knn_mask = self.knn_grouping(query_xyz, input_xyz) 
        knn_xyz = mask_batch_selecting(input_xyz, knn_indices, valid_knn_mask)  # [B, 3, H*W, 3]
        query_xyz = query_xyz.view(B, 3, H*W)
        knn_dists = torch.linalg.norm(knn_xyz - query_xyz[..., None], dim = 1).clamp(1e-8)
        # knn_weights: [B, H*W, 3]
        knn_weights = 1.0 / knn_dists
        knn_weights = knn_weights / torch.sum(knn_weights, dim = -1, keepdim = True)
        knn_features = mask_batch_selecting(input_features, knn_indices, valid_knn_mask)  # [B, C, H*W, 3]

        # interpolated: [B, C, H*W]
        interpolated = torch.sum(knn_features * knn_weights[:, None, :, :], dim=-1)

        interpolated = interpolated.view(B, -1, H, W)
        return interpolated

class FeaturePyramid3D(nn.Module):
    def __init__(self, n_channels, norm=None, k=16, ks = [10, 20]):
        super().__init__()

        self.mlps = nn.ModuleList([MLP2d(3, [n_channels[0], n_channels[0]])])
        self.convs = nn.ModuleList([PointConv(n_channels[0], n_channels[0], norm=norm, k=k, ks = ks)])

        for i in range(1, len(n_channels)):
            self.mlps.append(MLP2d(n_channels[i - 1], [n_channels[i - 1], n_channels[i]]))
            self.convs.append(
                PointConv(n_channels[i], n_channels[i], norm=norm, k=k, ks = ks)
            )

    def forward(self, xyzs):
        """
        :param xyzs: pyramid of points
        :return feats: pyramid of features
        """
        assert len(xyzs) == len(self.mlps) + 1

        input_feat = xyzs[0]  # [bs, 3, h, w]
        # input_feat = self.level0_mlp(inputs)
        feats = []

        for i in range(len(xyzs) - 1):
            if i == 0:
                feat = self.mlps[i](input_feat)
            else:
                feat = self.mlps[i](feats[-1])
                
            feat = self.convs[i](xyzs[i], feat, xyzs[i + 1])
            feats.append(feat)

        return feats

class FeaturePyramid3DS(nn.Module):
    def __init__(self, n_channels, norm=None, k=16):
        super().__init__()

        self.level0_mlp = MLP1d(3, [n_channels[0], n_channels[0]])

        self.pyramid_mlps = nn.ModuleList()
        self.pyramid_convs = nn.ModuleList()

        for i in range(len(n_channels) - 1):
            self.pyramid_mlps.append(MLP1d(n_channels[i], [n_channels[i], n_channels[i + 1]]))
            self.pyramid_convs.append(PointConvS(n_channels[i + 1], n_channels[i + 1], norm=norm, k=k))

    def forward(self, xyzs):
        """
        :param xyzs: pyramid of points
        :return feats: pyramid of features
        """
        assert len(xyzs) == len(self.pyramid_mlps) + 1

        inputs = xyzs[0] # [bs, 3, n_points]
        feats = [self.level0_mlp(inputs)]

        for i in range(len(xyzs) - 1):
            feat = self.pyramid_mlps[i](feats[-1])
            feats.append(self.pyramid_convs[i](xyzs[i], feat, xyzs[i + 1]))

        return feats
    
class Costvolume3D(nn.Module):
    def __init__(self, in_channels, out_channels, ks = [10, 20], dist = 100.0, k=16):
        super().__init__()

        self.k = k
        self.ks = ks
        self.dist = dist
        self.cost_mlp = MLP2d(3 + 2 * in_channels, [out_channels, out_channels], act='leaky_relu')
        self.weight_net1 = MLP2d(3, [8, 8, out_channels], act='relu')
        self.weight_net2 = MLP2d(3, [8, 8, out_channels], act='relu')

    
    def forward(self, xyz1, feat1, xyz2, feat2, idx_fetching=None, knn_indices_1in1=None, valid_mask_1in1 = None):
        """
        :param xyz1: [batch_size, 3, H, W]
        :param feat1: [batch_size, in_channels, H, W]
        :param xyz2: [batch_size, 3, H, W]
        :param feat2: [batch_size, in_channels, H, W]
        :param warping idx: for each warped point in xyz1, find its position in xyz2, [batch_size, H * W, 2]
        :return cost volume: [batch_size, n_cost_channels, H, W]
        """
        B, C, H, W = feat1.shape
        feat1 = feat1.view(B, C, H * W)
        
        knn_indices_1in2, valid_mask_1in2 = knn_grouping_2d(query_xyz=xyz1, input_xyz=xyz2, k=self.k, idx_fetching = idx_fetching)
        # knn_xyz2: [B, 3, H*W, k], 
        knn_xyz2 = mask_batch_selecting(xyz2, knn_indices_1in2, valid_mask_1in2)
        # knn_xyz2_norm: [B, 3, H*W, k]
        knn_xyz2_norm = knn_xyz2 - xyz1.view(B, 3, H * W, 1)
        # knn_features2: [B, C, H*W, k]
        knn_features2 = mask_batch_selecting(feat2, knn_indices_1in2, valid_mask_1in2)
        # features1_expand: [B, C, H*W, k]
        features1_expand = feat1[:, :, :, None].expand(B, C, H * W, self.k) 
        # concatenated_features: [B, 2C+3, H*W, k]
        concatenated_features = torch.cat([features1_expand, knn_features2, knn_xyz2_norm], dim=1) 
        # p2p_cost (point-to-point cost): [B, out_channels, H*W, k]
        p2p_cost = self.cost_mlp(concatenated_features)
        
        # weights2: [B, out_channels, H*W, k]
        weights2 = self.weight_net2(knn_xyz2_norm)
        # p2n_cost (point-to-neighbor cost): [B, out_channels, H * W]
        p2n_cost = torch.sum(weights2 * p2p_cost, dim=3)

        if knn_indices_1in1 is not None:
            assert knn_indices_1in1.shape[:2] == torch.Size([B, H * W])
            assert knn_indices_1in1.shape[2] >= self.k
            knn_indices_1in1 = knn_indices_1in1[:, :, :self.k]
            valid_mask_1in1 = valid_mask_1in1[:, :, :self.k]
        else:
            knn_indices_1in1, valid_mask_1in1 = knn_grouping_2d(query_xyz=xyz1, input_xyz=xyz1, k=self.k)  # [bs, n_points, k]
 
        # knn_xyz1: [B, 3, H*W, k]
        knn_xyz1 = mask_batch_selecting(xyz1, knn_indices_1in1, valid_mask_1in1)
        # knn_xyz1_norm: [B, 3, H*W, k]
        knn_xyz1_norm = knn_xyz1 - xyz1.view(B, 3, H * W, 1)
        # weights1: [B, out_channels, H*W, k]
        weights1 = self.weight_net1(knn_xyz1_norm)
        # n2n_cost: [B, out_channels, H*W, k]
        n2n_cost = mask_batch_selecting(p2n_cost, knn_indices_1in1, valid_mask_1in1)
        # n2n_cost: [B, out_channels, H * W]
        n2n_cost = torch.sum(weights1 * n2n_cost, dim=3)
        
        n2n_cost = n2n_cost.view(B, -1, H, W)

        return n2n_cost

class Costvolume3DS(nn.Module):
    def __init__(self, in_channels, out_channels, align_channels=None, k=16):
        super().__init__()
        self.k = k

        self.cost_mlp = MLP2d(3 + 2 * in_channels, [out_channels, out_channels], act='leaky_relu')
        self.weight_net1 = MLP2d(3, [8, 8, out_channels], act='relu')
        self.weight_net2 = MLP2d(3, [8, 8, out_channels], act='relu')

        if align_channels is not None:
            self.feat_aligner = Conv1dNormRelu(out_channels, align_channels)
        else:
            self.feat_aligner = nn.Identity()

    def forward(self, xyz1, feat1, xyz2, feat2, knn_indices_1in1=None):
        """
        :param xyz1: [batch_size, 3, n_points]
        :param feat1: [batch_size, in_channels, n_points]
        :param xyz2: [batch_size, 3, n_points]
        :param feat2: [batch_size, in_channels, n_points]
        :param knn_indices_1in1: for each point in xyz1, find its neighbors in xyz1, [batch_size, n_points, k]
        :return cost volume for each point in xyz1: [batch_size, n_cost_channels, n_points]
        """
        batch_size, in_channels, n_points = feat1.shape

        # Step1: for each point in xyz1, find its neighbors in xyz2
        knn_indices_1in2 = k_nearest_neighbor(input_xyz=xyz2, query_xyz=xyz1, k=self.k)
        # knn_xyz2: [bs, 3, n_points, k]
        knn_xyz2 = batch_indexing(xyz2, knn_indices_1in2)
        # knn_xyz2_norm: [bs, 3, n_points, k]
        knn_xyz2_norm = knn_xyz2 - xyz1.view(batch_size, 3, n_points, 1)
        # knn_features2: [bs, in_channels, n_points, k]
        knn_features2 = batch_indexing(feat2, knn_indices_1in2)
        # features1_expand: [bs, in_channels, n_points, k]
        features1_expand = feat1[:, :, :, None].expand(batch_size, in_channels, n_points, self.k)
        # concatenated_features: [bs, 2 * in_channels + 3, n_points, k]
        concatenated_features = torch.cat([features1_expand, knn_features2, knn_xyz2_norm], dim=1)
        # p2p_cost (point-to-point cost): [bs, out_channels, n_points, k]
        p2p_cost = self.cost_mlp(concatenated_features)

        # weights2: [bs, out_channels, n_points, k]
        weights2 = self.weight_net2(knn_xyz2_norm)
        # p2n_cost (point-to-neighbor cost): [bs, out_channels, n_points]
        p2n_cost = torch.sum(weights2 * p2p_cost, dim=3)

        # Step2: for each point in xyz1, find its neighbors in xyz1
        if knn_indices_1in1 is not None:
            assert knn_indices_1in1.shape[:2] == torch.Size([batch_size, n_points])
            assert knn_indices_1in1.shape[2] >= self.k
            knn_indices_1in1 = knn_indices_1in1[:, :, :self.k]
        else:
            knn_indices_1in1 = k_nearest_neighbor(input_xyz=xyz1, query_xyz=xyz1, k=self.k)  # [bs, n_points, k]
        # knn_xyz1: [bs, 3, n_points, k]
        knn_xyz1 = batch_indexing(xyz1, knn_indices_1in1)
        # knn_xyz1_norm: [bs, 3, n_points, k]
        knn_xyz1_norm = knn_xyz1 - xyz1.view(batch_size, 3, n_points, 1)

        # weights1: [bs, out_channels, n_points, k]
        weights1 = self.weight_net1(knn_xyz1_norm)
        # n2n_cost: [bs, out_channels, n_points, k]
        n2n_cost = batch_indexing(p2n_cost, knn_indices_1in1)
        # n2n_cost (neighbor-to-neighbor cost): [bs, out_channels, n_points]
        n2n_cost = torch.sum(weights1 * n2n_cost, dim=3)
        # align features (optional)
        n2n_cost = self.feat_aligner(n2n_cost)

        return n2n_cost

class FlowPredictor3D(nn.Module):
    def __init__(self, n_channels, norm=None, conv_last=True, k=16):
        super().__init__()
        self.point_conv1 = PointConv(in_channels=n_channels[0], out_channels=n_channels[1], norm=norm, k=k)
        self.point_conv2 = PointConv(in_channels=n_channels[1], out_channels=n_channels[2], norm=norm, k=k)
        self.mlp = MLP2d(n_channels[2], [n_channels[2], n_channels[3]])
        self.flow_feat_dim = n_channels[3]

        if conv_last:
            self.conv_last = nn.Conv2d(n_channels[3], 3, kernel_size=1)
        else:
            self.conv_last = None

    def forward(self, xyz, feat, knn_indices, valid_knn_mask):
        """
        :param xyz: 3D locations of points, [B, 3, H, W]
        :param feat: features of points, [B, in_channels, H, W]
        :return flow_feat: [B, 64, H, W]
        :return flow: [B, 3, H, W]
        """
        feat = self.point_conv1(xyz, feat, knn_indices = knn_indices, valid_knn_mask = valid_knn_mask)  # [bs, 128, H, W]
        feat = self.point_conv2(xyz, feat, knn_indices = knn_indices, valid_knn_mask = valid_knn_mask)  # [bs, 128, H, W]
        feat = self.mlp(feat)  # [bs, 64, H, W]

        if self.conv_last is not None:
            flow = self.conv_last(feat)  # [bs, 3, H, W]
            return feat, flow
        else:
            return feat


class FlowPredictor3DS(nn.Module):
    def __init__(self, n_channels, norm=None, conv_last=False, k=16):
        super().__init__()
        self.point_conv1 = PointConvS(in_channels=n_channels[0], out_channels=n_channels[1], norm=norm, k=k)
        self.point_conv2 = PointConvS(in_channels=n_channels[1], out_channels=n_channels[2], norm=norm, k=k)
        self.mlp = MLP1d(n_channels[2], [n_channels[2], n_channels[3]])
        self.flow_feat_dim = n_channels[3]

        if conv_last:
            self.conv_last = nn.Conv1d(n_channels[3], 3, kernel_size=1)
        else:
            self.conv_last = None

    def forward(self, xyz, feat, knn_indices, mask):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param feat: features of points, [batch_size, in_channels, n_points]
        :param knn_indices: knn indices of points, [batch_size, n_points, k]
        :return flow_feat: [batch_size, 64, n_points]
        :return flow: [batch_size, 3, n_points]
        """
        feat = self.point_conv1(xyz, feat, knn_indices=knn_indices)
        feat = self.point_conv2(xyz, feat, knn_indices=knn_indices)
        feat = self.mlp(feat)

        if self.conv_last is not None:
            flow = self.conv_last(feat)
            return feat, flow
        else:
            return feat