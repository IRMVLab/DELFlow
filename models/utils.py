import time
import torch
import torch.nn as nn
from torch.nn.functional import grid_sample, interpolate, pad
from ops_pytorch.fused_conv_select.fused_conv_select_k import fused_conv_select_k
from .csrc import correlation2d, k_nearest_neighbor, furthest_point_sampling

def mask_batch_selecting(batched_data: torch.Tensor, batched_indices: torch.Tensor, mask : torch.Tensor, layout='channel_first'):
    select_data = batch_indexing(batched_data, batched_indices, layout)
    # print(select_data.shape)
    if layout == 'channel_first':
        mask = mask.unsqueeze(1)
    else:
        mask = mask.unsqueeze(-1)
    return select_data * mask.detach()

def batch_indexing(batched_data: torch.Tensor, batched_indices: torch.Tensor, layout='channel_first'):
    def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, C, N]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, C, I1, I2, ..., Im]
        """
        def product(arr):
            p = 1
            for i in arr:
                p *= i
            return p
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size, n_channels = batched_data.shape[:2]
        indices_shape = list(batched_indices.shape[1:])
        batched_indices = batched_indices.reshape([batch_size, 1, -1])
        batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
        result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
        result = result.view([batch_size, n_channels] + indices_shape)
        return result

    def batch_indexing_channel_last(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, N, C]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, I1, I2, ..., Im, C]
        """
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)  # [bs, I1, I2, ..., Im]
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]

    if layout == 'channel_first':
        if len(batched_data.shape) == 4:
            B, C, H, W = batched_data.shape
            batched_data = batched_data.reshape(B, C, H * W)
        return batch_indexing_channel_first(batched_data, batched_indices)
    elif layout == 'channel_last':
        if len(batched_data.shape) == 4:
            B, H, W, C = batched_data.shape
            batched_data = batched_data.reshape(B, H * W, C)
        return batch_indexing_channel_last(batched_data, batched_indices)
    else:
        raise ValueError
    
def get_hw_idx(B, out_H, out_W, stride_H = 1, stride_W = 1):

    H_idx = torch.reshape(torch.arange(0, out_H * stride_H, stride_H, device = "cuda", dtype = torch.int), [1, -1, 1, 1]).expand(B, out_H, out_W, 1)
    W_idx = torch.reshape(torch.arange(0, out_W * stride_W, stride_W, device = "cuda", dtype = torch.int), [1, 1, -1, 1]).expand(B, out_H, out_W, 1)

    idx_n2 = torch.cat([H_idx, W_idx], dim = -1).reshape(B, -1, 2)

    return idx_n2.contiguous()

def normalize_image(image):
    
    norm_mean = torch.tensor([123.675, 116.280, 103.530], device=image.device)
    norm_std = torch.tensor([58.395, 57.120, 57.375], device=image.device)
    image = image - norm_mean.reshape(1, 3, 1, 1)
    image = image / norm_std.reshape(1, 3, 1, 1)
    
    return image

def build_labels(target, indices):
    labels, masks = [], []
    for index in indices:
        labels.append(batch_indexing(target, index))
        masks.append(torch.ones_like(labels, dtype=torch.bool))
    return labels, masks   

def build_pc_pyramid(pc1, pc2, n_samples_list):
    batch_size, _, n_points = pc1.shape

    # sub-sampling point cloud
    pc_both = torch.cat([pc1, pc2], dim=0)
    sample_index_both = furthest_point_sampling(pc_both.transpose(1, 2), max(n_samples_list))  # 1/4
    sample_index1 = sample_index_both[:batch_size]
    sample_index2 = sample_index_both[batch_size:]

    # build point cloud pyramid
    lv0_index = torch.arange(n_points, device=pc1.device)
    lv0_index = lv0_index[None, :].expand(batch_size, n_points)
    xyzs1, xyzs2, sample_indices1, sample_indices2 = [pc1], [pc2], [lv0_index], [lv0_index]

    for n_samples in n_samples_list:  # 1/4
        sample_indices1.append(sample_index1[:, :n_samples])
        sample_indices2.append(sample_index2[:, :n_samples])
        xyzs1.append(batch_indexing(pc1, sample_index1[:, :n_samples]))
        xyzs2.append(batch_indexing(pc2, sample_index2[:, :n_samples]))

    return xyzs1, xyzs2, sample_indices1, sample_indices2


@torch.no_grad()
def stride_sample(src, stride_H_list, stride_W_list):
    """
    Input:
        src: input points data, [B, ..., H, W]
    Return:
        dsts:, list of sampled pc
    """   

    dsts = [src]

    for s_h, s_w in zip(stride_H_list, stride_W_list):
        dsts.append(dsts[-1][..., ::s_h, ::s_w].contiguous())

    return dsts

@torch.no_grad()
def stride_sample_pc(pc1, pc2, stride_H_list, stride_W_list):
    """
    Input:
        pc1: input points data, [B, 3, H, W]
        pc2: input points data, [B, 3, H, H]
        stride_list: list of sampling strides
    Return:
        xyzs:, list of sampled pc
    """   

    xyzs1 = [pc1]; xyzs2 = [pc2];

    for s_h, s_w in zip(stride_H_list, stride_W_list):
        xyzs1.append(xyzs1[-1][..., ::s_h, ::s_w].contiguous())
        xyzs2.append(xyzs2[-1][..., ::s_h, ::s_w].contiguous())

    return xyzs1, xyzs2

@torch.no_grad()
def knn_grouping_2d(query_xyz: torch.Tensor, input_xyz: torch.Tensor, k: int, ks = [10, 20], idx_fetching = None):
    """
    Calculate k-nearest neighbor for each query.
    :param input_xyz: a set of points, [batch_size, n_points, 3] or [batch_size, 3, n_points]
    :param query_xyz: a set of centroids, [batch_size, n_queries, 3] or [batch_size, 3, n_queries]
    :param k: int
    :return: indices of k-nearest neighbors, [batch_size, n_queries, k]
    """ 
    
    B, C, H, W = input_xyz.shape
    h = query_xyz.shape[2]
    w = query_xyz.shape[3]
    n_sampled = h * w
    
    # // assert H >= h and W >= w, "input size must be no smaller than query size"

    #################   Calculate k nearest neighbors
    if H > h:   # downsample, input size > query size
        idx_n2 = get_hw_idx(B, h, w, H // h, W // w) # [b, -1, 2]
    elif H <= h:
        idx_n2 = get_hw_idx(B, h, w)
        
    random_HW = torch.arange(ks[0] * ks[1], device = "cuda", dtype = torch.int)    
    input_xyz_hw3 = input_xyz.permute(0, 2, 3, 1).contiguous()    # [B, H, W, 3]
    query_xyz_hw3 = query_xyz.permute(0, 2, 3, 1).contiguous()    # [B, H, W, 3]
    
    if idx_fetching is None:
        idx_fetching = idx_n2
    
    
    select_b_idx = torch.zeros(B, n_sampled, k, 1, device = 'cuda').long().detach()  # (B, n_sampled, k, 1)
    select_h_idx = torch.zeros(B, n_sampled, k, 1, device = 'cuda').long().detach()
    select_w_idx = torch.zeros(B, n_sampled, k, 1, device = 'cuda').long().detach()
    valid_mask = torch.zeros(B, n_sampled, k, 1, device = 'cuda').float().detach()   # (B, n_sampled, k, 1)
    
    # with torch.no_grad():
    # Sample n' points from input n points 
    
    if H > h:
        select_b_idx, select_h_idx, select_w_idx, valid_mask = \
        fused_conv_select_k(input_xyz_hw3, input_xyz_hw3, idx_n2, idx_fetching, random_HW, H, W, n_sampled, ks[0], ks[1], k, 1, 100,\
            1, 1, select_b_idx, select_h_idx, select_w_idx, valid_mask, H, W)
    elif H <= h:
        stride_H = h // H; stride_W = w // W
        select_b_idx, select_h_idx, select_w_idx, valid_mask = \
        fused_conv_select_k(query_xyz_hw3, input_xyz_hw3, idx_n2, idx_fetching, random_HW, h, w, n_sampled, ks[0], ks[1], k, 1, 100,\
            stride_H, stride_W, select_b_idx, select_h_idx, select_w_idx, valid_mask, H, W)   
            
    neighbor_idx = select_h_idx * W + select_w_idx   # [B, h*w, k, 1]
    neighbor_idx = neighbor_idx.squeeze(-1)
    
    valid_mask = valid_mask.squeeze(-1)
    # [B, h*w, k], # [B, h*w, k]
    return neighbor_idx, valid_mask
    
def warppingProject(pc, fx, fy, cx, cy, out_h, out_w):
    """ 
    To project the pc to the 2D plane
    
    Args:
        @pc: (B, N, 3)    
    """
    # (pc[..., 0] / (pc[..., 2] + 1e-10)) * fx + cx
    B = pc.shape[0]
    pc_w = torch.round((pc[..., 0] / (pc[..., 2] + 1e-10)) * fx + cx).int()
    pc_h = torch.round((pc[..., 1] / (pc[..., 2] + 1e-10)) * fy + cy).int()
    width = torch.clamp((torch.reshape(pc_w, (B, -1, 1))), 0, out_w - 1)  
    height = torch.clamp((torch.reshape(pc_h, (B, -1, 1))), 0, out_h - 1)
    indices = torch.cat([height, width], -1)
    return indices.contiguous()

def forwarp_3d(xyz1, xyz2, upsampled_flow, camera_info):
    """
    :param xyz1: [B, 3, h, w]
    :param xyz2: [B, 3, h, w]
    :param upsampled_flow: [B, 3, h, w]
    :param intrinsics: [B, 4]
    :return warped pc1: [B, 3, h, w]
    :return warping idx: [B, h * w, 2]
    """
    B, _, h, w = xyz1.shape
    warped_xyz1 = xyz1 + upsampled_flow # (B, 3, h, w)
    intrinsics = camera_info['intrinsics']  # (B, 4)
    origin_h, origin_w = camera_info['sensor_h'], camera_info['sensor_w']
    fxs, fys, cxs, cys = torch.unbind(intrinsics[:, :, None], 1) # (B, 1) (B, 1) (B, 1) (B, 1)
    fxs = fxs / (origin_w / w)
    cxs = cxs / (origin_w / w)
    fys = fys / (origin_h / h)
    cys = cys / (origin_h / h)

    # projectIndexes: [B, h * w, 2]
    projectIndexes = warppingProject(torch.reshape(warped_xyz1.permute(0, 2, 3, 1).contiguous(), [B, h * w, 3]), fxs, fys, cxs, cys, h, w)
    xyz1_bn3 = torch.reshape(xyz1.permute(0, 2, 3, 1).contiguous(), [B, h * w, 3])
    
    #! EDITED
    # invalidMask = (torch.all(torch.eq(xyz1_bn3, 0), dim=-1, keepdim=True)).repeat([1, 1, 2])  # (B, h*w, 2)
    invalidMask = torch.linalg.norm(xyz1_bn3, dim=-1, keepdim=True) < 1e-5 # [B, N, 1]
    maskedProjectIndexes = torch.where(invalidMask.expand(-1, -1, 2), torch.ones_like(projectIndexes, dtype = torch.int32) * -1, projectIndexes) # (B, h*w, 2)
    maskedProjectIndexes = maskedProjectIndexes.contiguous()

    return warped_xyz1, maskedProjectIndexes

class LayerNormCF1d(nn.Module):
    """LayerNorm that supports the channel_first data format."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x

class Conv1dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, act='leaky_relu'):
        super().__init__()

        self.conv_fn = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels, affine=True)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm == 'instance_norm_affine':
            self.norm_fn = nn.InstanceNorm1d(out_channels, affine=True)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if act == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown act function: %s' % act)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.act_fn(x)
        return x


class Conv2dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, act='leaky_relu'):
        super().__init__()

        self.conv_fn = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif norm == 'instance_norm_affine':
            self.norm_fn = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if act == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown act function: %s' % act)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.act_fn(x)
        return x


class MLP1d(nn.Module):
    def __init__(self, in_channels, mlps, norm=None, act='leaky_relu'):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(mlps, list)
        n_channels = [in_channels] + mlps

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.convs.append(Conv1dNormRelu(in_channels, out_channels, norm=norm, act=act))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, mlps, norm=None, act='leaky_relu'):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(mlps, list)
        n_channels = [in_channels] + mlps

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.convs.append(Conv2dNormRelu(in_channels, out_channels, norm=norm, act=act))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


def knn_interpolation(input_xyz, input_features, query_xyz, k=3):
    """
    :param input_xyz: 3D locations of input points, [batch_size, 3, n_inputs]
    :param input_features: features of input points, [batch_size, n_features, n_inputs]
    :param query_xyz: 3D locations of query points, [batch_size, 3, n_queries]
    :param k: k-nearest neighbor, int
    :return interpolated features: [batch_size, n_features, n_queries]
    """
    knn_indices = k_nearest_neighbor(input_xyz, query_xyz, k)  # [batch_size, n_queries, 3]
    knn_xyz = batch_indexing(input_xyz, knn_indices)  # [batch_size, 3, n_queries, k]
    knn_dists = torch.linalg.norm(knn_xyz - query_xyz[..., None], dim=1).clamp(1e-8)  # [bs, n_queries, k]
    knn_weights = 1.0 / knn_dists  # [bs, n_queries, k]
    knn_weights = knn_weights / torch.sum(knn_weights, dim=-1, keepdim=True)  # [bs, n_queries, k]
    knn_features = batch_indexing(input_features, knn_indices)  # [bs, n_features, n_queries, k]
    interpolated = torch.sum(knn_features * knn_weights[:, None, :, :], dim=-1)  # [bs, n_features, n_queries]

    return interpolated


def backwarp_3d(xyz1, xyz2, flow12, k=3):
    """
    :param xyz1: 3D locations of points1, [batch_size, 3, n_points]
    :param xyz2: 3D locations of points2, [batch_size, 3, n_points]
    :param flow12: scene flow, [batch_size, 3, n_points]
    :param k: k-nearest neighbor, int
    """
    xyz1_warp = xyz1 + flow12
    flow21 = knn_interpolation(xyz1_warp, -flow12, query_xyz=xyz2, k=k)
    xyz2_warp = xyz2 + flow21
    return xyz2_warp

mesh_grid_cache = {}
def mesh_grid(n, h, w, device, channel_first=True):
    global mesh_grid_cache
    str_id = '%d,%d,%d,%s,%s' % (n, h, w, device, channel_first)
    if str_id not in mesh_grid_cache:
        x_base = torch.arange(0, w, dtype=torch.float32, device=device)[None, None, :].expand(n, h, w)
        y_base = torch.arange(0, h, dtype=torch.float32, device=device)[None, None, :].expand(n, w, h)  # NWH
        grid = torch.stack([x_base, y_base.transpose(1, 2)], 1)  # B2HW
        if not channel_first:
            grid = grid.permute(0, 2, 3, 1)  # BHW2
        mesh_grid_cache[str_id] = grid
    return mesh_grid_cache[str_id]


def backwarp_2d(x, flow12, padding_mode):
    def norm_grid(g):
        grid_norm = torch.zeros_like(g)
        grid_norm[:, 0, :, :] = 2.0 * g[:, 0, :, :] / (g.shape[3] - 1) - 1.0
        grid_norm[:, 1, :, :] = 2.0 * g[:, 1, :, :] / (g.shape[2] - 1) - 1.0
        return grid_norm.permute(0, 2, 3, 1)

    assert x.size()[-2:] == flow12.size()[-2:], "{} and {}".format(x.shape, flow12.shape)
    batch_size, _, image_h, image_w = x.size()
    grid = mesh_grid(batch_size, image_h, image_w, device=x.device)
    grid = norm_grid(grid + flow12)

    return grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)


def convex_upsample(flow, mask, scale_factor=8):
    """
    Upsample flow field [image_h / 4, image_w / 4, 2] -> [image_h, image_w, 2] using convex combination.
    """
    batch_size, _, image_h, image_w = flow.shape
    mask = mask.view(batch_size, 1, 9, scale_factor, scale_factor, image_h, image_w)
    mask = torch.softmax(mask, dim=2)

    up_flow = torch.nn.functional.unfold(flow * scale_factor, [3, 3], padding=1)
    up_flow = up_flow.view(batch_size, 2, 9, 1, 1, image_h, image_w)
    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

    return up_flow.reshape(batch_size, 2, image_h * scale_factor, image_w * scale_factor)


def resize_flow2d(flow, target_h, target_w):
    origin_h, origin_w = flow.shape[2:]
    if target_h == origin_h and target_w == origin_w:
        return flow
    flow = interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
    flow[:, 0] *= target_w / origin_w
    flow[:, 1] *= target_h / origin_h
    return flow


def resize_to_64x(inputs, target):
    n, c, h, w = inputs.shape

    if h % 64 == 0 and w % 64 == 0:
        return inputs, target

    resized_h, resized_w = ((h + 63) // 64) * 64, ((w + 63) // 64) * 64
    inputs = interpolate(inputs, size=(resized_h, resized_w), mode='bilinear', align_corners=True)

    if target is not None:
        target = interpolate(target, size=(resized_h, resized_w), mode='bilinear', align_corners=True)
        target[:, 0] *= resized_w / w
        target[:, 1] *= resized_h / h

    return inputs, target


def pad_to_64x(inputs, target):
    n, c, h, w = inputs.shape

    pad_h = 0 if h % 64 == 0 else 64 - (h % 64)
    pad_w = 0 if w % 64 == 0 else 64 - (w % 64)

    if pad_h == 0 and pad_w == 0:
        return inputs, target

    inputs = pad(inputs, [0, pad_w, 0, pad_h], value=0)
    if target is not None:
        target = pad(target, [0, pad_w, 0, pad_h], value=0)

    return inputs, target

## Newly created
@torch.no_grad()
def project_3d_to_2d(pc, camera_info, h, w):
    assert pc.shape[1] == 3  # channel first
    
    dense = camera_info['type'] == "dense"
    intrinsics = camera_info["intrinsics"]  # (B, 4)
    origin_h, origin_w = camera_info['sensor_h'], camera_info['sensor_w']
    # (B, 1) (B, 1) (B, 1) (B, 1)
    fxs, fys, cxs, cys = torch.unbind(intrinsics[:, :, None], 1)    
    
    if dense:
        B, _, h, w = pc.shape
        valid_mask = torch.linalg.norm(pc, dim=1, keepdim=True) > 1e-5  # [B, 1, h, w]
        pc = torch.reshape(pc, [B, 3, h * w])

        pc_x, pc_y, pc_z = pc[:, 0, :], pc[:, 1, :], pc[:, 2, :]  # (B, n) (B, n) (B, n)
        image_x = torch.reshape(cxs + (pc_x / (pc_z + 10e-10)) * fxs, [B, 1, h, w])
        image_y = torch.reshape(cys + (pc_y / (pc_z + 10e-10)) * fys, [B, 1, h, w])
        image_x *= (w - 1) / (origin_w - 1)
        image_y *= (h - 1) / (origin_h - 1)
        projected_uv = torch.cat([image_x, image_y], dim=1)  # [B, 2, h, w]
        uv = torch.where(
            valid_mask.expand(-1, 2, -1, -1), projected_uv, torch.zeros_like(projected_uv)
        )
    else:
        B, _, n_points = pc.shape
        valid_mask = torch.ones([B, 1, n_points], dtype=torch.bool)
        pc_x, pc_y, pc_z = pc[:, 0, :], pc[:, 1, :], pc[:, 2, :]
        image_x = cxs + (pc_x / (pc_z + 10e-10)) * fxs
        image_y = cys + (pc_y / (pc_z + 10e-10)) * fys        
        image_x *= (w - 1) / (origin_w - 1)
        image_y *= (h - 1) / (origin_h - 1)      
        uv = torch.cat([image_x, image_y], dim=1)  # [B, 2, n]
      
    return uv, valid_mask.squeeze(1)


def project_pc2image(pc, camera_info):
    assert pc.shape[1] == 3  # channel first
    batch_size, n_points = pc.shape[0], pc.shape[-1]

    if isinstance(camera_info['cx'], torch.Tensor):
        cx = camera_info['cx'][:, None].expand([batch_size, n_points])
        cy = camera_info['cy'][:, None].expand([batch_size, n_points])
    else:
        cx = camera_info['cx']
        cy = camera_info['cy']

    if camera_info['projection_mode'] == 'perspective':
        f = camera_info['f'][:, None].expand([batch_size, n_points])
        pc_x, pc_y, pc_z = pc[:, 0, :], pc[:, 1, :], pc[:, 2, :]
        image_x = cx + (f / pc_z) * pc_x
        image_y = cy + (f / pc_z) * pc_y
    elif camera_info['projection_mode'] == 'parallel':
        image_x = pc[:, 0, :] + cx
        image_y = pc[:, 1, :] + cy
    else:
        raise NotImplementedError

    return torch.cat([
        image_x[:, None, :],
        image_y[:, None, :],
    ], dim=1)



def mask_grid_sample_wrapper(feat_2d, xy, xy_mask):
    """
    feat2d: B, C, H, W
    xy: B, 2, H, W
    xy_mask: B, H, W
    """
    
    image_h, image_w = feat_2d.shape[2:]
    new_x = 2.0 * xy[:, 0] / (image_w - 1) - 1.0  # [B, H, W]
    new_y = 2.0 * xy[:, 1] / (image_h - 1) - 1.0  # [B, H, W]
    new_xy = torch.cat([new_x[:, :, :, None], new_y[:, :, :, None]], dim=-1)  # [B, H, W, 2]
    result = grid_sample(feat_2d, new_xy, 'bilinear', align_corners=True)  # [B, n_channels, H, W]
    xy_mask = xy_mask.unsqueeze(1)
    return result * xy_mask.detach()

def grid_sample_wrapper(feat_2d, xy):
    image_h, image_w = feat_2d.shape[2:]
    new_x = 2.0 * xy[:, 0] / (image_w - 1) - 1.0  # [bs, n_points]
    new_y = 2.0 * xy[:, 1] / (image_h - 1) - 1.0  # [bs, n_points]
    new_xy = torch.cat([new_x[:, :, None, None], new_y[:, :, None, None]], dim=-1)  # [bs, n_points, 1, 2]
    result = grid_sample(feat_2d, new_xy, 'bilinear', align_corners=True)  # [bs, n_channels, n_points, 1]
    return result[..., 0]

