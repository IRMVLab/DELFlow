import torch
import torch.nn as nn
from .utils import MLP2d, LayerNormCF1d, batch_indexing, knn_grouping_2d, mask_batch_selecting


class PointConvS(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, act='leaky_relu', k=16):
        super().__init__()
        self.k = k

        self.weight_net = MLP2d(3, [8, 16], act=act)
        self.linear = nn.Linear(16 * (in_channels + 3), out_channels)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels, affine=True)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels, affine=True)
        elif norm == 'layer_norm':
            self.norm_fn = LayerNormCF1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if act == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % act)

    def forward(self, xyz, features, sampled_xyz=None, knn_indices=None):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param features: features of points, [batch_size, in_channels, n_points]
        :param sampled_xyz: 3D locations of sampled points, [batch_size, 3, n_samples]
        :return weighted_features: features of sampled points, [batch_size, out_channels, n_samples]
        """
        if sampled_xyz is None:
            sampled_xyz = xyz

        bs, n_samples = sampled_xyz.shape[0], sampled_xyz.shape[-1]
        features = torch.cat([xyz, features], dim=1)  # [bs, in_channels + 3, n_points]
        features_cl = features.transpose(1, 2)  # [bs, n_points, n_channels + 3]

        # Calculate k nearest neighbors
        if knn_indices is None:
            knn_indices = k_nearest_neighbor(xyz, sampled_xyz, self.k)  # [bs, n_samples, k]
        else:
            assert knn_indices.shape[:2] == torch.Size([bs, n_samples])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, :self.k]

        # Calculate weights
        knn_xyz = batch_indexing(xyz, knn_indices)  # [bs, 3, n_samples, k]
        knn_xyz_norm = knn_xyz - sampled_xyz[:, :, :, None]  # [bs, 3, n_samples, k]
        weights = self.weight_net(knn_xyz_norm)  # [bs, n_weights, n_samples, k]

        # Calculate weighted features
        weights = weights.transpose(1, 2)  # [bs, n_samples, n_weights, k]
        knn_features = batch_indexing(features_cl, knn_indices, layout='channel_last')  # [bs, n_samples, k, 3 + in_channels]
        out = torch.matmul(weights, knn_features)  # [bs, n_samples, n_weights, 3 + in_channels]
        out = out.view(bs, n_samples, -1)  # [bs, n_samples, (3 + in_channels) * n_weights]
        out = self.linear(out)  # [bs, n_samples, out_channels]
        out = self.act_fn(self.norm_fn(out.transpose(1, 2)))  # [bs, out_channels, n_samples]

        return out
    

class PointConv(nn.Module):

    def __init__(self, in_channels, out_channels, ks = [10, 20], dist = 100.0, norm=None, act='leaky_relu', k=16):
        super().__init__()
        self.k = k
        self.kernel_size = ks
        self.distance = dist
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_net = MLP2d(3, [8, 16], act=act)
        self.linear = nn.Linear(16 * (in_channels + 3), out_channels)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if act == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown act function: %s' % act)

    def forward(self, xyz, features, sampled_xyz = None, knn_indices=None, valid_knn_mask = None):
        
        """
        :param xyz: [batch_size, 3, H, W]
        :param features: [batch_size, C, H, W]
        :param sampled_xyz: [batch_size, 3, h, w]
        :return: out: [batch_size, C', h, w]
        """
        if sampled_xyz == None:
            sampled_xyz = xyz

        
        B, C, H, W = features.shape
        h, w = sampled_xyz.shape[2:]
        features = torch.cat([xyz, features], dim=1).reshape(B, C + 3, H * W)  # [B, in_channels + 3, H, W]
        features_cl = features.transpose(1, 2) # [B, H * W, in_channels + 3]
        
        #################   Calculate k nearest neighbors
        if knn_indices is None:
            # [B, h*w, k], [B, h*w, k]
            knn_indices, valid_knn_mask = knn_grouping_2d(sampled_xyz, xyz, self.k)
        else:
            assert knn_indices.shape[:2] == torch.Size([B, h * w])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, :self.k] 
            valid_knn_mask = valid_knn_mask[:, :, :self.k]
            # valid_mask = valid_mask[:, :, :self.k] 
                                                 
        # Calculate weights
        knn_xyz = mask_batch_selecting(xyz, knn_indices, valid_knn_mask)  # [B, 3, h * w, k]
        new_xyz = sampled_xyz.reshape(B, 3,-1)  # [B, 3, h*w]       
        knn_xyz_norm = knn_xyz - new_xyz[:, :, :, None]   # [B, 3, h*w, k]
        weights = self.weight_net(knn_xyz_norm)  # [B, n_weights, h*w, k]
        
        # Calculate weighted features
        weights = weights.transpose(1, 2) # [B, h*w, n_weights, k]
        knn_features = mask_batch_selecting(features_cl, knn_indices, valid_knn_mask, layout='channel_last')  # [B, h*w, k, C + 3] 

        out = torch.matmul(weights, knn_features)  # [B, h*w, n_weights, C+3]
        out = out.view(B, h*w, -1)  # [B, h*w, n_weights*(C+3)]
        out = self.linear(out)  # [B, h*w, out_channels]
        out = self.act_fn(self.norm_fn(out.transpose(1, 2)))  # [B, out_channels, h * w]
        
        # out = torch.reshape(out, [B, -1, h, w])
        out = out.view(B, -1, h, w)
        return out
    
    
class PointConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, act='leaky_relu', k=16):
        super().__init__()
        self.k = k        
        self.mlp = MLP2d(in_channels, [out_channels], norm, act)
        self.weight_net = MLP2d(3, [8, 32, out_channels], act='relu')

    def forward(self, xyz, features, sampled_xyz=None, knn_indices=None, valid_knn_mask = None):
        """
        :param xyz: [batch_size, 3, H, W]
        :param features: [batch_size, C, H, W]
        :param sampled_xyz: [batch_size, 3, h, w]
        :return: out: [batch_size, C', h, w]
        """
        
        if sampled_xyz is None:
            sampled_xyz = xyz

        B, C, H, W = features.shape
        h, w = sampled_xyz.shape[2:]
        
        # Calculate k nearest neighbors
        if knn_indices is None:
            # [B, h*w, k], [B, h*w, k]
            knn_indices, valid_knn_mask = knn_grouping_2d(sampled_xyz, xyz, self.k)
        else:
            assert knn_indices.shape[:2] == torch.Size([B, h * w])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, :self.k] 
            valid_knn_mask = valid_knn_mask[:, :, :self.k]
        
        # Calculate weights
        knn_xyz = mask_batch_selecting(xyz, knn_indices, valid_knn_mask)  # [B, 3, h * w, k]
        new_xyz = sampled_xyz.reshape(B, 3,-1)  # [B, 3, h*w]       
        knn_offset = knn_xyz - new_xyz[:, :, :, None]   # [B, 3, h*w, k]
   
        features = self.mlp(features) # [B, C_out, H, W]
        features = mask_batch_selecting(features, knn_indices, valid_knn_mask)    # [B, C_out, h * w, k]
        features = features * self.weight_net(knn_offset) # [B, C_out, h * w, k] * [B, C_out, h*w, k]
        features = torch.max(features, dim=-1)[0]  # [B, C_out, h*w]
        features = features.view(B, -1, h, w)

        return features