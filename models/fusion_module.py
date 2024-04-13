import torch
import torch.nn as nn
from torch.nn.functional import softmax
from .utils import grid_sample_wrapper, mesh_grid, batch_indexing, get_hw_idx, mask_grid_sample_wrapper
from .utils import Conv1dNormRelu, Conv2dNormRelu
from ops_pytorch.gpu_threenn_sample.no_sort_knn import no_sort_knn
from .csrc import k_nearest_neighbor

########### Ours #############
class GlobalFuser(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, fusion_fn="sk", norm=None):
        super().__init__()

        self.interp = FusionInterp3D(in_channels_3d, k=1, norm=norm)
        self.mlps3d = Conv2dNormRelu(in_channels_2d, in_channels_2d, norm=norm)

        if fusion_fn == None:
            self.fuse2d = GatedChannelWiseFuser(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm
            )
            self.fuse3d = GatedChannelWiseFuser(
                in_channels_2d, in_channels_3d, in_channels_3d, "nchw", norm
            )
        elif fusion_fn == "add":
            self.fuse2d = AddFusion(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm
            )
            self.fuse3d = AddFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "nchw", norm
            )
        elif fusion_fn == "concat":
            self.fuse2d = ConcatFusion(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm
            )
            self.fuse3d = ConcatFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "nchw", norm
            )
        elif fusion_fn == "gated":
            self.fuse2d = GatedFusion(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm
            )
            self.fuse3d = GatedFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "nchw", norm
            )
        elif fusion_fn == "sk":
            self.fuse2d = SKFusion(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm, 2
            )
            self.fuse3d = SKFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "nchw", norm, 2
            )
        else:
            raise ValueError

    def forward(self, uv, uv_mask, feat_2d, feat_3d):
        """
        :param uv: 2D coordinates of points: [B, 2, H, W]
        :param feat_2d: features of images: [B, n_channels_2d, H, W]
        :param feat_3d: features of points: [B, n_channels_3d, H, W]
        :return: out2d: fused features of images: [B, n_channels_2d, H, W]
        :return: out3d: fused features of points: [B, n_channels_3d, H, W]
        """
        feat_2d = feat_2d.float()
        feat_3d = feat_3d.float()

        feat_3d_interp = self.interp(uv, feat_2d.detach(), feat_3d.detach())
        out2d = self.fuse2d(feat_2d, feat_3d_interp)

        # feat_2d_sampled = mask_grid_sample_wrapper(feat_2d.detach(), uv, uv_mask)
        # out3d = self.fuse3d(self.mlps3d(feat_2d_sampled.detach()), feat_3d, uv_mask)
        out3d = self.fuse3d(self.mlps3d(feat_2d.detach()), feat_3d)
        return out2d, out3d


class GatedChannelWiseFuser(nn.Module):
    def __init__(
        self,
        in_channels_2d,
        in_channels_3d,
        out_channels,
        feat_format="nchw",
        norm=None,
    ):
        super().__init__()

        if feat_format == "nchw":
            self.gate_i = Conv2dNormRelu(
                in_channels_2d, out_channels, norm=None, act="sigmoid"
            )
            self.gate_p = Conv2dNormRelu(
                in_channels_3d, out_channels, norm=None, act="sigmoid"
            )
            self.input_i = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.input_p = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
        elif feat_format == "ncm":
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv1dNormRelu(out_channels, 2, norm=None, act="sigmoid")
            self.mlp2 = Conv1dNormRelu(out_channels, 2, norm=None, act="sigmoid")
        else:
            raise ValueError

    def forward(self, img_feat, pc_feat):
        """
        :param img_feat: [batch_size, C1, H, W]
        :param pc_feat: [batch_size, C2, H, W]
        :return: fused_features: [batch_size, Out_channels, H, W]
        """
        gate_p = self.gate_p(pc_feat)
        gate_i = self.gate_i(img_feat)
        obj_fused = gate_p.mul(self.input_p(pc_feat)) + gate_i.mul(
            self.input_i(img_feat)
        )
        obj_feats = obj_fused.div(gate_p + gate_i)  # B x C x H x W

        return obj_feats


#### Sparse fusion ########


class GlobalFuserS(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, fusion_fn="sk", norm=None):
        super().__init__()

        self.interp = FusionAwareInterp(in_channels_3d, k=1, norm=norm)
        self.mlps3d = Conv1dNormRelu(in_channels_2d, in_channels_2d, norm=norm)

        if fusion_fn == "add":
            self.fuse2d = AddFusion(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm
            )
            self.fuse3d = AddFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "ncm", norm
            )
        elif fusion_fn == "concat":
            self.fuse2d = ConcatFusion(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm
            )
            self.fuse3d = ConcatFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "ncm", norm
            )
        elif fusion_fn == "gated":
            self.fuse2d = GatedFusion(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm
            )
            self.fuse3d = GatedFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "ncm", norm
            )
        elif fusion_fn == "sk":
            self.fuse2d = SKFusion(
                in_channels_2d, in_channels_3d, in_channels_2d, "nchw", norm, 2
            )
            self.fuse3d = SKFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "ncm", norm, 2
            )
        else:
            raise ValueError

    # @timer.timer_func
    def forward(self, uv, uv_mask, feat_2d, feat_3d):
        feat_2d = feat_2d.float()
        feat_3d = feat_3d.float()

        feat_3d_interp = self.interp(uv, feat_2d.detach(), feat_3d.detach())
        out2d = self.fuse2d(feat_2d, feat_3d_interp)

        feat_2d_sampled = grid_sample_wrapper(feat_2d.detach(), uv)
        out3d = self.fuse3d(self.mlps3d(feat_2d_sampled.detach()), feat_3d)

        return out2d, out3d


class FusionAwareInterp(nn.Module):
    def __init__(self, n_channels_3d, k=1, norm=None):
        super().__init__()
        self.k = k
        self.out_conv = Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm)
        self.score_net = nn.Sequential(
            Conv2dNormRelu(3, 16),  # [dx, dy, |dx, dy|_2, sim]
            Conv2dNormRelu(16, n_channels_3d, act="sigmoid"),
        )

    def forward(self, uv, feat_2d, feat_3d):

        bs, _, image_h, image_w = feat_2d.shape
        n_channels_3d = feat_3d.shape[1]

        grid = mesh_grid(bs, image_h, image_w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        knn_indices = k_nearest_neighbor(uv, grid, self.k)  # [B, HW, k]

        knn_uv, knn_feat3d = torch.split(
            batch_indexing(torch.cat([uv, feat_3d], dim=1), knn_indices),
            [2, n_channels_3d],
            dim=1,
        )

        knn_offset = knn_uv - grid[..., None]  # [B, 2, HW, k]
        knn_offset_norm = torch.linalg.norm(
            knn_offset, dim=1, keepdim=True
        )  # [B, 1, HW, k]

        score_input = torch.cat([knn_offset, knn_offset_norm], dim=1)  # [B, 4, HW, K]
        score = self.score_net(score_input)  # [B, n_channels_3d, HW, k]
        # score = softmax(score, dim=-1)  # [B, n_channels_3d, HW, k]

        final = score * knn_feat3d  # [B, n_channels_3d, HW, k]
        final = final.sum(dim=-1).reshape(
            bs, -1, image_h, image_w
        )  # [B, n_channels_3d, H, W]
        final = self.out_conv(final)

        return final


class FusionInterp3D(nn.Module):
    def __init__(self, n_channels_3d, k=1, ks=[10, 20], norm=None):
        super().__init__()
        self.k = k
        self.kernel_size = ks
        self.distance = 100.0
        self.out_conv = Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm)
        self.score_net = nn.Sequential(
            Conv2dNormRelu(3, 16),  # [dx, dy, |dx, dy|_2, sim]
            Conv2dNormRelu(16, n_channels_3d, act="sigmoid"),
        )

    @torch.no_grad()
    def get_neighbors(self, uv, grid, k):
        B, _, H, W = uv.shape
        idx_n2 = get_hw_idx(B, H, W)  # [b, -1, 2]
        random_HW = torch.arange(
            self.kernel_size[0] * self.kernel_size[1], device="cuda", dtype=torch.int
        )
        uvz = torch.cat([uv, torch.zeros(B, 1, H, W, device=uv.device)], dim=1)
        gridz = torch.cat([grid, torch.zeros(B, 1, H, W, device=grid.device)], dim=1)
        n_sampled = H * W
        select_b_idx = torch.zeros(B, n_sampled, k, 1, device="cuda").long().detach()
        select_h_idx = torch.zeros(B, n_sampled, k, 1, device="cuda").long().detach()
        select_w_idx = torch.zeros(B, n_sampled, k, 1, device="cuda").long().detach()
        valid_mask = torch.zeros(B, n_sampled, k, 1, device="cuda").float().detach()

        with torch.no_grad():
            # Sample n' points from input n points
            select_b_idx, select_h_idx, select_w_idx, valid_mask = no_sort_knn(
                gridz.permute(0, 2, 3, 1).contiguous(),
                uvz.permute(0, 2, 3, 1).contiguous(),
                idx_n2.contiguous(),
                random_HW,
                H,
                W,
                n_sampled,
                self.kernel_size[0],
                self.kernel_size[1],
                k,
                0,
                self.distance,
                1,
                1,
                select_b_idx,
                select_h_idx,
                select_w_idx,
                valid_mask,
            )

        neighbor_idx = select_h_idx * W + select_w_idx
        # B, HW, k
        return neighbor_idx.squeeze(-1)

    def forward(self, uv, feat_2d, feat_3d):
        """
        :param uv: [B, 2, H, W]
        :param feat_2d: [B, n_channels_2d, H, W]
        :param feat_3d: [B, n_channels_3d, H, W]
        :return: final: [B, n_channels_3d, h, w]
        """

        B, _, H, W = feat_2d.shape
        n_channels_3d = feat_3d.shape[1]

        grid = mesh_grid(B, H, W, uv.device)  # [B, 2, H, W]

        knn_indices = self.get_neighbors(uv, grid, self.k)  # [B, HW, k]

        # [B, 2, HW, k], [B, n_channels_3d, HW, k]
        knn_uv, knn_feat3d = torch.split(
            batch_indexing(
                torch.cat([uv, feat_3d], dim=1).reshape(B, -1, H * W), knn_indices
            ),
            [2, n_channels_3d],
            dim=1,
        )

        grid = grid.reshape([B, 2, -1])
        knn_offset = knn_uv - grid[..., None]  # [B, 2, HW, k]
        knn_offset_norm = torch.linalg.norm(
            knn_offset, dim=1, keepdim=True
        )  # [B, 1, HW, k]

        score_input = torch.cat([knn_offset, knn_offset_norm], dim=1)  # [B, 3, HW, K]
        score = self.score_net(score_input)  # [B, n_channels_3d, HW, k]
        # score = softmax(score, dim=-1)  # [B, n_channels_3d, HW, k]

        final = score * knn_feat3d  # [B, n_channels_3d, HW, k]
        final = final.sum(dim=-1).reshape(B, -1, H, W)  # [B, n_channels_3d, H, W]
        final = self.out_conv(final)

        return final


class FusionAwareInterpCVPR(nn.Module):
    def __init__(self, n_channels_2d, n_channels_3d, k=3, norm=None) -> None:
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(n_channels_3d + 3, n_channels_3d, norm=norm),
            Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm),
            Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm),
        )

    def forward(self, uv, feat_2d, feat_3d):
        bs, _, h, w = feat_2d.shape

        grid = mesh_grid(bs, h, w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        with torch.no_grad():
            nn_indices = k_nearest_neighbor(uv, grid, k=1)[..., 0]  # [B, HW]
            nn_feat2d = batch_indexing(
                grid_sample_wrapper(feat_2d, uv), nn_indices
            )  # [B, n_channels_2d, HW]
            nn_feat3d = batch_indexing(
                feat_3d, nn_indices
            )  # [B, n_channels_3d, HW]
            nn_offset = (
                batch_indexing(uv, nn_indices) - grid
            )  # [B, 2, HW]
            nn_corr = torch.mean(
                nn_feat2d * feat_2d.reshape(bs, -1, h * w), dim=1, keepdim=True
            )  # [B, 1, HW]

        feat = torch.cat(
            [nn_offset, nn_corr, nn_feat3d], dim=1
        )  # [B, n_channels_3d + 3, HW]
        feat = feat.reshape([bs, -1, h, w])  # [B, n_channels_3d + 3, H, W]
        final = self.mlps(feat)  # [B, n_channels_3d, H, W]

        return final


class AddFusion(nn.Module):
    def __init__(
        self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None
    ):
        super().__init__()

        if feat_format == "nchw":
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
        elif feat_format == "ncm":
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
        else:
            raise ValueError

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feat_2d, feat_3d):
        return self.relu(self.align1(feat_2d) + self.align2(feat_3d))


class ConcatFusion(nn.Module):
    def __init__(
        self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None
    ):
        super().__init__()

        if feat_format == "nchw":
            self.mlp = Conv2dNormRelu(
                in_channels_2d + in_channels_3d, out_channels, norm=norm
            )
        elif feat_format == "ncm":
            self.mlp = Conv1dNormRelu(
                in_channels_2d + in_channels_3d, out_channels, norm=norm
            )
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        return self.mlp(torch.cat([feat_2d, feat_3d], dim=1))


class GatedFusion(nn.Module):
    def __init__(
        self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None
    ):
        super().__init__()

        if feat_format == "nchw":
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv2dNormRelu(out_channels, 2, norm=None, act="sigmoid")
            self.mlp2 = Conv2dNormRelu(out_channels, 2, norm=None, act="sigmoid")
        elif feat_format == "ncm":
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv1dNormRelu(out_channels, 2, norm=None, act="sigmoid")
            self.mlp2 = Conv1dNormRelu(out_channels, 2, norm=None, act="sigmoid")
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        feat_2d = self.align1(feat_2d)  # [N, C_out, H, W]
        feat_3d = self.align2(feat_3d)  # [N, C_out, H, W]
        weight = self.mlp1(feat_2d) + self.mlp2(feat_3d)  # [N, 2, H, W]
        weight = softmax(weight, dim=1)  # [N, 2, H, W]
        return feat_2d * weight[:, 0:1] + feat_3d * weight[:, 1:2]

class MaskGlobalAvgPool2d(nn.Module):
    def __init__(
        self,
        output_size
    ):   
        super().__init__()
        
        self.output_size = output_size
        assert output_size == 1 or output_size == [1, 1]
    
    def forward(self, input, mask):
        """
        input: [B, C, H, W]
        mask: [B, H, W]
        output: [B, C]
        """
        mask = mask.float()
        mask = mask.unsqueeze(1)
        output = torch.sum(input * mask, dim = [2, 3]) / torch.sum(mask, dim = [2, 3])
        return output
        


class SKFusion(nn.Module):
    def __init__(
        self,
        in_channels_2d,
        in_channels_3d,
        out_channels,
        feat_format,
        norm=None,
        reduction=1,
    ):
        super().__init__()

        if feat_format == "nchw":
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif feat_format == "ncm":
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = MaskGlobalAvgPool2d(1)
        else:
            raise ValueError

        self.fc_mid = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(out_channels // reduction, out_channels * 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_2d, feat_3d, mask = None):
        bs = feat_2d.shape[0]

        feat_2d = self.align1(feat_2d)
        feat_3d = self.align2(feat_3d)
        
        if mask is None:
            weight = self.avg_pool(feat_2d + feat_3d).reshape([bs, -1])  # [bs, C]
        else:
            weight = self.avg_pool(feat_2d + feat_3d, mask.detach()).reshape([bs, -1])  # [bs, C]
        weight = self.fc_mid(weight)  # [bs, C / r]
        weight = self.fc_out(weight).reshape([bs, -1, 2])  # [bs, C, 2]
        weight = softmax(weight, dim=-1)
        w1, w2 = weight[..., 0], weight[..., 1]  # [bs, C]

        if len(feat_2d.shape) == 4:
            w1 = w1.reshape([bs, -1, 1, 1])
            w2 = w2.reshape([bs, -1, 1, 1])
        else:
            w1 = w1.reshape([bs, -1, 1])
            w2 = w2.reshape([bs, -1, 1])

        output_feat = feat_2d * w1 + feat_3d * w2
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            output_feat = output_feat * mask.detach()
        return output_feat

