import torch
import torch.nn as nn

def dist_reduce_sum(value):
    if torch.distributed.is_initialized():
        value_t = torch.Tensor([value]).cuda()
        torch.distributed.all_reduce(value_t)
        return value_t
    else:
        return value


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None
        self.metrics = {}

    def clear_metrics(self):
        self.metrics = {}

    @torch.no_grad()
    def update_metrics(self, name, var):
        if isinstance(var, torch.Tensor):
            var = var.reshape(-1)
            count = var.shape[0]
            var = var.float().sum().item()

        var = dist_reduce_sum(var)
        count = dist_reduce_sum(count)

        if count <= 0:
            return

        if name not in self.metrics.keys():
            self.metrics[name] = [0, 0]  # [var, count]

        self.metrics[name][0] += var
        self.metrics[name][1] += count

    def get_metrics(self):
        results = {}
        for name, (var, count) in self.metrics.items():
            results[name] = var / count
        return results

    def get_loss(self):
        if self.loss is None:
            raise ValueError('Loss is empty.')
        return self.loss

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        raise RuntimeError('Function `is_better` must be implemented.')


class FlowModel(BaseModel):
    def __init__(self):
        super(FlowModel, self).__init__()

    @torch.no_grad()
    def update_2d_metrics(self, pred, target):
        if target.shape[1] == 3:  # sparse evaluation
            mask = target[:, 2, :, :] > 0
            target = target[:, :2, :, :]
        else:  # dense evaluation
            mask = torch.ones_like(target)[:, 0, :, :] > 0 # B x H x W

        # compute endpoint error
        diff = pred - target
        epe2d_map = torch.linalg.norm(diff, dim=1) # B x H x W
        self.update_metrics('epe2d', epe2d_map[mask])

        # compute 1px accuracy
        acc2d_map = epe2d_map < 1.0
        self.update_metrics('acc2d_1px', acc2d_map[mask])
        
        # compute flow outliers
        mag = torch.linalg.norm(target, dim=1) + 1e-5
        out2d_map = torch.logical_and(epe2d_map > 3.0, epe2d_map / mag > 0.05)
        self.update_metrics('outlier2d', out2d_map[mask])
        
    @torch.no_grad()
    def update_3d_metrics(self, pred, target, mask, noc_mask = None):
 
        # compute endpoint error
        diff = pred - target # [B, 3, H, W]
        epe3d_map = torch.linalg.norm(diff, dim=1) # [B, H, W]
        acc5_3d_map = epe3d_map < 0.05 # compute 5cm accuracy
        acc10_3d_map = epe3d_map < 0.10 # compute 10cm accuracy
        
        if noc_mask is not None:
            self.update_metrics('epe3d(noc)', epe3d_map[noc_mask])
            self.update_metrics('acc3d_5cm(noc)', acc5_3d_map[noc_mask])
            self.update_metrics('acc3d_10cm(noc)', acc10_3d_map[noc_mask])
        else:
            self.update_metrics('epe3d', epe3d_map[mask])
            self.update_metrics('acc3d_5cm', acc5_3d_map[mask])
            self.update_metrics('acc3d_10cm', acc10_3d_map[mask])           