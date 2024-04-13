import torch



def calc_supervised_loss_3d(flows, targets, cfgs, masks):
    assert len(flows) <= len(cfgs.level_weights)

    total_loss = 0
    for idx, (flow, level_weight) in enumerate(zip(flows, cfgs.level_weights)):
        level_target = targets[idx]

        mask = masks[idx]

        diff = flow - level_target # B, 3, H, W
        
        if cfgs.order == 'robust':
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4)[mask].mean()
            total_loss += level_weight * epe_l1
        elif cfgs.order == 'l2-norm':
            epe_l2 = torch.linalg.norm(diff, dim=1)[mask].mean()
            total_loss += level_weight * epe_l2
        else:
            raise NotImplementedError

    return total_loss


