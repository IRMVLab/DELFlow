import numpy as np
import os
import os.path as osp


def get_batch_2d_flow(pc1, pc2, predicted_pc2, intrinsics):
    
    focallengths = intrinsics[0][0]
    cxs = intrinsics[0][2]
    cys = intrinsics[0][3]
    constx = 0
    consty = 0
    constz = 0

    px1, py1 = project_3d_to_2d(pc1, f=focallengths, cx=cxs, cy=cys,
                                constx=constx, consty=consty, constz=constz)
    px2, py2 = project_3d_to_2d(predicted_pc2, f=focallengths, cx=cxs, cy=cys,
                                constx=constx, consty=consty, constz=constz)
    px2_gt, py2_gt = project_3d_to_2d(pc2, f=focallengths, cx=cxs, cy=cys,
                                        constx=constx, consty=consty, constz=constz)
   

    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1
    flow_y_gt = py2_gt - py1

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)
    return flow_pred, flow_gt


def project_3d_to_2d(pc, f=1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * f + cx * pc[..., 2] + constx) / (pc[..., 2] + constz + 10e-10)
    y = (pc[..., 1] * f + cy * pc[..., 2] + consty) / (pc[..., 2] + constz + 10e-10)

    return x, y
