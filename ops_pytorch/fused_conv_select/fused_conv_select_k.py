import torch
import sys
import os
import numpy as np
import fused_conv_select_k_cuda as fused_conv_select_k_module


def fused_conv_select_k(xyz1, xyz2, idx_n2, idx_fetching, random_hw, H, W,
                        npoints, kernel_size_H, kernel_size_W, K, flag_copy,
                        distance, stride_h, stride_w, select_b_idx,
                        select_h_idx, select_w_idx, select_mask, small_h, small_w):
    '''
    Input:
        xyz1:(b, h, w, 3) float, projected xyz1 points 
        xyz2_feature:(b, h, w, c+3) float, projected xyz2 points with features
        idx_n2: (b, n, 2) int array, query idx of central points
        H, W : Input shape
        kernel_size_H, kernel_size_W: (size, size) int32 array, size
        k: the number of selected points (knn)
        distance: ( distance ) float  distance
        flag_copy  (bool)  whether copy or not for the output points
    
    Output:
        space_weight:(batch_size, npoint,  size*size , c)
    '''
    
    fused_conv_select_k_module.fused_conv_select_k(
        xyz1, xyz2, idx_n2, idx_fetching, random_hw, H, W, npoints,
        kernel_size_H, kernel_size_W, K, flag_copy, distance, stride_h,
        stride_w, select_b_idx, select_h_idx, select_w_idx, select_mask, small_h, small_w)
    return select_b_idx, select_h_idx, select_w_idx, select_mask

