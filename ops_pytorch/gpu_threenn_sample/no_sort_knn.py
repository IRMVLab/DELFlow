import torch
import sys
import os
import numpy as np

import no_sort_knn_cuda as no_sort_knn_module

def no_sort_knn(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, stride_h, stride_w, select_b_idx, select_h_idx, select_w_idx, select_mask):

    no_sort_knn_module.no_sort_knn(xyz1, xyz2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, stride_h, stride_w, select_b_idx, select_h_idx, select_w_idx, select_mask)
    return select_b_idx, select_h_idx, select_w_idx, select_mask

