#include <algorithm>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstdlib>// Header file needed to use rand
#include <math.h>
#include <cuda_runtime.h>

struct special_idx{
    int idx_h;
    int idx_w;
};

__global__ void no_sort_knn_gpu(int batch_size, int H, int W, int npoints, int kernel_size_H, int kernel_size_W, int K, int flag_copy, float distance, int stride_h, int stride_w, const float *xyz1, const float *xyz2, const int *idx_n2, const int *random_hw, long *selected_b_idx, long * selected_h_idx, long * selected_w_idx, float *selected_mask, int small_h, int small_w)
{
    // in this function, only select 3 closest points
    int batch_index = blockIdx.x;
    int index_thread = threadIdx.x;
	int stride_thread = blockDim.x;

	int kernel_total = kernel_size_H * kernel_size_W;
	int selected_W_idx = 0, selected_H_idx =0;

	float dist_square = distance * distance;

	int kernel_half_H = kernel_size_H / 2;
	int kernel_half_W = kernel_size_W / 2;

	xyz1 += batch_index * H * W * 3;
    xyz2 += batch_index * small_h * small_w * 3;
	idx_n2 += batch_index * npoints * 2;	
	selected_b_idx += batch_index * npoints * K * 1; 
    selected_h_idx += batch_index * npoints * K * 1; 
    selected_w_idx += batch_index * npoints * K * 1; 


	// valid_idx += batch_index * npoints * kernel_total * 1 ; //(b, npoints, h*w, 1)
	// valid_in_dis_idx += batch_index * npoints * kernel_total * 1 ; //(b, npoints, h*w, 1)
	
	selected_mask += batch_index * npoints * K * 1 ; //(b, npoints, h*w, 1)

    for (int dense_i = index_thread; dense_i < npoints; dense_i += stride_thread) {
        // for each point in dense pl, search closest three points in sparse pl
        selected_H_idx = idx_n2[dense_i * 2];
        selected_W_idx = idx_n2[dense_i * 2 + 1];
        int central_idx = selected_H_idx * W * 3 + selected_W_idx * 3;
        float dense_x = xyz1[central_idx];
        float dense_y = xyz1[central_idx + 1];
        float dense_z = xyz1[central_idx + 2];
        // int num_valid_idx = 0;
        int num_select = 0;

        float dist_from_origin = dense_x * dense_x + dense_y * dense_y + dense_z * dense_z;
        if (dist_from_origin <= 1e-10) {
            continue;
        }
        float best1 = 1e30, best2 = 1e30, best3 = 1e30;
        special_idx besti[3];
        for (int i = 0; i < 3; ++i) {
            besti[i].idx_h = besti[i].idx_w = 0;
        }

        // once the central points in xyz1 are valid, begin to dispose xyz2 points
        for (int current_HW_idx = 0; current_HW_idx < kernel_total; ++current_HW_idx) {
            int kernel_hw_idx = random_hw[current_HW_idx];
            int kernel_select_h_idx = selected_H_idx / stride_h + kernel_hw_idx / kernel_size_W - kernel_half_H;
            int kernel_select_w_idx = selected_W_idx / stride_w + kernel_hw_idx % kernel_size_W - kernel_half_W;

            if ((kernel_select_h_idx < 0) || (kernel_select_w_idx < 0) || (kernel_select_h_idx >= small_h) || (kernel_select_w_idx >= small_w)) {
                continue;
            }
            int select_idx = kernel_select_h_idx * small_w * 3 + kernel_select_w_idx * 3;
            float sparse_x = xyz2[select_idx];
            float sparse_y = xyz2[select_idx + 1];
            float sparse_z = xyz2[select_idx + 2];
            
            float queried_dist_from_center = sparse_x * sparse_x + sparse_y * sparse_y + sparse_z * sparse_z;
            if (queried_dist_from_center <= 1e-10) {
                continue;
                // queried points are invalid points
            }
            float dist_from_query = (dense_x - sparse_x) * (dense_x - sparse_x) + (dense_y - sparse_y) * (dense_y - sparse_y) + (dense_z - sparse_z) * (dense_z - sparse_z);
            
            // if (num_select == 0 && flag_copy) {
            //     // if at least one point is found, copy its information to all three points
            //     best1 = best2 = best3 = dist_from_query;
            //     special_idx temp;
            //     temp.idx_h = kernel_select_h_idx;
            //     temp.idx_w = kernel_select_w_idx;
            //     besti[0] = besti[1] = besti[2] = temp;
            // }
            

            if (dist_from_query < 1e-10) {
                // we treat it as the original point and the first queried point
                best3 = best2;
                besti[2] = besti[1];
                best2 = best1;
                besti[1] = besti[0];
                best1 = dist_from_query;
                besti[0].idx_h = kernel_select_h_idx;
                besti[0].idx_w = kernel_select_w_idx;
                continue;
            }
            if (dist_from_query > dist_square) {
                continue;
            }

            ++num_select;
            // given a central point, select the closest 3 points in a kernel
            
            if (dist_from_query < best1) {
                best3 = best2;
                besti[2] = besti[1];
                best2 = best1;
                besti[1] = besti[0];
                best1 = dist_from_query;
                besti[0].idx_h = kernel_select_h_idx;
                besti[0].idx_w = kernel_select_w_idx;
            } else if (dist_from_query < best2) {
                best3 = best2;
                besti[2] = besti[1];
                best2 = dist_from_query;
                besti[1].idx_h = kernel_select_h_idx;
                besti[1].idx_w = kernel_select_w_idx;
            } else if (dist_from_query < best3) {
                best3 = dist_from_query;
                besti[2].idx_h = kernel_select_h_idx;
                besti[2].idx_w = kernel_select_w_idx;
            }
        }
        // bool no_point_flag = (best1 >= 1e30 && best2 >= 1e30 && best3 >= 1e30);

        int max_points = num_select < K ? num_select : K;
        int temp;
        for (int k = 0; k < max_points; ++k) {
            temp = dense_i * K + k;
            selected_b_idx[temp] = batch_index;
            selected_h_idx[temp] = besti[k].idx_h;
            selected_w_idx[temp] = besti[k].idx_w;
            selected_mask[temp] = 1.0;
        }
        if (flag_copy) {
            // if no points are selected, copy the first item
            for (int k = max_points; k < K; ++k) {
                int temp = dense_i * K + k;
                selected_b_idx[temp] = batch_index;
                selected_h_idx[temp] = besti[0].idx_h;
                selected_w_idx[temp] = besti[0].idx_w;
                selected_mask[temp] = 1.0;
            }
        } 



        // for (int k = 0; k < K; ++k) {
        //     int temp = dense_i * K + k;
        //     selected_b_idx[temp] = batch_index;
        //     selected_h_idx[temp] = besti[k].idx_h;
        //     selected_w_idx[temp] = besti[k].idx_w;
        //     selected_mask[dense_i * K + k] = 1.0;
        // }
    }
}



void NoSortKnnLauncher(int batch_size, int H, int W, int npoints, int kernel_size_H, int kernel_size_W, int K, int flag_copy, float distance, int stride_h, int stride_w, const float *xyz1, const float *xyz2, const int *idx_n2, const int *random_hw, long *selected_b_idx, long *selected_h_idx, long *selected_w_idx, float *selected_mask, cudaStream_t stream)
{
    int small_h = ceil(H / stride_h);
    int small_w = ceil(W / stride_w);
    cudaError_t err;
    no_sort_knn_gpu<<<batch_size, 256, 0, stream>>>(batch_size, H, W, npoints, kernel_size_H, kernel_size_W, K, flag_copy, distance, stride_h, stride_w, xyz1, xyz2, idx_n2, random_hw, selected_b_idx, selected_h_idx, selected_w_idx, selected_mask, small_h, small_w);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }	
}
