// Convolutional Layer: Forward & Training Gradients
// Forward:
// Y[b,m,h,w] = sum_c sum_p sum_q  X[b,c,h+p,w+q] * W[m,c,p,q]
// > with output size H_out = H - K + 1,  W_out = W - K + 1.
//
// Training Gradients:
// dE/dX = W . dE/dY     -> conv_backward_dgrad  (gradient to previous layer)
// dE/dW = dE/dY . X     -> conv_backward_wgrad  (gradient for weight update)
//  > update W with launch_sgd_update from fc_layer.h.
//
//   X : [B, C, H, W]            B images, C channels, HxW pixels per channel.
//   W : [M, C, K, K]            M output feature maps, KxK filter per channel.
//   Y : [B, M, H_out, W_out]    M feature maps out per image.

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "nn_common.h"

// Forward
// one thread per output pixel; grid = (M, H_grid*W_grid, B)
void conv_forward_basic(const float* d_X, const float* d_W, float* d_Y, int B, int M, int C, int H, int W, int K);

// Tiled kernel
// each block computes one TILE_WIDTH x TILE_WIDTH
void conv_forward_tiled(const float* d_X, const float* d_W, float* d_Y, int B, int M, int C, int H, int W, int K);

// Backward
// dE/dX from dE/dY and W
void conv_backward_dgrad(const float* d_dE_dY, const float* d_W, float* d_dE_dX, int B, int M, int C, int H, int W, int K);

// dE/dW from dE/dY and X 
// per-image partial gradients are summed across the batch on host or via atomics
void conv_backward_wgrad(const float* d_dE_dY, const float* d_X, float* d_dE_dW, int B, int M, int C, int H, int W, int K);

#endif /* CONV_LAYER_H */
