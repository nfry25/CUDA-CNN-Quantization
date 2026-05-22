// A convolution layer can be unrolled (im2col) so every KxK x C receptive field becomes a column, then:
// Y(M x [H_out*W_out])  =  W'(M x [C*K*K])  *  X_unroll([C*K*K] x [H_out*W_out])
//
// Unroll Mapping for output element (h,w), filter element (p,q), channel c:
//       w_unroll = h * W_out + w            (column = which output pixel)
//       w_base   = c * (K*K)
//       h_unroll = w_base + p * K + q       (row = which (c,p,q) tap)
//       X_unroll[h_unroll, w_unroll] = X[c, h+p, w+q]
//
// X        : [C, H, W]               
// X_unroll : [C*K*K, H_out*W_out]
// W_flat   : [M, C*K*K]              
// Y        : [M, H_out*W_out]        

#ifndef CONV_IM2COL_H
#define CONV_IM2COL_H

#include "nn_common.h"

// im2col matrix for single image                     
void im2col(const float* d_X, float* d_X_unroll, int C, int H, int W, int K);

// Tiled matrix multiply 
// C = A * B;  A[Ar x Ac], B[Ac x Bc], C[Ar x Bc]
void matmul_tiled(const float* d_A, const float* d_B, float* d_C, int Ar, int Ac, int Bc);

// im2col-based FC for single image
// d_X[C,H,W], d_W_flat[M, C*K*K] -> d_Y[M, H_out*W_out].
void conv_forward_im2col(const float* d_X, const float* d_W_flat, float* d_X_unroll, float* d_Y, int M, int C, int H, int W, int K);

#endif /* CONV_IM2COL_H */
