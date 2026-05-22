//  W[M, C, K, K]         convolution weights.  M = #output feature maps, C = #input feature maps, K = filter height=width
//  X[C, H, W]            input feature maps;  H = W = (output dim) + K - 1
//  Y[M, H_out, W_out]    output feature maps
// All tensors are stored as flat float* arrays in row-major order

#ifndef NN_COMMON_H
#define NN_COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


// X4: input X[b,c,i,j]  with channel count C, height H, width W (per channel)
#define X4(X, b, c, i, j, C, H, W)  ((X)[ ((((b)*(C) + (c))*(H) + (i))*(W)) + (j) ])

// W4: weight W[m,c,p,q]  with channels C and filter side K                 
#define W4(W, m, c, p, q, C, K)     ((W)[ ((((m)*(C) + (c))*(K) + (p))*(K)) + (q) ])

// Y4: output Y[b,m,i,j]  with feature count M, output dims H_out, W_out  
#define Y4(Y, b, m, i, j, M, HO, WO)((Y)[ ((((b)*(M) + (m))*(HO) + (i))*(WO)) + (j) ])

// 2D matrix index, row-major 
#define MAT(A, row, col, ncols)     ((A)[ (row)*(ncols) + (col) ])

// Error-checking wrapper 
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Default tile width
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#endif /* NN_COMMON_H */
