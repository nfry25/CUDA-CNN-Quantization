// Convolutional Layer Kernels

#include "conv_layer.h"

// Fwd Kernel
__global__ void conv_fwd_basic_kernel(const float* X, const float* W, float* Y, int B, int M, int C, int H, int Wd, int K, int W_grid) {
    int H_out = H - K + 1;
    int W_out = Wd - K + 1;

    int b = blockIdx.z;                                  
    int m = blockIdx.x;                                 
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y; 
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;  

    if (h < H_out && w < W_out) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c)          
            for (int p = 0; p < K; ++p)     
                for (int q = 0; q < K; ++q)
                    acc += X4(X, b, c, h + p, w + q, C, H, Wd) * W4(W, m, c, p, q, C, K);
        Y4(Y, b, m, h, w, M, H_out, W_out) = acc;
    }
}
void conv_forward_basic(const float* d_X, const float* d_W, float* d_Y, int B, int M, int C, int H, int W, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;  
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;  
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid(M, H_grid * W_grid, B);                 
    conv_fwd_basic_kernel<<<grid, block>>>(d_X, d_W, d_Y, B, M, C, H, W, K, W_grid);
    CUDA_CHECK(cudaGetLastError());
}

// Tiled Fwd Kernel
__global__ void conv_fwd_tiled_kernel(const float* X, const float* W, float* Y, int C, int H, int Wd, int K, int W_grid, int M, int H_out, int W_out, int B) {
    extern __shared__ float shmem[];
    int X_tile_width = TILE_WIDTH + K - 1;
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];

    int b  = blockIdx.z;
    int m  = blockIdx.x;
    int h0 = threadIdx.y, w0 = threadIdx.x;            
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;   
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;    
    int h = h_base + h0;
    int w = w_base + w0;

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        if (h0 < K && w0 < K)
            W_shared[h0 * K + w0] = W4(W, m, c, h0, w0, C, K);
        __syncthreads();
        for (int i = h0; i < X_tile_width; i += TILE_WIDTH)
            for (int j = w0; j < X_tile_width; j += TILE_WIDTH) {
                int gi = h_base + i, gj = w_base + j;
                float v = (gi < H && gj < Wd) ? X4(X, b, c, gi, gj, C, H, Wd) : 0.0f;
                X_shared[i * X_tile_width + j] = v;
            }
        __syncthreads();
        if (h < H_out && w < W_out)
            for (int p = 0; p < K; ++p)
                for (int q = 0; q < K; ++q)
                    acc += X_shared[(h0 + p) * X_tile_width + (w0 + q)] * W_shared[p * K + q];
        __syncthreads();
    }
    if (h < H_out && w < W_out)
        Y4(Y, b, m, h, w, M, H_out, W_out) = acc;
}
void conv_forward_tiled(const float* d_X, const float* d_W, float* d_Y, int B, int M, int C, int H, int W, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int X_tile_width = TILE_WIDTH + K - 1;
    size_t shmem = (X_tile_width * X_tile_width + K * K) * sizeof(float);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid(M, H_grid * W_grid, B);
    conv_fwd_tiled_kernel<<<grid, block, shmem>>>(d_X, d_W, d_Y, C, H, W, K, W_grid, M, H_out, W_out, B);
    CUDA_CHECK(cudaGetLastError());
}

// Backward
__global__ void conv_dgrad_kernel(const float* dE_dY, const float* W, float* dE_dX, int B, int M, int C, int H, int Wd, int K, int W_grid, int H_out, int W_out) {
    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    if (h < H_out && w < W_out) {
        float g = Y4(dE_dY, b, m, h, w, M, H_out, W_out);
        for (int c = 0; c < C; ++c)
            for (int p = 0; p < K; ++p)
                for (int q = 0; q < K; ++q)
                    atomicAdd(&X4(dE_dX, b, c, h + p, w + q, C, H, Wd), g * W4(W, m, c, p, q, C, K));
    }
}
void conv_backward_dgrad(const float* d_dE_dY, const float* d_W, float* d_dE_dX, int B, int M, int C, int H, int W, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid(M, H_grid * W_grid, B);
    conv_dgrad_kernel<<<grid, block>>>(d_dE_dY, d_W, d_dE_dX, B, M, C, H, W, K,
                                       W_grid, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}


__global__ void conv_wgrad_kernel(const float* dE_dY, const float* X, float* dE_dW, int B, int M, int C, int H, int Wd, int K, int W_grid, int H_out, int W_out) {
    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    if (h < H_out && w < W_out) {
        float g = Y4(dE_dY, b, m, h, w, M, H_out, W_out);
        for (int c = 0; c < C; ++c)
            for (int p = 0; p < K; ++p)
                for (int q = 0; q < K; ++q)
                    atomicAdd(&W4(dE_dW, m, c, p, q, C, K), X4(X, b, c, h + p, w + q, C, H, Wd) * g);
    }
}
void conv_backward_wgrad(const float* d_dE_dY, const float* d_X, float* d_dE_dW, int B, int M, int C, int H, int W, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid(M, H_grid * W_grid, B);
    conv_wgrad_kernel<<<grid, block>>>(d_dE_dY, d_X, d_dE_dW, B, M, C, H, W, K, W_grid, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}
