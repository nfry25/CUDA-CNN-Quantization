// im2col unrolling + tiled MatMul

#include "conv_im2col.h"

__global__ void im2col_kernel(const float* X, float* X_unroll, int C, int H, int Wd, int K, int H_out, int W_out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H_out * W_out;
    if (t >= total) return;

    int W_unroll_cols = H_out * W_out;      
    int c = t / W_unroll_cols;              
    int rem = t % W_unroll_cols;
    int h = rem / W_out;                   
    int w = rem % W_out;                   

    int w_unroll = h * W_out + w;           
    int w_base   = c * (K * K);             
    for (int p = 0; p < K; ++p)
        for (int q = 0; q < K; ++q) {
            int h_unroll = w_base + p * K + q;            

            X_unroll[h_unroll * W_unroll_cols + w_unroll] = X[((size_t)c * H + (h + p)) * Wd + (w + q)];
        }
}
void im2col(const float* d_X, float* d_X_unroll, int C, int H, int W, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int total = C * H_out * W_out;
    int threads = 256, blocks = (total + threads - 1) / threads;
    im2col_kernel<<<blocks, threads>>>(d_X, d_X_unroll, C, H, W, K, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

// Tiled MatMul
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int Ar, int Ac, int Bc) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;

    int nTiles = (Ac + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < nTiles; ++t) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < Ar && aCol < Ac) ? A[row * Ac + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < Ac && col < Bc) ? B[bRow * Bc + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < Ar && col < Bc) C[row * Bc + col] = acc;
}
void matmul_tiled(const float* d_A, const float* d_B, float* d_C, int Ar, int Ac, int Bc) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((Bc + TILE_WIDTH - 1) / TILE_WIDTH, (Ar + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, Ar, Ac, Bc);
    CUDA_CHECK(cudaGetLastError());
}

// im2col forward for one image
void conv_forward_im2col(const float* d_X, const float* d_W_flat, float* d_X_unroll, float* d_Y, int M, int C, int H, int W, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int Ac = C * K * K;         
    int Bc = H_out * W_out;    
    im2col(d_X, d_X_unroll, C, H, W, K);
    matmul_tiled(d_W_flat, d_X_unroll, d_Y, /*Ar=*/M, /*Ac=*/Ac, /*Bc=*/Bc);
}
