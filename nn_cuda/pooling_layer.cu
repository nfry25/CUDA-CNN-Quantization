// Average Pooling

#include "pooling_layer.h"

// Pooling Fwd
__global__ void pooling_fwd_kernel(const float* Y, const float* bias, float* S, int B, int M, int H_out, int W_out, int N) {
    int HS = H_out / N;   
    int WS = W_out / N;  

    int b = blockIdx.z;
    int m = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (b >= B || m >= M || idx >= HS * WS) return;
    int x = idx / WS;   
    int y = idx % WS;  

    float acc = 0.0f;
    for (int p = 0; p < N; ++p)                       
        for (int q = 0; q < N; ++q)
            acc += Y4(Y, b, m, N * x + p, N * y + q, M, H_out, W_out);
    acc /= (float)(N * N);                         

    float v = acc + (bias ? bias[m] : 0.0f);        
    S[((((size_t)b * M + m) * HS + x) * WS) + y] = 1.0f / (1.0f + expf(-v)); 
}
void pooling_forward(const float* d_Y, const float* d_bias, float* d_S, int B, int M, int H_out, int W_out, int N) {
    int HS = H_out / N, WS = W_out / N;
    int threads = 256;
    int blocks_x = (HS * WS + threads - 1) / threads;
    dim3 grid(blocks_x, M, B);
    pooling_fwd_kernel<<<grid, threads>>>(d_Y, d_bias, d_S, B, M, H_out, W_out, N);
    CUDA_CHECK(cudaGetLastError());
}
