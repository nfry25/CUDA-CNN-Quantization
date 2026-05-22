// FC Layer Kernels & SGD

#include "fc_layer.h"

__global__ void fc_forward_kernel(const float* x, const float* W, const float* b, float* y, int batch, int in_f, int out_f) {
    int s = blockIdx.y * blockDim.y + threadIdx.y; 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (s < batch && i < out_f) {
        float acc = b ? b[i] : 0.0f;               
        for (int j = 0; j < in_f; ++j)
            acc += MAT(x, s, j, in_f) * MAT(W, i, j, in_f); 
        MAT(y, s, i, out_f) = acc;
    }
}
void fc_forward(const float* d_x, const float* d_W, const float* d_b, float* d_y, int batch, int in_f, int out_f) {
    dim3 block(16, 16);
    dim3 grid((out_f + 15) / 16, (batch + 15) / 16);
    fc_forward_kernel<<<grid, block>>>(d_x, d_W, d_b, d_y, batch, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());
}


__global__ void fc_wgrad_kernel(const float* x, const float* dY, float* dW, int batch, int in_f, int out_f) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;  
    if (i < out_f && j < in_f) {
        float acc = 0.0f;
        for (int s = 0; s < batch; ++s)
            acc += MAT(dY, s, i, out_f) * MAT(x, s, j, in_f);
        MAT(dW, i, j, in_f) = acc;
    }
}
void fc_backward_wgrad(const float* d_x, const float* d_dY, float* d_dW, int batch, int in_f, int out_f) {
    dim3 block(16, 16);
    dim3 grid((in_f + 15) / 16, (out_f + 15) / 16);
    fc_wgrad_kernel<<<grid, block>>>(d_x, d_dY, d_dW, batch, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fc_dgrad_kernel(const float* dY, const float* W, float* dX, int batch, int in_f, int out_f) {
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < batch && j < in_f) {
        float acc = 0.0f;
        for (int i = 0; i < out_f; ++i)
            acc += MAT(dY, s, i, out_f) * MAT(W, i, j, in_f);
        MAT(dX, s, j, in_f) = acc;
    }
}
void fc_backward_dgrad(const float* d_dY, const float* d_W, float* d_dX, int batch, int in_f, int out_f) {
    dim3 block(16, 16);
    dim3 grid((in_f + 15) / 16, (batch + 15) / 16);
    fc_dgrad_kernel<<<grid, block>>>(d_dY, d_W, d_dX, batch, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fc_bgrad_kernel(const float* dY, float* db, int batch, int out_f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_f) {
        float acc = 0.0f;
        for (int s = 0; s < batch; ++s) acc += MAT(dY, s, i, out_f);
        db[i] = acc;
    }
}
void fc_backward_bgrad(const float* d_dY, float* d_db, int batch, int out_f) {
    int threads = 256, blocks = (out_f + threads - 1) / threads;
    fc_bgrad_kernel<<<blocks, threads>>>(d_dY, d_db, batch, out_f);
    CUDA_CHECK(cudaGetLastError());
}

// SGD update
__global__ void sgd_kernel(float* theta, const float* dTheta, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) theta[i] -= lr * dTheta[i];
}
void launch_sgd_update(float* d_theta, const float* d_dTheta, float lr, int n) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    sgd_kernel<<<blocks, threads>>>(d_theta, d_dTheta, lr, n);
    CUDA_CHECK(cudaGetLastError());
}
