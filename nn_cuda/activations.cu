// Implementations of the Activation Launchers

#include "activations.h"

// Elementwise activations
__global__ void sigmoid_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = dev_sigmoid(x[i]);  
}
__global__ void relu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = dev_relu(x[i]);    
}
__global__ void softplus_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = log1pf(expf(x[i]));
}

static void launch_pointwise(void(*k)(float*,int), float* d_x, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    k<<<blocks, threads>>>(d_x, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_sigmoid (float* d_x, int n){ launch_pointwise(sigmoid_kernel , d_x, n); }
void launch_relu    (float* d_x, int n){ launch_pointwise(relu_kernel    , d_x, n); }
void launch_softplus(float* d_x, int n){ launch_pointwise(softplus_kernel, d_x, n); }

// Softmax 
__global__ void softmax_kernel(float* z, int rows, int C) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* r = z + (size_t)row * C;

    // max for stability 
    float m = -1e30f;
    for (int j = 0; j < C; ++j) m = fmaxf(m, r[j]);
    // exponentiate and accumulate denominator
    float sum = 0.0f;
    for (int j = 0; j < C; ++j) { r[j] = expf(r[j] - m); sum += r[j]; }
    // normalise so the C outputs sum to 1
    for (int j = 0; j < C; ++j) r[j] /= sum;
}

void launch_softmax(float* d_z, int rows, int C) {
    softmax_kernel<<<rows, 1>>>(d_z, rows, C);
    CUDA_CHECK(cudaGetLastError());
}
