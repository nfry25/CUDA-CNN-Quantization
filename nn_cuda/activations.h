// Pointwise Activation Functions & Their Derivatives
//
//     sigmoid : f(z)=1/(1+e^-z),  range (0,1).  Smooth, so the chain rule works
//               for backprop. Use for the pooling non-linearity or small nets
//     relu    : f(z)=max(0,z). faster. Use for hidden layers of FC and conv nets
//               default for the "desired output" of any hidden layer
//     softplus: f(z)=ln(1+e^z).  Smooth approximation of ReLU 
//
//     softmax : turns a length-C score vector into C probabilities that sum to 1.
//               Use on the final layer of a classifier

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "nn_common.h"


// Apply an elementwise activation in-place over n floats
void launch_sigmoid (float* d_x, int n);   // f(x)=1/(1+e^-x)
void launch_relu    (float* d_x, int n);   // f(x)=max(0,x)    
void launch_softplus(float* d_x, int n);   // f(x)=ln(1+e^x)  

// Softmax over a [rows x C] matrix                    
void launch_softmax (float* d_z, int rows, int C);

// __device__ inline math
#ifdef __CUDACC__
__device__ __forceinline__ float dev_sigmoid(float x) { return 1.0f/(1.0f+expf(-x)); }
__device__ __forceinline__ float dev_relu   (float x) { return x > 0.0f ? x : 0.0f; }

// Derivatives needed for the BACKWARD pass
//  sigmoid':  f'(x) = f(x)*(1 - f(x))
//  relu'   :  1 if x>0 else 0
__device__ __forceinline__ float dev_dsigmoid(float fx){ return fx*(1.0f-fx); }
__device__ __forceinline__ float dev_drelu   (float x ){ return x > 0.0f ? 1.0f : 0.0f; }
#endif

#endif /* ACTIVATIONS_H */
