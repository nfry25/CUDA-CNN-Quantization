// Fully-Connected (FC) Layer: Forward & Training Gradients
// Each output is a weighted sum of EVERY input plus a bias:
//       fc1[i] = sum_j W1[i,j] * x[j] + b[i]      ->  y = W.x + b
//
// x : [batch, in_features]    -- one row per sample; flatten images first
// W : [out_features, in_features]   (W[i,j] = weight from input j to output i)
// b : [out_features]
// y : [batch, out_features]
// 
// SGD update: theta_{i+1} = theta_i - eps*dTheta

#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "nn_common.h"

// Forward
// y = x.W^T + b.  x[batch,in], W[out,in], b[out] -> y[batch,out]
void fc_forward(const float* d_x, const float* d_W, const float* d_b, float* d_y, int batch, int in_features, int out_features);

// Backward w.r.t. weights: dW[i,j] = sum over batch of dY[*,i] * x[*,j]
void fc_backward_wgrad(const float* d_x, const float* d_dY, float* d_dW, int batch, int in_features, int out_features);

// Backward w.r.t. input: dX[*,j] = sum_i dY[*,i] * W[i,j]
void fc_backward_dgrad(const float* d_dY, const float* d_W, float* d_dX, int batch, int in_features, int out_features);

// Backward w.r.t. bias: db[i] = sum over batch of dY[*,i]                  
void fc_backward_bgrad(const float* d_dY, float* d_db, int batch, int out_features);

// SGD  update: theta -= lr * dTheta 
void launch_sgd_update(float* d_theta, const float* d_dTheta, float lr, int n);

#endif /* FC_LAYER_H */
