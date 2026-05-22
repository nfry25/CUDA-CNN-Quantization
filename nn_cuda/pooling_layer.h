// Average Pooling Layer
// Shrinks each MxH_out x W_out feature map by averaging over non-overlapping blocks, then adds a bias and applies a non-linearity
//
// Y    : [B, M, H_out, W_out]   (conv layer output)
// bias : [M]
// S    : [B, M, HS, WS]

#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "nn_common.h"

// Average-pool Y over NxN blocks, add bias[m], apply sigmoid        
void pooling_forward(const float* d_Y, const float* d_bias, float* d_S, int B, int M, int H_out, int W_out, int N);

#endif /* POOLING_LAYER_H */
