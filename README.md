# CUDA-CNN-Quantization

A CIFAR-10 image classifier built on custom CUDA C convolution kernels, then compressed through iterative magnitude pruning (IMP) and INT8 quantization to compress model size while maintaining accuracy. The compressed model can then be extracted for deploying on edge-hardware.

## Overview

This project trains a compact convolutional neural network (CNN) for CIFAR-10 image classification and applies two compression techniques to compress the trained model for deployment on resource-constrained hardware. The convolution is executed by a custom tiled CUDA C kernel, compiled in-notebook with `nvcc` and called from Python through `ctypes`, and is verified against a PyTorch reference for correctness. Because neural-network inference is typically bound by memory bandwidth rather than arithmetic throughput, reducing the number of active weights directly lowers the data movement that governs latency and energy on GPU hardware.

## Pipeline

The notebook proceeds in five stages. It first loads and normalizes the CIFAR-10 dataset and trains a baseline FP32 model using mini-batch stochastic gradient descent. It then records baseline metrics, including test accuracy, active parameter count, model size, and a per-class confusion matrix. Next, it analyzes weight redundancy by auditing the per-layer parameter distribution and the weight-magnitude histogram. The compression stage applies an accumalted pruning schedule with fine-tuning, followed by post-training INT8 quantization of all convolution and linear weights. Finally, the compressed model is exported to Google Drive in sparse INT8 form, and a final metric evaluation compares the baseline and compressed models.

A separate stage compiles the CUDA C kernel toolkit into a shared library and validates the tiled convolution kernel against `torch.nn.functional.conv2d` within floating-point tolerance.

## Model Architecture

The network is a LeNet-style classifier with two convolution-and-pooling stages followed by two fully-connected layers. The first convolution maps 3 input channels to 32 feature maps and the second maps 32 to 64, each using 3x3 filters with ReLU activations and 2x2 max pooling. The flattened features pass through a 256-unit fully-connected layer and a final 10-unit output layer. The softmax is folded into the cross-entropy loss during training.

## Compression Techniques

**Iterative Magnitude Pruning.** After training, many weights cluster near zero and contribute little to the output. The pipeline removes the smallest-magnitude weights across multiple rounds, retraining briefly between rounds so the surviving weights re-adapt. Pruning a fraction *p* of the weights leaves a sparsity *s = 1 - p* and a theoretical compression of 1 / *s*.

**INT8 Quantization.** Each weight tensor is mapped from 32-bit float to 8-bit integer using a per-tensor scale factor *S* and zero-point *Z*, where *q* = clip(round(*r* / *S*) + *Z*, 0, 2^b - 1) and *S* = (*r*max - *r*min) / (2^b - 1). Dequantization recovers *r* = *S*(*q* - *Z*). Moving from FP32 to INT8 reduces the bytes per stored weight by a factor of four.

## Requirements

This project is designed to run in Google Colab on an NVIDIA A100 GPU with the High-RAM runtime profile. A Google Drive account is used for dataset and source-code storage.

## Setup

Organize Google Drive so the notebook can locate both the dataset and the CUDA source:

```
MyDrive/
├── CIFAR10/                 # CIFAR-10 binary files
│   ├── data_batch_1.bin ... data_batch_5.bin
│   ├── test_batch.bin
│   └── batches.meta.txt
└── nn_cuda/                 # CUDA C Code (.cu / .h)
    ├── conv_layer, fc_layer, activations
    ├── pooling_layer, conv_im2col
    └── nn_common.h
```

Open `CIFAR10_CNN.ipynb` in Colab, confirm the A100 High-RAM runtime, update the `DRIVE_DATA_DIR` and `DRIVE_CODE_DIR` paths if needed, and run the cells in order.

## Results

The notebook reports a baseline-versus-compressed scorecard covering test accuracy on the 10,000-image test set, model size, the resulting compression ratio, and final weight sparsity, along with a per-class precision, recall, and F1 report.

## Tech Stack

The kernels are written in CUDA C and compiled with `nvcc`, using shared-memory tiling and a `ctypes` bridge to Python. Baseline training and compression use PyTorch and `torch.nn.utils.prune`, with a custom INT8 quantizer. NumPy, scikit-learn, and Matplotlib handle metrics and visualization. The project runs on Google Colab with Google Drive for storage.

## Dataset

CIFAR-10 contains 60,000 color images at 32x32 resolution across ten classes — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck — split into 50,000 training and 10,000 test images. Each binary record stores one label byte followed by 3,072 pixel bytes in channel-major order, loaded directly into the tensor layout expected by the CUDA convolution kernel.
