#!/bin/sh
set -x

# make build_ci_image
make build_cuda_runtime_image 
make build_inference_runtime_image 
make build_cpu_runtime_image 

