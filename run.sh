#!/bin/bash

HOST=433689@airacuda.fi.muni.cz
scp framework.cu kernel_CPU.C kernel.cu $HOST:~/cuda
ssh $HOST 'cd ~/cuda && /usr/local/cuda/bin/nvcc -o run -arch=compute_30 -code=sm_30 framework.cu; ~/cuda/run; rm ~/cuda/run'
