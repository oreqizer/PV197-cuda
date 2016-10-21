#!/bin/bash

HOST=433689@airacuda.fi.muni.cz
scp framework.cu kernel_CPU.C kernel.cu $HOST:~/cuda
ssh $HOST "cd ~/cuda &&\
/usr/local/cuda/bin/nvcc -o run -arch=compute_30 -code=sm_30 framework.cu;"
ssh -X $HOST "export DISPLAY='127.0.0.1:10.0';\
/usr/local/cuda/bin/nvvp ~/cuda/run"
