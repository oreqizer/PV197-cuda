#include <stdio.h>

#define N 32
#define BLOCK_SIZE (N * N)

__global__ void solver(
    const int *results,
    float *avg_stud,
    float *avg_que,
    const int s,
    const int q
) {
    __shared__ float tmp_stud[BLOCK_SIZE];
    __shared__ float tmp_que[BLOCK_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y*q + x;

    // write to shared memory
    tmp_que[threadIdx.x] = results[idx];
    tmp_stud[threadIdx.y] = results[idx];
    __syncthreads();

    // reduce
    for (int i = 0; i < blockDim.x; i *= 2) {
        int index = 2 * i * threadIdx.x;
        if (index < blockDim.x) {
            tmp_que[index] += tmp_que[index + i];
        }
        __syncthreads();
    }
    for (int i = 0; i < blockDim.y; i *= 2) {
        int index = 2 * i * threadIdx.y;
        if (index < blockDim.y) {
            tmp_stud[index] += tmp_stud[index + i];
        }
        __syncthreads();
    }

    // write
    if (y == 0) avg_que[x] = tmp_que[0];
    if (x == 0) avg_stud[y] = tmp_stud[0];
}

void solveGPU(
    const int *results,  // students * questions
    float *avg_stud,
    float *avg_que,
    const int students,  // y: always divisible by 32
    const int questions  // x: always divisible by 32
) {
    int n = students * questions;
    solver<<<n/BLOCK_SIZE, BLOCK_SIZE>>>(
        results, avg_stud, avg_que, students, questions
    );

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
