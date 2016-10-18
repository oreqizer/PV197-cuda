#include <stdio.h>

#define N 32  // also the warpSize
#define BLOCK_SIZE (N * N)

__shared__ float sh_tmp[BLOCK_SIZE];      // temporary storage
__shared__ float sh_res[BLOCK_SIZE + 1];  // max: 1024x1 + 1x1

__inline__ __device__
int warpSum(int val) {
    // warpSize is 32
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

__global__
void solver(
    const int *results,
    float *avg_stud,
    float *avg_que,
    const int X,  // students
    const int Y   // questions
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = y * X + x;
    int res_x = idx % X;
    int res_y = idx / X;
    int val = results[idx];

    atomicAdd(avg_que + res_y, val);
    atomicAdd(avg_stud + res_x, val);
    __syncthreads();

    if (res_x == 0) avg_que[res_y] *= 1.0/X;
    if (res_y == 0) avg_stud[res_x] *= 1.0/Y;
}

void solveGPU(
    const int *results,  // students * questions
    float *avg_stud,     // score per student: total / questions
    float *avg_que,      // score per question: total / students
    const int students,  // x: always divisible by 32
    const int questions  // y: always divisible by 32
) {
    int n = students * questions;
    solver<<<n/BLOCK_SIZE, BLOCK_SIZE>>>(
        results, avg_stud, avg_que, students, questions
    );

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
