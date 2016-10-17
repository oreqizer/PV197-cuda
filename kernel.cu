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
    const int X,  // questions
    const int Y  // students
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    float sum = 0;

    if (y < Y) {
        for (int i = 0; i < X; i++) {
            sum += results[i * Y + y];
        }
        avg_stud[y] = sum * (1.0/X);
    }

    sum = 0;
    if (x < X) {
        for (int i = 0; i < Y; i++) {
            sum += results[i * X + x];
        }
        avg_que[x] = sum * (1.0/Y);
    }
}

void solveGPU(
    const int *results,  // students * questions
    float *avg_stud,
    float *avg_que,
    const int students,  // y: always divisible by 32
    const int questions  // x: always divisible by 32
) {
    // int n = students * questions;  TODO optimize
    solver<<<questions/BLOCK_SIZE, BLOCK_SIZE>>>(
        results, avg_stud, avg_que, questions, students
    );

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
