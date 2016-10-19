#include <stdio.h>

#define N 32  // also the warpSize
#define BLOCK_SIZE (N * N)

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
    const int Y   // students
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int x = idx % X;
    int y = idx / X;
    int val = results[idx];

    atomicAdd(avg_stud + y, val);
    atomicAdd(avg_que + x, val);
    __syncthreads();

    // if (x == 0) avg_stud[y] /= X;
    // if (y == 0) avg_que[x] /= Y;
}

void solveGPU(
    const int *results,  // questions * students
    float *avg_stud,     // score per student: total / questions
    float *avg_que,      // score per question: total / students
    const int students,  // y: always divisible by 32
    const int questions  // x: always divisible by 32
) {
    int n = questions * students;

    dim3 grid(n/BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    solver<<<grid, block>>>(
        results, avg_stud, avg_que, questions, students
    );

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
