#include <stdio.h>

#define N 32  // also the warpSize
#define BLOCK_SIZE (N * N)

__inline__ __device__
int warpSum(int val) {
    // warpSize always 32
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

__global__
void reduceRows(const int *in, float *out, int X, int Y) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int sum = in[x];
    sum = warpSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(out + x / X, sum);
    }
}

__global__  // BOTTLENECK
void reduceCols(const int *in, float *out, int X, int Y) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int col = x % Y;
    int row = x / Y;
    int sum = in[col*X + row];
    sum = warpSum(sum);
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(out + row, sum);
    }
}

__global__
void divide(float *arr, float count) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    arr[i] /= count;
}

void solveGPU(
    const int *results,  // questions * students
    float *avg_stud,     // score per student: total / questions -> len Y
    float *avg_que,      // score per question: total / students -> len X
    const int Y,         // students: always divisible by 32
    const int X          // questions: always divisible by 32
) {
    int n = X * Y;

    // reset arrays
    cudaMemset(avg_stud, 0, Y*sizeof(avg_stud[0]));
    cudaMemset(avg_que, 0, X*sizeof(avg_que[0]));

    dim3 threads(N);
    dim3 blocks(n/N);

    // load all results
    reduceRows<<<blocks, threads>>>(results, avg_stud, X, Y);
    reduceCols<<<blocks, threads>>>(results, avg_que, X, Y);

    // divide results
    divide<<<Y/N, N>>>(avg_stud, X);
    divide<<<X/N, N>>>(avg_que, Y);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
