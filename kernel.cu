#include <stdio.h>

#define N 32  // also the warpSize
#define BLOCK_SIZE (N * N)

__inline__ __device__
int warpSum(int val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__inline__ __device__
int blockSum(int val) {
    static __shared__ int shared[N]; // shared mem for 32 partial sums
    int x = threadIdx.x % warpSize;
    int y = threadIdx.x / warpSize;

    val = warpSum(val);              // reduce warp
    if (x == 0) shared[y] = val;     // write result to shared memory
    __syncthreads();

    val = shared[x];
    if (y == 0) val = warpSum(val);  // reduce within 1st warp

    return val;
}

__global__
void reduceRows(const int *in, float *out, int X, int Y) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = x; i < X*Y; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockSum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(out + x / X, sum);
    }
}

__global__
void reduceCols(const int *in, float *out, int X, int Y) {
    int y = blockIdx.x*blockDim.x + threadIdx.x;
    int row = y % X;
    int col = y / X;
    int sum = in[col*X + row];
    // for (int i = y; i < X*Y; i += blockDim.x * gridDim.x) {
    //     sum += in[i*X + col];
    // }
    sum = blockSum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(out + col, sum);
    }
}

__global__
void nullify(float *arr) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    arr[i] = 0;
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
    nullify<<<Y/N, N>>>(avg_stud);
    nullify<<<X/N, N>>>(avg_que);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(n/BLOCK_SIZE);

    // load all results
    reduceCols<<<blocks, threads>>>(results, avg_que, X, Y);
    reduceRows<<<blocks, threads>>>(results, avg_stud, X, Y);

    // divide results
    divide<<<Y/N, N>>>(avg_stud, X);
    divide<<<X/N, N>>>(avg_que, Y);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
