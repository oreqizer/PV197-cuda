/*
TODO:
- reduce 'y' block size by 2 or 4
- switch reading to 'int2' or 'int4'
*/
#include <stdio.h>

#define N 32  // also the tile size
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
void reduce(const int *in, float *out_stud, float *out_que, int X, int Y) {
    __shared__ int tile[N][N + 1];      // bank conflict
    int x = blockIdx.x*N + threadIdx.x;
    int y = blockIdx.y*N + threadIdx.y;
    int width = gridDim.x*N;            // width of the whole matrix
    int idx = y*width + x;              // global index
    int val = in[idx];

    tile[threadIdx.x][threadIdx.y] = val;
    __syncthreads();

    register int sum_stud = val;
    register int sum_que = tile[threadIdx.y][threadIdx.x];

    sum_stud = warpSum(sum_stud);
    sum_que = warpSum(sum_que);

    if (threadIdx.x == 0) {
        int stud_i = idx / X;
        int que_i = y + (idx & X - 1);
        atomicAdd(out_stud + stud_i, sum_stud);
        atomicAdd(out_que + que_i, sum_que);
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

    dim3 threads(N, N);
    dim3 blocks(n/BLOCK_SIZE);

    // load all results
    reduce<<<blocks, threads>>>(results, avg_stud, avg_que, X, Y);

    // divide results - TODO merge to 'reduce'
    divide<<<Y/N, N>>>(avg_stud, X);
    divide<<<X/N, N>>>(avg_que, Y);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
