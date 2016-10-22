/*
TODO:
- reduce 'y' block size by 2 or 4
- switch reading to 'int2' or 'int4'
*/
#include <stdio.h>

#define N          32               // block columns, tile side
#define PARTS      4                // block partitioning
#define BLOCK_ROWS (N / PARTS)      // block rows, block 'y' side
#define BLOCKS     (N * N)          // total blocks

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
void reduce(const int *in, float *out_stud, float *out_que) {
    __shared__ int tile[N][N + 1];      // bank conflict
    int x = blockIdx.x*N + threadIdx.x;
    int y = blockIdx.y*N + threadIdx.y;
    int width = gridDim.x*N;            // width of the whole matrix

    // global indexes
    int idx1 = y*width + x;

    int val1 = in[idx1];

    tile[threadIdx.x][threadIdx.y] = val1;
    __syncthreads();

    register int sum_stud1 = val1;
    register int sum_que1 = tile[threadIdx.y][threadIdx.x];

    sum_stud1 = warpSum(sum_stud1);
    sum_que1 = warpSum(sum_que1);

    if (threadIdx.x == 0) {
        int que_i = blockIdx.x*N + threadIdx.y;
        atomicAdd(out_stud + y, sum_stud1);
        atomicAdd(out_que + que_i, sum_que1);
    }
}

__global__
void reduce2(const int *in, float *out_stud, float *out_que) {
    __shared__ int tile[N][N + 1];      // bank conflict
    int x = blockIdx.x*N + threadIdx.x;
    int y = blockIdx.y*N + threadIdx.y;
    int width = gridDim.x*N;            // width of the whole matrix

    // global indexes
    int idx1 = y*width + x;
    int idx2 = (y + BLOCK_ROWS)*width + x;

    int val1 = in[idx1];
    int val2 = in[idx2];

    tile[threadIdx.x][threadIdx.y] = val1;
    tile[threadIdx.x][threadIdx.y + BLOCK_ROWS] = val2;
    __syncthreads();

    register int sum_stud1 = val1;
    register int sum_stud2 = val2;
    register int sum_que1 = tile[threadIdx.y][threadIdx.x];
    register int sum_que2 = tile[threadIdx.y + BLOCK_ROWS][threadIdx.x];

    sum_stud1 = warpSum(sum_stud1);
    sum_stud2 = warpSum(sum_stud2);
    sum_que1 = warpSum(sum_que1);
    sum_que2 = warpSum(sum_que2);

    if (threadIdx.x == 0) {
        int que_i = blockIdx.x*N + threadIdx.y;
        atomicAdd(out_stud + y, sum_stud1);
        atomicAdd(out_stud + y + BLOCK_ROWS, sum_stud2);
        atomicAdd(out_que + que_i, sum_que1);
        atomicAdd(out_que + que_i + BLOCK_ROWS, sum_que2);
    }
}

__global__
void reduce4(const int *in, float *out_stud, float *out_que) {
    __shared__ int tile[N][N + 1];      // bank conflict
    int x = blockIdx.x*N + threadIdx.x;
    int y = blockIdx.y*N + threadIdx.y;
    int width = gridDim.x*N;            // width of the whole matrix

    // global indexes
    int idx1 = y*width + x;
    int idx2 = (y + BLOCK_ROWS)*width + x;
    int idx3 = (y + BLOCK_ROWS*2)*width + x;
    int idx4 = (y + BLOCK_ROWS*3)*width + x;

    int val1 = in[idx1];
    int val2 = in[idx2];
    int val3 = in[idx3];
    int val4 = in[idx4];

    tile[threadIdx.x][threadIdx.y] = val1;
    tile[threadIdx.x][threadIdx.y + BLOCK_ROWS] = val2;
    tile[threadIdx.x][threadIdx.y + BLOCK_ROWS*2] = val3;
    tile[threadIdx.x][threadIdx.y + BLOCK_ROWS*3] = val4;
    __syncthreads();

    register int sum_stud1 = val1;
    register int sum_stud2 = val2;
    register int sum_stud3 = val3;
    register int sum_stud4 = val4;
    register int sum_que1 = tile[threadIdx.y][threadIdx.x];
    register int sum_que2 = tile[threadIdx.y + BLOCK_ROWS][threadIdx.x];
    register int sum_que3 = tile[threadIdx.y + BLOCK_ROWS*2][threadIdx.x];
    register int sum_que4 = tile[threadIdx.y + BLOCK_ROWS*3][threadIdx.x];

    sum_stud1 = warpSum(sum_stud1);
    sum_stud2 = warpSum(sum_stud2);
    sum_stud3 = warpSum(sum_stud3);
    sum_stud4 = warpSum(sum_stud4);
    sum_que1 = warpSum(sum_que1);
    sum_que2 = warpSum(sum_que2);
    sum_que3 = warpSum(sum_que3);
    sum_que4 = warpSum(sum_que4);

    if (threadIdx.x == 0) {
        int que_i = blockIdx.x*N + threadIdx.y;
        atomicAdd(out_stud + y, sum_stud1);
        atomicAdd(out_stud + y + BLOCK_ROWS, sum_stud2);
        atomicAdd(out_stud + y + BLOCK_ROWS*2, sum_stud3);
        atomicAdd(out_stud + y + BLOCK_ROWS*3, sum_stud4);
        atomicAdd(out_que + que_i, sum_que1);
        atomicAdd(out_que + que_i + BLOCK_ROWS, sum_que2);
        atomicAdd(out_que + que_i + BLOCK_ROWS*2, sum_que3);
        atomicAdd(out_que + que_i + BLOCK_ROWS*3, sum_que4);
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
    // int n = X * Y;
    int parts = PARTS;   // TODO dynamic

    // reset arrays
    cudaMemset(avg_stud, 0, Y*sizeof(avg_stud[0]));
    cudaMemset(avg_que, 0, X*sizeof(avg_que[0]));

    dim3 threads(N, BLOCK_ROWS);
    dim3 blocks(X/N, Y/N);

    // load all results
    if (parts <= 1) {
        reduce<<<blocks, threads>>>(results, avg_stud, avg_que);
    } else if (parts <= 2) {
        reduce2<<<blocks, threads>>>(results, avg_stud, avg_que);
    } else {
        reduce4<<<blocks, threads>>>(results, avg_stud, avg_que);
    }

    // divide results - TODO merge to 'reduce'
    divide<<<Y/N, N>>>(avg_stud, X);
    divide<<<X/N, N>>>(avg_que, Y);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
