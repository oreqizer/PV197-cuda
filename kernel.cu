/*
TODO:
- reduce 'y' block size by 2 or 4
- switch reading to 'int2' or 'int4'
*/
#include <stdio.h>

#define N          32               // block columns, tile side
#define BLOCK_ROWS (N / 4)          // block rows, block 'y' side
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
void reduce(const int *in, float *out_stud, float *out_que, int X, int Ystep) {
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
    register int sum_que =
        tile[threadIdx.y][threadIdx.x] +
        tile[threadIdx.y + BLOCK_ROWS][threadIdx.x] +
        tile[threadIdx.y + BLOCK_ROWS*2][threadIdx.x] +
        tile[threadIdx.y + BLOCK_ROWS*3][threadIdx.x];

    sum_stud1 = warpSum(sum_stud1);
    sum_stud2 = warpSum(sum_stud2);
    sum_stud3 = warpSum(sum_stud3);
    sum_stud4 = warpSum(sum_stud4);
    sum_que = warpSum(sum_que);

    if (threadIdx.x == 0) {
        int stud_i = idx1 / X;
        int que_i = y + (idx1 & X - 1);
        // printf("%d %d -> %d | %d\n", x, y, stud_i, stud_i + BLOCK_ROWS);
        atomicAdd(out_stud + stud_i, sum_stud1);
        atomicAdd(out_stud + stud_i + Ystep, sum_stud2);
        atomicAdd(out_stud + stud_i + Ystep*2, sum_stud3);
        atomicAdd(out_stud + stud_i + Ystep*3, sum_stud4);
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

    dim3 threads(N, BLOCK_ROWS);
    dim3 blocks(n/BLOCKS);

    // load all results
    reduce<<<blocks, threads>>>(results, avg_stud, avg_que, X, Y/4);

    // divide results - TODO merge to 'reduce'
    divide<<<Y/N, N>>>(avg_stud, X);
    divide<<<X/N, N>>>(avg_que, Y);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
