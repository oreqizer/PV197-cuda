#include <stdio.h>

#define N           32               // block columns, tile side
#define BLOCK_ROWS2 (N / 2)          // block rows, block 'y' side
#define BLOCK_ROWS4 (N / 4)          // block rows, block 'y' side
#define BLOCKS      (N * N)          // total blocks

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
    int blockX = blockIdx.x*N;
    int x = blockX + threadIdx.x;
    int y = blockIdx.y*N + threadIdx.y;
    int width = gridDim.x*N;            // width of the whole matrix

    // values
    int val1 = in[y*width + x];

    tile[threadIdx.x][threadIdx.y] = val1;
    __syncthreads();

    register int sum_stud1 = val1;
    register int sum_que1 = tile[threadIdx.y][threadIdx.x];

    sum_stud1 = warpSum(sum_stud1);
    sum_que1 = warpSum(sum_que1);

    if (threadIdx.x == 0) {
        int que_i = blockX + threadIdx.y;
        atomicAdd(out_stud + y, sum_stud1);
        atomicAdd(out_que + que_i, sum_que1);
    }
}

__global__
void reduce2(const int *in, float *out_stud, float *out_que) {
    __shared__ int tile[N][N + 1];      // bank conflict
    int blockX = blockIdx.x*N;
    int x = blockX + threadIdx.x;
    int y = blockIdx.y*N + threadIdx.y;
    int width = gridDim.x*N;            // width of the whole matrix

    // values
    int val1 = in[y*width + x];
    int val2 = in[(y + BLOCK_ROWS2)*width + x];

    tile[threadIdx.x][threadIdx.y] = val1;
    tile[threadIdx.x][threadIdx.y + BLOCK_ROWS2] = val2;
    __syncthreads();

    register int sum_stud1 = val1;
    register int sum_stud2 = val2;
    register int sum_que1 = tile[threadIdx.y][threadIdx.x];
    register int sum_que2 = tile[threadIdx.y + BLOCK_ROWS2][threadIdx.x];

    sum_stud1 = warpSum(sum_stud1);
    sum_stud2 = warpSum(sum_stud2);
    sum_que1 = warpSum(sum_que1);
    sum_que2 = warpSum(sum_que2);

    if (threadIdx.x == 0) {
        int que_i = blockX + threadIdx.y;
        atomicAdd(out_stud + y, sum_stud1);
        atomicAdd(out_stud + y + BLOCK_ROWS2, sum_stud2);
        atomicAdd(out_que + que_i, sum_que1);
        atomicAdd(out_que + que_i + BLOCK_ROWS2, sum_que2);
    }
}

__global__
void reduce4(const int *in, float *out_stud, float *out_que) {
    __shared__ int tile[N][N + 1];      // bank conflict
    int blockX = blockIdx.x*N;
    int x = blockX + threadIdx.x;
    int y = blockIdx.y*N + threadIdx.y;
    int width = gridDim.x*N;            // width of the whole matrix

    // values
    int val1 = in[y*width + x];
    int val2 = in[(y + BLOCK_ROWS4)*width + x];
    int val3 = in[(y + BLOCK_ROWS4*2)*width + x];
    int val4 = in[(y + BLOCK_ROWS4*3)*width + x];

    tile[threadIdx.x][threadIdx.y] = val1;
    tile[threadIdx.x][threadIdx.y + BLOCK_ROWS4] = val2;
    tile[threadIdx.x][threadIdx.y + BLOCK_ROWS4*2] = val3;
    tile[threadIdx.x][threadIdx.y + BLOCK_ROWS4*3] = val4;
    __syncthreads();

    register int sum_stud1 = val1;
    register int sum_stud2 = val2;
    register int sum_stud3 = val3;
    register int sum_stud4 = val4;
    register int sum_que1 = tile[threadIdx.y][threadIdx.x];
    register int sum_que2 = tile[threadIdx.y + BLOCK_ROWS4][threadIdx.x];
    register int sum_que3 = tile[threadIdx.y + BLOCK_ROWS4*2][threadIdx.x];
    register int sum_que4 = tile[threadIdx.y + BLOCK_ROWS4*3][threadIdx.x];

    sum_stud1 = warpSum(sum_stud1);
    sum_stud2 = warpSum(sum_stud2);
    sum_stud3 = warpSum(sum_stud3);
    sum_stud4 = warpSum(sum_stud4);
    sum_que1 = warpSum(sum_que1);
    sum_que2 = warpSum(sum_que2);
    sum_que3 = warpSum(sum_que3);
    sum_que4 = warpSum(sum_que4);

    if (threadIdx.x == 0) {
        int que_i = blockX + threadIdx.y;
        atomicAdd(out_stud + y, sum_stud1);
        atomicAdd(out_stud + y + BLOCK_ROWS4, sum_stud2);
        atomicAdd(out_stud + y + BLOCK_ROWS4*2, sum_stud3);
        atomicAdd(out_stud + y + BLOCK_ROWS4*3, sum_stud4);
        atomicAdd(out_que + que_i, sum_que1);
        atomicAdd(out_que + que_i + BLOCK_ROWS4, sum_que2);
        atomicAdd(out_que + que_i + BLOCK_ROWS4*2, sum_que3);
        atomicAdd(out_que + que_i + BLOCK_ROWS4*3, sum_que4);
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
    int parts = n/BLOCKS;

    // reset arrays
    cudaMemset(avg_stud, 0, Y*sizeof(avg_stud[0]));
    cudaMemset(avg_que, 0, X*sizeof(avg_que[0]));

    dim3 blocks(X/N, Y/N);

    // load all results
    if (parts <= 4) {
        dim3 threads(N, N);
        reduce<<<blocks, threads>>>(results, avg_stud, avg_que);
    } else if (parts <= 8) {
        dim3 threads(N, BLOCK_ROWS2);
        reduce2<<<blocks, threads>>>(results, avg_stud, avg_que);
    } else {
        dim3 threads(N, BLOCK_ROWS4);
        reduce4<<<blocks, threads>>>(results, avg_stud, avg_que);
    }

    // divide results
    divide<<<Y/N, N>>>(avg_stud, X);
    divide<<<X/N, N>>>(avg_que, Y);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
