#include <stdio.h>

#define N      32       // block rows, tile side
#define C      (N / 4)  // block columns
#define BLOCKS (N * N)  // total blocks

template <int T>
__global__ void reduce(const int *in, float *out_stud, float *out_que) {
    __shared__ int tile[T * (T + 1)];
    int x4 = blockIdx.x*(T / 4) + threadIdx.x;
    int y  = blockIdx.y* T      + threadIdx.y;
    int width = gridDim.x*(T / 4);

    int4 val = reinterpret_cast<const int4*>(in)[y*width + x4];
    reinterpret_cast<int4*>(tile)[(threadIdx.y*(T / 4)) + threadIdx.x] = val;
    __syncthreads();

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < T; i++) {
            sum += tile[threadIdx.y*T + i];
        }
        atomicAdd(out_stud + y, sum);
    } else if (threadIdx.x == 1) {
        int x = blockIdx.x*T + threadIdx.y;
        int sum = 0;
        for (int i = 0; i < T; i++) {
            sum += tile[i*T + threadIdx.y];
        }
        atomicAdd(out_que + x, sum);
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
    // int parts = n/BLOCKS;

    // reset arrays
    cudaMemset(avg_stud, 0, Y*sizeof(avg_stud[0]));
    cudaMemset(avg_que, 0, X*sizeof(avg_que[0]));

    dim3 blocks(X/N, Y/N);
    dim3 threads(N/4, N);
    reduce<N><<<blocks, threads>>>(results, avg_stud, avg_que);

    // divide results
    divide<<<Y/N, N>>>(avg_stud, X);
    divide<<<X/N, N>>>(avg_que, Y);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
