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
void reduceRows(const int *in, float *out, int X, int Y) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    int sum = 0;
    for (int y = 0; y < Y; y++) {
        sum += in[y*X + x];
    }
    out[x] = sum;
}

__global__
void reduceCols(const int *in, float *out, int X, int Y) {
    int y = blockIdx.x*blockDim.x + threadIdx.x;

    int sum = 0;
    for (int x = 0; x < X; x++) {
        sum += in[y*X + x];
    }
    out[y] = sum;
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
    const int students,  // Y: always divisible by 32
    const int questions  // X: always divisible by 32
) {
    // reset arrays
    nullify<<<students/N, N>>>(avg_stud);
    nullify<<<questions/N, N>>>(avg_que);

    dim3 gridS(students/N);
    dim3 gridQ(questions/N);
    dim3 block(N);

    // load all results
    reduceCols<<<gridS, block>>>(results, avg_stud, questions, students);
    reduceRows<<<gridQ, block>>>(results, avg_que, questions, students);

    // divide results
    divide<<<students/N, N>>>(avg_stud, questions);
    divide<<<questions/N, N>>>(avg_que, students);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
