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
    float *avg_stud,     // score per student: total / questions
    float *avg_que,      // score per question: total / students
    const int students,  // y: always divisible by 32
    const int questions  // x: always divisible by 32
) {
    int n = questions * students;

    dim3 grid(n/BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    // reset arrays
    nullify<<<students/N, N>>>(avg_stud);
    nullify<<<questions/N, N>>>(avg_que);

    // load all results
    solver<<<grid, block>>>(
        results, avg_stud, avg_que, questions, students
    );

    // divide results
    divide<<<students/N, N>>>(avg_stud, questions);
    divide<<<questions/N, N>>>(avg_que, students);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
