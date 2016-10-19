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
void blockSum(int *shared, int val) {
    int x = threadIdx.x % warpSize;
    int y = threadIdx.x / warpSize;

    val = warpSum(val);             // reduce warp
    if (x == 0) shared[y] = val;    // write result to shared memory
}

__global__
void reduceRows(const int *in, float *out, int X, int Y) {
    __shared__ int shared[N];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int sum = in[i];
    // for (int i = x; i < X; i += blockDim.x * gridDim.x) {
    //     sum += in[i];
    // } TODO: grid stride

    blockSum(shared, sum); // TODO
}

__global__
void reduceCols(const int *in, float *out, int X, int Y) {
    int y = blockIdx.x*blockDim.x + threadIdx.x;
    for (; y < Y; y += blockDim.x * gridDim.x) {
        int sum = 0;
        for (int x = 0; x < X; x++) {
            sum += in[y*X + x];
        }
        out[y] = sum;
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
    reduceRows<<<gridQ, 1024>>>(results, avg_que, questions, students);

    // divide results
    divide<<<students/N, N>>>(avg_stud, questions);
    divide<<<questions/N, N>>>(avg_que, students);

    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}
