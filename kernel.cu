#include <stdio.h>


__global__ void solver(
    const int *results,
    float *avg_stud,
    float *avg_que,
    const int dim_x,  // always divisible by 32
    const int dim_y  // always divisible by 32
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int resultIds = y*dim_x + x;

    // add to resulting array
    avg_que[x] += results[resultIds];
    avg_stud[y] += results[resultIds];
}

void solveGPU(
    const int *results,  // students * questions
    float *avg_stud,
    float *avg_que,
    const int students,  // y: always divisible by 32
    const int questions  // x: always divisible by 32
) {
    solver<<<questions, students>>>(
        results, avg_stud, avg_que, questions, students
    );
}
