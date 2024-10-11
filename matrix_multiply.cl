
#define MATRIX_SIZE 8

__kernel void matrixMultiply(__global float* A, __global float* B, __global float* C)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;

    for(int i = 0; i < MATRIX_SIZE; ++i)
    {
        sum += A[row * MATRIX_SIZE + i] * B[i * MATRIX_SIZE + col];

    }
    C[row * MATRIX_SIZE + col] = sum;

}