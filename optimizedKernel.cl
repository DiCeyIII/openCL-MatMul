#define BLOCK_SIZE 16
#define COARSE_SIZE 4  // Adjust as needed

__kernel void optimizedGEMM_Coarsened(const int M, const int N, const int K,
                                    const __global float* A,
                                    const __global float* B,
                                    __global float* C) {

    // Thread identifiers
    int row = get_local_id(0);
    int col = get_local_id(1);
    int globalRow = get_group_id(0) * BLOCK_SIZE + row;
    int globalCol = get_group_id(1) * BLOCK_SIZE + col;

    // Shared memory for tiles of A and B
    __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    // Initialize the accumulation register
    float acc = 0.0f;

    // Loop over tiles (K dimension)
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {

        // Load A tile into shared memory (coarsened)
        int tiledRow = t * BLOCK_SIZE + row;
        int tiledCol = col;
        if (tiledRow < M && tiledCol < K && row % COARSE_SIZE == 0 && col % COARSE_SIZE == 0) {
            Asub[row][col] = A[tiledRow * M + tiledCol];
        }

        // Load B tile into shared memory (coarsened)
        tiledRow = row;
        tiledCol = t * BLOCK_SIZE + col;
        if (tiledCol < N && tiledRow < K && row % COARSE_SIZE == 0 && col % COARSE_SIZE == 0) {
            Bsub[row][col] = B[tiledCol * K + tiledRow];
        }

        // Synchronize threads before computation
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute the dot product for the current tile (coarsened)
        for (int k = 0; k < BLOCK_SIZE; k += COARSE_SIZE) {
            acc += Asub[row][k] * Bsub[k][col];
        }

        // Synchronize threads after computation
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the result in global memory (out-of-bounds check)
    if (globalRow < M && globalCol < N) {
        C[globalCol * M + globalRow] = acc;
    }
}

