#include <iostream>
#include <cuda_runtime.h>

// Kernel for matrix transposition
__global__ void matrix_transposition(const float *input, float *output, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && column < N) {
        output[column * N + row] = input[row * N + column];
    }
}

// Function to print matrices
void printMatrix(const float *matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int N = 4;
    size_t size = sizeof(float) * N * N;

    // Host memory allocation
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);

    // Initialize matrix A
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(i);
    }

    // Device memory allocation
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(2, 2);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrix_transposition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    // Print matrices
    printf("Original Matrix A:\n");
    printMatrix(A, N);

    printf("\nTransposed Matrix B:\n");
    printMatrix(B, N);

    // Free memory
    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}

