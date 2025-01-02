#include <iostream>
#include <cuda_runtime.h>


// Kernel for matrix multiplication
__global__ void matrix_multiplication(const float *A, const float *B, float *C, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void print_matrix(const float* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int row = 4;  // Smaller size for demonstration purposes
    int column = 4;
    int N = row;

    size_t size = sizeof(float) * row * column;

    // Allocate host memory
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < row * column; ++i) {
        A[i] = i;
        B[i] = i * i;
    }

    // Allocate device memory
    float *G_A, *G_B, *G_C;
    CUDA_CHECK(cudaMalloc((void**)&G_A, size));
    CUDA_CHECK(cudaMalloc((void**)&G_B, size));
    CUDA_CHECK(cudaMalloc((void**)&G_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(G_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G_B, B, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 threadsPerBlock(2, 2);  // Reduced size for easier visualization
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrix_multiplication<<<blocksPerGrid, threadsPerBlock>>>(G_A, G_B, G_C, N);
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());

    // Copy the result back from device to host
    CUDA_CHECK(cudaMemcpy(C, G_C, size, cudaMemcpyDeviceToHost));

    // Print matrices A, B, and C
    printf("Matrix A:\n");
    print_matrix(A, N);

    printf("\nMatrix B:\n");
    print_matrix(B, N);

    printf("\nResultant Matrix C:\n");
    print_matrix(C, N);

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(G_A);
    cudaFree(G_B);
    cudaFree(G_C);

    return 0;
}

