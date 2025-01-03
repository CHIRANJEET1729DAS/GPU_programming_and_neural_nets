#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 2
#define STRIDE 1
#define PADDING 0

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

__global__ void MaxPool(const Matrix Input_Matrix, Matrix Output_matrix, int poolSize) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;  // Iterate over rows
    int column = threadIdx.x + blockIdx.x * blockDim.x;  // Iterate over columns

    if (row < Output_matrix.height && column < Output_matrix.width) {
        float max_Value = -FLT_MAX;

        for (int i = 0; i < poolSize; i++) {
            for (int j = 0; j < poolSize; j++) {
                int r = row * poolSize + i;
                int c = column * poolSize + j;

                if (r < Input_Matrix.height && c < Input_Matrix.width) {
                    max_Value = fmaxf(max_Value, Input_Matrix.elements[r * Input_Matrix.width + c]);
                }
            }
        }

        Output_matrix.elements[row * Output_matrix.width + column] = max_Value;
    }
}

int main() {
    int N = 5;  // Input matrix size
    int poolSize = 2;

    // Allocate and initialize input matrix
    Matrix Matrix_A;
    Matrix_A.width = N;
    Matrix_A.height = N;
    size_t matrix_A_size = sizeof(float) * N * N;
    Matrix_A.elements = (float*)malloc(matrix_A_size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Matrix_A.elements[i * N + j] = i * i + j * j;  // Example initialization
        }
    }

    // Allocate and initialize output matrix
    Matrix Matrix_B;
    Matrix_B.width = N / poolSize;
    Matrix_B.height = N / poolSize;
    size_t matrix_B_size = sizeof(float) * Matrix_B.width * Matrix_B.height;
    Matrix_B.elements = (float*)malloc(matrix_B_size);

    // Device memory for input and output matrices
    Matrix d_A, d_B;
    d_A.width = Matrix_A.width;
    d_A.height = Matrix_A.height;
    cudaMalloc(&d_A.elements, matrix_A_size);
    cudaMemcpy(d_A.elements, Matrix_A.elements, matrix_A_size, cudaMemcpyHostToDevice);

    d_B.width = Matrix_B.width;
    d_B.height = Matrix_B.height;
    cudaMalloc(&d_B.elements, matrix_B_size);

    // Kernel launch configuration
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((Matrix_B.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (Matrix_B.height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    MaxPool<<<dimGrid, dimBlock>>>(d_A, d_B, poolSize);

    // Copy the result back to host
    cudaMemcpy(Matrix_B.elements, d_B.elements, matrix_B_size, cudaMemcpyDeviceToHost);

    // Print the input matrix
    std::cout << "Input Matrix:" << std::endl;
    for (int i = 0; i < Matrix_A.height; i++) {
        for (int j = 0; j < Matrix_A.width; j++) {
            std::cout << Matrix_A.elements[i * Matrix_A.width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print the output matrix
    std::cout << "Max-Pooled Output Matrix:" << std::endl;
    for (int i = 0; i < Matrix_B.height; i++) {
        for (int j = 0; j < Matrix_B.width; j++) {
            std::cout << Matrix_B.elements[i * Matrix_B.width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    free(Matrix_A.elements);
    free(Matrix_B.elements);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);

    return 0;
}

