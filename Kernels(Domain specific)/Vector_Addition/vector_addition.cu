#include <iostream>
#include <cuda_runtime.h>


// CUDA kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    int n = 1000000; // Size of the vectors
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    // Initialize host vectors
    for (int i = 0; i < n; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy the result from device to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < 10; ++i) // Check the first 10 results
    {
        std::cout << "h_a[" << i << "] + h_b[" << i << "] = " << h_c[i] << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}

