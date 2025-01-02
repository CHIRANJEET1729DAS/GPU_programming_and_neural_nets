#include <iostream>
#include <cuda_runtime.h>

__global__ void convolution(const float *input_matrix,float *output_matrix,float *kernel,int N,int K){

  int row  = threadIdx.y + blockIdx.y*blockDim.y;
  int column = threadIdx.x + blockIdx.x*blockDim.x;

  int halfK = K/2;

  if (row<N && column<N && halfK<N){
      float sum = 0.0f;
      for(int i=-halfK;i<=halfK;++i){
	for(int j=-halfK;j<=halfK;j++){
          int r = row + i ;
	  int c = column + j ;
	  if (r>=0&&j>=0){
	   sum += input_matrix[r*N+c] * kernel[(i+halfK)*K+(j+halfK)];
	  }
	}   
      }
      output_matrix[row*N+column] = sum ;
  }
} 
void printMatrix(const float *matrix, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
            printf("%6.2f ", matrix[i * N + j]);
        }
    printf("\n");
  }
}
int main(){
  int N = 5;
  int K = 3; 

  size_t inputSize = sizeof(float)*N*N;
  size_t kernelSize = sizeof(float)*K*K;
  size_t outputSize = sizeof(float)*(N-K+1)*(N-K+1);

  float *Input_Matrix = (float*)malloc(inputSize);
  float *Kernel_Matrix = (float*)malloc(kernelSize);
  float *Output_Matrix = (float*)malloc(outputSize);

  for (int i = 0; i < N * N; i++) { Input_Matrix[i] = static_cast<float>(i + 1);}
  for (int i = 0; i < K * K; i++) { Kernel_Matrix[i] = static_cast<float>(i + 1);}

  float *d_input_matrix,*d_kernel_matrix,*d_output_matrix;
  cudaMalloc((void **)&d_input_matrix,inputSize);
  cudaMalloc((void **)&d_kernel_matrix,kernelSize);
  cudaMalloc((void **)&d_output_matrix,outputSize);
   
  cudaMemcpy(d_input_matrix,Input_Matrix,inputSize,cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel_matrix,Kernel_Matrix,kernelSize,cudaMemcpyHostToDevice);

  dim3 threadPerBlock(16,16);
  dim3 blockPerGrid((outputSize+threadPerBlock.x-1)/threadPerBlock.x,(outputSize+threadPerBlock.y-1)/threadPerBlock.y);

  convolution<<<blockPerGrid,threadPerBlock>>>(d_input_matrix,d_output_matrix,d_kernel_matrix,N,K);

  cudaMemcpy(Output_Matrix,d_output_matrix,outputSize,cudaMemcpyDeviceToHost);
  
  printf("Input Matrix ::\n");
  printMatrix(Input_Matrix,N);
  printf("Kernel Matrix ::\n");
  printMatrix(Kernel_Matrix,K);
  printf("Output Matrix ::\n");
  printMatrix(Output_Matrix,N-K+1);

  free(Input_Matrix);
  free(Output_Matrix);
  free(Kernel_Matrix);
  
  cudaFree(d_input_matrix);
  cudaFree(d_output_matrix);
  cudaFree(d_kernel_matrix);

  return 0;
}
