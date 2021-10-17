/**
GPGPU assignment 1: Matrix Addition in CUDA - Unified Memory/Grid Stride Loop Kernels version
    @file um_grid_stride_loop_kernels.cu
    @author Canonaco Martina @author Gena Davide @author Morello Michele
    @version 13 October 2021 
*
Let A and B be the matrices of double-precision floating-point numbers to be added,
and C the resulting matrix; Let m = 2^12 and n=2^16 be their number of rows and columns, respectively.
*
Implement four versions of the matrix addition application in CUDA using:
    - Standard-Memory/Monolithic-Kernels;
    - Standard-Memory/Grid-Stride-Loop-Kernels;
    - Unified-Memory/Monolithic-Kernels;
    - Unified-Memory/Grid-Stride-Loop-Kernels.
*/

#include <algorithm>
#include <iostream>
using namespace std;

#define M 5
#define N 5

__global__
void add(float *A, float *B, float *C)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < M*N; i += stride)
    C[i] = A[i] + B[i];
}



int main()
{
    float* A; cudaMallocManaged(&A, M*N*sizeof(float));
    float* B; cudaMallocManaged(&B, M*N*sizeof(float));
    float* C; cudaMallocManaged(&C, M*N*sizeof(float));

    float valori_A[M][N];
    float valori_B[M][N];
    float valori_C[M][N];

    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++) {
            valori_A[i][j] = 1.0f;
            valori_B[i][j] = 2.0f;
            valori_C[i][j] = 0.0f;
        }

    //A = &valori_A[0][0];
    //B = &valori_B[0][0];
    //C = &valori_C[0][0];

    memcpy(&A[0], &valori_A[0][0], M*N*sizeof(float));
    memcpy(&B[0], &valori_B[0][0], M*N*sizeof(float));
    memcpy(&C[0], &valori_C[0][0], M*N*sizeof(float));

    for(int i=0; i<M*N; i++)
        cout << A[i] << " ";
    cout<<endl;

    for(int i=0; i<M*N; i++)
        cout << B[i] << " ";
    cout<<endl;

    for(int i=0; i<M*N; i++)
        cout << C[i] << " ";
    cout<<endl;

    int blockSize = 256;
    int numBlocks = (M * N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(A, B, C);

    cudaDeviceSynchronize();

    for(int i=0; i<M*N; i++)
        cout << C[i] << " ";
    cout<<endl;

    memcpy(&valori_C[0][0], &C[0], M*N*sizeof(float));

    for(int i=0; i<M; i++) {
      for (int j=0; j<N; j++) {
        cout << valori_C[i][j] << " ";
      }
    cout<<endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}