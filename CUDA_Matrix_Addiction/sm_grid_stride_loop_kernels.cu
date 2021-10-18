/**
GPGPU assignment 1: Matrix Addition in CUDA - Standard Memory/Grid Stride Loop Kernels version
    @file sm_grid_stride_loop_kernels.cu
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

void matrixInit(double A[][N], double value)
{
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++)
            A[i][j] = value;
}

__global__
void matrixAdd(double *A, double *B, double *C)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < M*N; i += stride)
    C[i] = A[i] + B[i];
}

void printMatrix(double A[][N])
{
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
            cout<<" "<<A[i][j]<<" ";
        cout<<endl;
    }
}

int main()
{
//variables declaration
    double size = M * N * sizeof(double);
    
    int blockSize = 256;
    int numBlocks = (M * N + blockSize - 1) / blockSize;

//create and allocate matrix A, B and C
    double A[M][N];
    double B[M][N];
    double C[M][N];

    double *dev_A, *dev_B, *dev_C;

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

//init all the matrix with a passed value
    matrixInit(A,1.0f);
    matrixInit(B,2.0f);
    matrixInit(C,0.0f);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

//addiction operation and print results
    matrixAdd<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

//printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    printMatrix(C);

//free cuda memory
    cudaFree(dev_A); 
    cudaFree(dev_B); 
    cudaFree(dev_C);

    return 0;
}