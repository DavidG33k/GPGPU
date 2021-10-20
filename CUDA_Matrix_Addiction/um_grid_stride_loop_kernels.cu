/**
GPGPU assignment 1: Matrix Addition in CUDA - Unified Memory/Grid Stride Loop Kernels version
    @file um_grid_stride_loop_kernels.cu
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
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

#define M 1024
#define N 2048

__global__
void matrixInit(double* A, double value)
{
    for(int i=0; i<M*N; i++)
        A[i] = value;
}

__global__
void matrixAdd(double *A, double *B, double *C)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < M*N; i += stride)
    C[i] = A[i] + B[i];
}

void printMatrix(double* A)
{
    for(int i=0; i<M*N; i++)
    {
        cout<<" "<<A[i]<<" ";
        if(i%6==0)
            cout<<endl;
    }
}

int main()
{
//variables declaration
    int size = M * N * sizeof(double); //expect a size in bytes

    int blockSize = 256;
    int numBlocks = (M * N + blockSize - 1) / blockSize;


//create and allocate matrix A, B and C
    double* A; cudaMallocManaged(&A, size);
    double* B; cudaMallocManaged(&B, size);
    double* C; cudaMallocManaged(&C, size);

//init all the matrix with a passed value
    matrixInit<<<numBlocks, blockSize>>>(A,1.0);
    matrixInit<<<numBlocks, blockSize>>>(B,2.0);
    matrixInit<<<numBlocks, blockSize>>>(C,0.0);
    cout<<endl<<"M-init done"<<endl;
    
//addiction operation and print results

    matrixAdd<<<numBlocks, blockSize>>>(A, B, C);

cout<<endl<<"Sync starts"<<endl;
    cudaDeviceSynchronize();
cout<<endl<<"Sync ends"<<endl;

//printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    //printMatrix(C);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}