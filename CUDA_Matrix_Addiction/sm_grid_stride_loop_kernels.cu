/**
GPGPU assignment 1: Matrix Addition in CUDA - Standard Memory/Grid Stride Loop Kernels version
    @file sm_grid_stride_loop_kernels.cu
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 13 October 2021 
*
Let A and B be the matrices of double-precision floating-point numbers to be added,
and C the resulting matrix; Let m = 2^12 and n=2^15 be their number of rows and columns, respectively.
*
Implement four versions of the matrix addition application in CUDA using:
    - Standard-Memory/Grid-Stride-Loop-Kernels.
*/

#include <algorithm>
#include <iostream>
using namespace std;

#define M 4096 //m=2^12 = 4096
#define N 32768 //n=2^15 = 32768

__global__
void matrixInit(double* A, double value)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < M*N; i += stride)
    A[i]=value;
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
        cout<<" "<<A[i]<<" ";
}

int main()
{
//variables declaration
    double size = M * N * sizeof(double);
    cout<<"size: "<<size<<endl;
    
    int blockSize = 256;
    int numBlocks = (M * N + blockSize - 1) / blockSize;

//create and allocate matrix A, B and C
    //allocate dynamic matrix
    double *A, *B, *C; //host matrix

    //in standard memory we have to allocate CPU
    A = (double*)malloc(size);
    B = (double*)malloc(size);
    C = (double*)malloc(size);

    double *dev_A, *dev_B, *dev_C; //device matrix

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

//init all the matrix with a passed value
    matrixInit<<<numBlocks, blockSize>>>(dev_A,1.0);
    matrixInit<<<numBlocks, blockSize>>>(dev_B,2.0);
    matrixInit<<<numBlocks, blockSize>>>(dev_C,0.0);
    cout<<endl<<"M-init done"<<endl;

//addiction operation and print results
    cout<<endl<<"add starts"<<endl;
    matrixAdd<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C);
    cout<<endl<<"add ends"<<endl;

    // cout<<endl<<"synch starts"<<endl;
    // cudaDeviceSynchronize();
    // cout<<endl<<"synch ends"<<endl;

    cout<<endl<<"final copy starts"<<endl;
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);
    cout<<endl<<"final copy ends"<<endl;

//printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    //printMatrix(C);

//free cuda memory
    cudaFree(dev_A); 
    cudaFree(dev_B); 
    cudaFree(dev_C);

    return 0;
}