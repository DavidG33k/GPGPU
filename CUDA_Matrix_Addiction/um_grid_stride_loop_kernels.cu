/**
GPGPU assignment 1: Matrix Addition in CUDA - Unified Memory/Grid Stride Loop Kernels version
    @file um_grid_stride_loop_kernels.cu
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 13 October 2021 
*
Let A and B be the matrices of double-precision floating-point numbers to be added,
and C the resulting matrix; Let m = 2^12 and n=2^15 be their number of rows and columns, respectively.
*
Implement four versions of the matrix addition application in CUDA using:
    - Unified-Memory/Grid-Stride-Loop-Kernels.
*/

#include <algorithm>
#include <iostream>
using namespace std;

#define M (1 << 12) //m=2^12 = 4096
#define N (1 << 15) //n=2^15 = 32768

__global__
void matrixInit(double* A, double value)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;

  for (int i = index_x; i < M; i += stride_x)
    for (int j = index_y; j < N; j += stride_y)
        A[j*M+i]=value;
}

__global__
void matrixAdd(double *A, double *B, double *C)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;

  for (int i = index_x; i < M; i += stride_x)
    for (int j = index_y; j < N; j += stride_y)
        C[j*M+i] = A[j*M+i] + B[j*M+i];
}

void printMatrix(double* A)
{
    for(int i=0; i<M*N; i++)
        cout<<" "<<A[i]<<" ";
}

int main()
{
//variables declaration
    double size = M * N * sizeof(double); //expect a size in bytes
    cout<<"size: "<<size<<endl;

    dim3 dimBlock(32,32);
    dim3 dimGrid(((N+dimBlock.x-1)/dimBlock.x),((M+dimBlock.y-1)/dimBlock.y));


//create and allocate matrix A, B and C
    double* A; cudaMallocManaged(&A, size);
    double* B; cudaMallocManaged(&B, size);
    double* C; cudaMallocManaged(&C, size);

//init all the matrix with a passed value
    matrixInit<<<dimGrid, dimBlock>>>(A,1.0);
    matrixInit<<<dimGrid, dimBlock>>>(B,2.0);
    matrixInit<<<dimGrid, dimBlock>>>(C,0.0);
    cout<<endl<<"M-init done"<<endl;
    
//addiction operation and print results
    cout<<endl<<"add starts"<<endl;
    matrixAdd<<<dimGrid, dimBlock>>>(A, B, C);

    cout<<endl<<"Sync starts"<<endl;
    cudaDeviceSynchronize();
    cout<<endl<<"Sync ends"<<endl;

//printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    //printMatrix(C);

// Check for errors (all values should be 3.0f)
    float maxError = 0;
    for (int i = 0; i < M * N; i++)
	maxError=fmax(maxError, fabs(C[i]-3.0f));
    cout << "Max error: " << maxError << endl;


    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}