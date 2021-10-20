/**
GPGPU assignment 1: Matrix Addition in CUDA - Unified Memory/Monolithic Kernels version
    @file um_monolithic_kernels.cu
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

#include<iostream>
#include<math.h>
using namespace std;
#define M 1500
#define N 500

__global__
void matrixInit(double* A, double value)
{
    for(int i=0; i<M*N; i++)
        A[i] = value;
}

__global__
void matrixAdd(double* A, double* B, double* C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = col + row * N;

    if (col < N && row < M) {
        C[index] = A[index] + B[index];
    }
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
    int size = M * N * sizeof(double); //expect a size in bytes

    dim3 dimBlock(16,16);
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
    matrixAdd<<<dimGrid, dimBlock>>>(A, B, C);

cout<<endl<<"Sync starts"<<endl;
    cudaDeviceSynchronize();
cout<<endl<<"Sync ends"<<endl;

//printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    //printMatrix(values_C);

//free cuda memory
    cudaFree(A); 
    cudaFree(B); 
    cudaFree(C);
    
    return 0;
}