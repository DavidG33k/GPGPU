/**
GPGPU assignment 1: Matrix Addition in CUDA - Unified Memory/Monolithic Kernels version
    @file um_monolithic_kernels.cu
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

#include<iostream>
#include<math.h>
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

    double values_A[M][N];
    double values_B[M][N];
    double values_C[M][N];

//init all the matrix with a passed value
    matrixInit(values_A, 1.0f);
    matrixInit(values_B, 2.0f);
    matrixInit(values_C, 0.0f);

    memcpy(&A[0], &values_A[0][0], size);
    memcpy(&B[0], &values_B[0][0], size);
    memcpy(&C[0], &values_C[0][0], size);
 
 //addiction operation and print results
    matrixAdd<<<dimGrid, dimBlock>>>(A, B, C);
    cudaDeviceSynchronize();

    memcpy(&values_C[0][0], &C[0], size);

//printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    printMatrix(values_C);

//free cuda memory
    cudaFree(A); 
    cudaFree(B); 
    cudaFree(C);
    
    return 0;
}