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

void matrixInit(float A[][N], float value)
{
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++)
            A[i][j] = value;
}

__global__
void matrixAdd(float* A, float* B, float* C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = col + row * N;

    if (col < N && row < M) {
        C[index] = A[index] + B[index];
    }
}

void printMatrix(float A[][N])
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
// variables declaration
    dim3 dimBlock(16,16);
    dim3 dimGrid(((N+dimBlock.x-1)/dimBlock.x),((M+dimBlock.y-1)/dimBlock.y));

 //create and allocate matrix A, B and C
    float* A; cudaMallocManaged(&A, M*N*sizeof(float));
    float* B; cudaMallocManaged(&B, M*N*sizeof(float));
    float* C; cudaMallocManaged(&C, M*N*sizeof(float));

    float valori_A[M][N];
    float valori_B[M][N];
    float valori_C[M][N];

    matrixInit(valori_A, 1.0f);
    matrixInit(valori_B, 2.0f);
    matrixInit(valori_C, 0.0f);

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

 
 //addiction operation and print results
    matrixAdd<<<dimGrid, dimBlock>>>(A, B, C);
    cudaDeviceSynchronize();

    memcpy(&valori_C[0][0], &C[0], M*N*sizeof(float));

    //printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    printMatrix(valori_C);

    cudaFree(A); 
    cudaFree(B); 
    cudaFree(C);
    
    return 0;
}