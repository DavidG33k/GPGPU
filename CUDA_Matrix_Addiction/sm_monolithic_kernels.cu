/**
GPGPU assignment 1: Matrix Addition in CUDA - Standard Memory/Monolithic Kernels version
    @file sm_monolithic_kernels.cu
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


void matrixAlloc(double** A, int m, int n)
{
    for(int i=0; i<m; i++)
        A[i]=new double[n];
}

void matrixInit(double A[][8], double value, int m, int n)
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            A[i][j] = value;
}

__global__
void matrixAdd(double* A, double* B, double* C, int m, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = col + row * m;

    if (col < n && row < m) {
        C[index] = A[index] + B[index];
    }
}

void printMatrix(double A[][8], int m, int n)
{
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
            cout<<" "<<A[i][j]<<" ";
        cout<<endl;
    }
}

void deleteMatrix(double** A, int m, int n)
{
    for(int i=0; i<m; i++)
        delete[] A[i];
    delete[] A;
}

int main()
{
    #pragma region // variables declaration
    int m = pow(2,2); //rows 12
    int n = pow(2,3); //columns 16
    int size = m * n * sizeof(int); // sizeof(int) convert numbers into bytes.
    int blockSize = 8*8; //16*16 and 32*32
    int numBlocks = (size + blockSize - 1) / blockSize;
    #pragma endregion 

    #pragma region //create and allocate matrix A, B and C
    double A[4][8];
    double B[4][8];
    double C[4][8];

    //matrixAlloc(A);
    //matrixAlloc(B);
    //matrixAlloc(C);

    double *dev_A, *dev_B, *dev_C;

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);
    #pragma endregion

    #pragma region //init all the matrix with a passed value
    matrixInit(A,1,m,n);
    matrixInit(B,2,m,n);
    matrixInit(C,0,m,n);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);
    #pragma endregion

    #pragma region //addiction operation and print results
    matrixAdd<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, m, n);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    //printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    printMatrix(C,m,n);
    #pragma endregion

    #pragma region //delete matrix
    //deleteMatrix(A);
    //deleteMatrix(B);
    //deleteMatrix(C);

    cudaFree(dev_A); 
    cudaFree(dev_B); 
    cudaFree(dev_C);
    #pragma endregion
    
    return 0;
}