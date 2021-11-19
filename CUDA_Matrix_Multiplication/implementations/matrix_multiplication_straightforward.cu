/**
GPGPU assignment 2: Matrix Multiplication in CUDA
    @file matrix_multiplication_serial.cpp
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 03 November 2021 
A straightforward implementation of the matrix multiplication algorithm in CUDA using the Unified Memory and the Monolithic models.
 - dims of M = 2000x500
 - dims of N = 500x2000
*/

#include<iostream>
#include<math.h>
#include<time.h>
#include<assert.h>
using namespace std;

#define rowsM 2000
#define colsM 500
#define rowsN 500
#define colsN 2000

__global__ void matrixInit(float* A, float value, int row, int col)
{
  int index_i = blockIdx.x * blockDim.x + threadIdx.x;
  int index_j = blockIdx.y * blockDim.y + threadIdx.y;
  int stride_i = blockDim.x * gridDim.x;
  int stride_j = blockDim.y * gridDim.y;

  for (int i = index_i; i < row; i += stride_i)
    for (int j = index_j; j < col; j += stride_j)
        A[j*row+i]=value;
}


__global__ void matrixMulti(float* M, float* N, float* P)
{
    int tc = threadIdx.x;
    int tr = threadIdx.y;
    
    // Calculate the column index of P and N
    int col = blockIdx.x*blockDim.x + tc;
    // Calculate the row index of the P element and M
    int row = blockIdx.y*blockDim.y + tr;
    
    int stride_i = blockDim.y * gridDim.y;       //equivalent to int stride_i = blockDim.y * gridDim.y;
    int stride_j = blockDim.x * gridDim.x;       //equivalent to int stride_j = blockDim.x * gridDim.x;

    for (int i = row; i < rowsM; i += stride_i)
    {
        for (int j = col; j < colsN; j += stride_j)
        {
            float pValue = 0.0f;

            // each thread computes one element of the block sub-matrix
            for (int k = 0; k < colsM; ++k)         
                pValue += M[i*colsM+k] * N[k*colsN+j];
            
            P[i*colsN+j] = pValue;
        }
    }

}

__global__ void matrixMultiMono(float* M, float* N, float* P)
{
    int tc = threadIdx.x;
    int tr = threadIdx.y;
    
    // Calculate the column index of P and N
    int col = blockIdx.x*blockDim.x + tc;
    // Calculate the row index of the P element and M
    int row = blockIdx.y*blockDim.y + tr;

    if( row < rowsM && col < colsN)
    {
        float pValue=0.0f;
        for(int k=0; k < colsM; ++k)    //rowsN is the same
            pValue += M[row*colsM + k] * N[k*colsN + col];
        P[row*colsN + col] = pValue;
    }
}


int main(int argc, char* argv[])
{
   
#pragma region  //managing argv, argc, time
    clock_t start, end;

    int blockSize;

    //colsM have to be equal to rowsN so you can do matrix multi
    assert(colsM == rowsN);
    assert(argc == 2);

    blockSize = atoi(argv[1]);

    assert(blockSize==8 || blockSize==16 || blockSize==32);    
#pragma endregion

#pragma region //variables declaration
    start=clock();
    float sizeM = rowsM * colsM * sizeof(float);
    float sizeN = rowsN * colsN * sizeof(float);
    float sizeP = rowsM * colsN * sizeof(float);
    
    dim3 dimBlock(blockSize,blockSize);
        
    dim3 dimGridM(ceil(colsM/dimBlock.x), ceil(rowsM/dimBlock.y));  //rowsM*colsM 500x2000
    dim3 dimGridN(ceil(colsN/dimBlock.x), ceil(rowsN/dimBlock.y));  //rowsN*colsN 2000x500
    dim3 dimGridP(ceil(colsN/dimBlock.x), ceil(rowsM/dimBlock.y));  //ceil works for float numbers
#pragma endregion

#pragma region //inizialize matrices M, N and P
    //unified memory allocation
    float* M; cudaMallocManaged(&M, sizeM);
    float* N; cudaMallocManaged(&N, sizeN);
    float* P; cudaMallocManaged(&P, sizeP);

    //init kernel
    matrixInit<<<dimGridM, dimBlock>>>(M, 2.0f, rowsM, colsM);
    matrixInit<<<dimGridN, dimBlock>>>(N, 3.0f, rowsN, colsN);
    matrixInit<<<dimGridP, dimBlock>>>(P, 0.0f, rowsM, colsN);
    cudaDeviceSynchronize();
#pragma endregion

#pragma region //multiplication operation

    matrixMulti<<<dimGridP, dimBlock>>>(M, N, P);
    //matrixMultiMono<<<dimGridP, dimBlock>>>(M, N, P);

    cudaDeviceSynchronize();
#pragma endregion

#pragma region  //checking errors

    //in this example i expect that all the elements are equal to eValue
    float eValue=3000.0f;

    cout<<"P[0] = "<<P[0]<<endl;
    cout<<"P[last] = "<<P[rowsM*colsN-1]<<endl;

    int cont=0;
    for (int i = 0; i < rowsM * colsN; ++i)
    {
        if(P[i]!=eValue)
            cont++;
    }
    cout<<"Missing elements: "<<cont<<endl;

    float maxError = 0;
    for (int i = 0; i < rowsM * colsN; ++i)
	    maxError=fmax(maxError, fabs(P[i]-eValue));
    cout << "Max error: " << maxError << endl;

#pragma endregion

#pragma region //free cuda memory and printing execution time
    cudaFree(M); 
    cudaFree(N); 
    cudaFree(P);

    end=clock();
    cout << "Exe time: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;
#pragma region

    return 0;
}
