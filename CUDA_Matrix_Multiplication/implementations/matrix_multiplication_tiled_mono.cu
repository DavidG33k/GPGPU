/**
GPGPU assignment 2: Matrix Multiplication in CUDA
    @file matrix_multiplication_serial.cpp
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 03 November 2021 
A tiled implementation of the matrix multiplication algorithm in CUDA using the Unified Memory and the Monolithic models.
 - dims of M = 2000x500
 - dims of N = 500x2000
*/

#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<assert.h>
using namespace std;

#define rowsM 2000  //corrisponding to rowsP
#define colsM 500
#define rowsN 500
#define colsN 2000  //corrisponding to colsP
#define TILE 8  //equal to blockSize 8x8, 16x16, 32x32

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
    //create variables
    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int row = blockIdx.y * TILE + tr;
    int col = blockIdx.x * TILE + tc;

    float pValue = 0.0f;

    //alloc shared memory     
    __shared__ float Mds[TILE][TILE];
    __shared__ float Nds[TILE][TILE]; 

    for(int ph = 0; ph < (TILE + colsM - 1)/TILE; ++ph)
    {
        if(row < rowsM && ((ph * TILE + tc)) < colsM)
            Mds[tr][tc] = M[row * colsM + (ph * TILE + tc)];
        else Mds[tr][tc] = 0;

        if((ph * TILE + tr) < colsM && col < colsN)
            Nds[tr][tc] = N[(ph * TILE + tr) * colsN + col];
        else Nds[tr][tc] = 0;
            
        __syncthreads();

        for(int i = 0; i < TILE; ++i)
            pValue += Mds[tr][i] * Nds[i][tc];

        __syncthreads();
    }

    if(row < rowsM && col < colsN)
        P[col * colsN + row] = pValue;
}

int main(int argc, char* argv[])
{
   
#pragma region //managing argv, argc, time
    clock_t start, end;

	int blockSize;

    assert(colsM == rowsN);
    assert(argc == 2);

    blockSize = atoi(argv[1]);

    assert(blockSize==8 || blockSize==16 || blockSize==32);
    
#pragma endregion

#pragma region //variables declaration
    start=clock();

    float sizeM = rowsM * colsM * sizeof(float);
    float sizeN = colsM * rowsM * sizeof(float);
    float sizeP = rowsM * colsN * sizeof(float);
    
    dim3 dimBlock(blockSize,blockSize);

    dim3 dimGridM(ceil(colsM/dimBlock.x), ceil(rowsM/dimBlock.y));  //rowsM*colsM 500x2000
    dim3 dimGridN(ceil(colsN/dimBlock.x), ceil(colsM/dimBlock.y));  //colsM*colsN 2000x500
    dim3 dimGridP(ceil(colsN/dimBlock.x), ceil(rowsM/dimBlock.y));  //ceil works for float vars
#pragma endregion

#pragma region //alloc and inizialize matrices M, N and P with a passed value
    float* M; cudaMallocManaged(&M, sizeM);
    float* N; cudaMallocManaged(&N, sizeN);
    float* P; cudaMallocManaged(&P, sizeP);

    matrixInit<<<dimGridM, dimBlock>>>(M, 2.0f, rowsM, colsM);
    matrixInit<<<dimGridN, dimBlock>>>(N, 3.0f, colsM, rowsM);
    matrixInit<<<dimGridP, dimBlock>>>(P, 0.0f, rowsM, colsN);
#pragma endregion

#pragma region //multiplication operation
    matrixMulti<<<dimGridP, dimBlock>>>(M, N, P);

    cudaDeviceSynchronize();
#pragma endregion

#pragma region //check for errors (all values should be eValue)

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

#pragma region //free cuda memory and printing time
    cudaFree(M); 
    cudaFree(N); 
    cudaFree(P);

    end=clock();
    cout << "Exe time: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;
#pragma region

    return 0;
}
