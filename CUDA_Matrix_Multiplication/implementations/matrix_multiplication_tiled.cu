/**
GPGPU assignment 2: Matrix Multiplication in CUDA
    @file matrix_multiplication_serial.cpp
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 03 November 2021 
A tiled implementation of the matrix multiplication algorithm in CUDA using the Unified Memory and the Grid-Stride Loop models.
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
    //automatic variables
    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int row = blockIdx.y * TILE + tr;
    int col = blockIdx.x * TILE + tc;

    float pValue = 0.0f;    //for every cell in P we have to reset pValue

    //variables for the grid stride loop
    int stride_c = TILE * gridDim.x;       //equivalent to int stride_j = blockDim.x * gridDim.x;
    int stride_r = TILE * gridDim.y;       //equivalent to int stride_i = blockDim.y * gridDim.y;

    //alloc shared memory
    __shared__ float Mds[TILE][TILE];
    __shared__ float Nds[TILE][TILE]; 

    //for loop managing the phase variable
    for(int ph = 0; ph < (TILE + colsM - 1)/TILE; ++ph) 
    {
        //elements in shared memory equal to 0.0f
        Mds[tr][tc] = 0.0f;
        Nds[tr][tc] = 0.0f;

        //rows
        for(int i = row; i < rowsM && ph*TILE + tc < colsM; i+=stride_r)
            Mds[tr][tc] = M[i*colsM + ph*TILE + tc];

        //cols
        for(int j = col; j < colsN && ph*TILE + tr < rowsN; j+= stride_c)            
            Nds[tr][tc] = N[(ph*TILE + tr)*colsN + j];

        __syncthreads();

        //calculate pValue
        for (int n = 0; n < TILE; ++n)
            pValue += Mds[tr][n] * Nds[n][tc];
        __syncthreads();
    }

    //out of the phase loop assign pValue to the right index in P matrix
    P[row*colsN + col] = pValue;
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

    dim3 dimGridM(ceil(colsM/dimBlock.x), ceil(rowsM/dimBlock.y));  
    dim3 dimGridN(ceil(colsN/dimBlock.x), ceil(rowsN/dimBlock.y));  
    dim3 dimGridP(ceil(colsN/dimBlock.x), ceil(rowsM/dimBlock.y));
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
#pragma endregion

#pragma region  //multiplication kernel
    matrixMulti<<<dimGridP, dimBlock>>>(M,N,P);
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

#pragma region //free cuda memory and printng execution time
    cudaFree(M); 
    cudaFree(N); 
    cudaFree(P);

    end=clock();
    cout << "Exe time: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;
#pragma region

    return 0;
}