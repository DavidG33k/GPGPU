/**
GPGPU assignment 2: Matrix Multiplication in CUDA
    @file matrix_multiplication_serial.cpp
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 03 November 2021 
*A serial implementation of the matrix multiplication algorithm in C/C++.
 - dims of M = 2000x500
 - dims of N = 500x2000
*/

#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
using namespace std;

#define d1 2000
#define d2 500
#define d3 2000

__global__
void matrixInit(float* A, float value, int raw, int col)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;

  for (int i = index_x; i < raw; i += stride_x)
    for (int j = index_y; j < col; j += stride_y)
        A[j*raw+i]=value;
}

__global__
void matrixMulti(float* dev_M, float* dev_N, float* dev_P, unsigned j, unsigned k, unsigned l, int tileSize)
{
     
     __shared__
     float Mds[tileSize][tileSize];
     __shared__
     float Nds[tileSize][tileSize];
     
     
     int tc = threadIdx.x;
     int tr = threadIdx.y;
     int Row = blockIdx.y * tileSize + tr;
     int Col = blockIdx.x * tileSize + tc;

     float Pvalue = 0.0f;
     for(int ph = 0; ph < k/tileSize; ++ph)
     {
          if((Row < j) && (ph * tileSize + tc) < k)
               Mds[tr][tc] = dev_M[Row * k + ph * tileSize +tc];
          else
               Mds[tr][tc] = 0;
          if((ph * tileSize +tr) < k && Col < l)
               Nds[tr][tc] = dev_N[ph * tileSize + tc];
          else
               Nds[tr][tc] = 0;
          __syncthreads();
          for(int i = 0; i < tileSize; ++i)
               Pvalue += Mds[tr][i] * Nds[i][tc];
          __syncthreads();
     }
     if((Row < j) && (Col < l))
          dev_P[Row * l + Col] = Pvalue;
}

int main(int argc, char* argv[])
{
#pragma region //managing argv && argc
	int blockSize,
	int tileSize;
    if(argc != 2){
    	cout<<"no Block Size declared!"<<endl;
    	return 0;
    }
    
    blockSize = atoi(argv[1]);
    
    if(blockSize!=8 && blockSize!=16 && blockSize!=32){
    	cout<<"Invalid Block Size!"<<endl;
    	return 0;
    }
    
    tileSize = blockSize;
    cout<<"TILE SIZE= "<<tileSize<<endl;

#pragma region //variables declaration
    float size_M = d1 * d2 * sizeof(float);
    float size_N = d2 * d1 * sizeof(float);
    float size_P = d1 * d3 * sizeof(float);
    cout<<"size of M: "<<size_M<<endl;
    cout<<"size of N: "<<size_N<<endl;
    cout<<"size of P: "<<size_P<<endl;
    
    
    dim3 dimBlock(blockSize,blockSize);
        
    dim3 dimGrid(((d1+dimBlock.x-1)/dimBlock.x),((d3+dimBlock.y-1)/dimBlock.y));
#pragma endregion

#pragma region //create and allocate matrix A, B and C
    float* M; cudaMallocManaged(&M, size_M);
    float* N; cudaMallocManaged(&N, size_N);
    float* P; cudaMallocManaged(&P, size_P);
#pragma endregion

#pragma region //init all the matrix with a passed value
   
    
    matrixInit<<<dimGrid, dimBlock>>>(M,2.0f, d1, d2);
    matrixInit<<<dimGrid, dimBlock>>>(N,3.0f, d2, d1);
    matrixInit<<<dimGrid, dimBlock>>>(P,0.0f, d1, d3);
#pragma endregion

#pragma region //multiplication operation
    matrixMulti<<<dimGrid, dimBlock>>>(M, N, P, d1, d2, d3, tileSize);

    cudaDeviceSynchronize();
#pragma endregion


cout<<"P[0] = "<<P[0]<<endl;

#pragma region //check for errors (all values should be 3000.0f)
    float maxError = 0;
    for (int i = 0; i < d1 * d3; i++)
	    maxError=fmax(maxError, fabs(P[i]-3000.0f));
    cout << "Max error: " << maxError << endl;
#pragma endregion

#pragma region //free cuda memory
    cudaFree(M); 
    cudaFree(N); 
    cudaFree(P);
#pragma region

    return 0;
}
