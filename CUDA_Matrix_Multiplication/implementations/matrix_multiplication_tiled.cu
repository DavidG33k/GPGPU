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
using namespace std;

const int d1 = 2000;
const int d2 = 500;
const int d3 = 2000;
const int TILE = 16;
__global__
void matrixInit(double* A, double value, int raw, int col)
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
void matrixMulti(float* dev_M, float* dev_N, float* dev_P, unsigned j, unsigned k, unsigned l)
{
     __shared__
     float Mds[TILE][TILE];
     __shared__
     float Nds[TILE][TILE];
     
     int tc = threadIdx.x;
     int tr = threadIdx.y;
     int Row = blockIdx.y * TILE + tr;
     int Col = blockIdx.x * TILE + tc;

     Pvalue = 0;
     for(int ph = 0; ph < k/TILE; ++ph)
     {
          if((Row < j) && (ph * TILE + tc) < k)
               Mds[tr][tc] = dev_M[Row * k + ph * TILE +tc];
          else
               Mds[tr][tc] = 0;
          if((ph * TILE +tr) < k && Col < l)
               Nds[tr][tc] = dev_N[ph * TILE + tc];
          else
               Nds[tr][tc] = 0;
          __syncthreads();
          for(int i = 0; i < TILE; ++i)
               Pvalue += Mds[tr][i] * Nds[i][tc];
          __syncthreads();
     }
     if((Row < j) && (Col < l))
          dev_P[Row * l + Col] = Pvalue;
}

int main()
{
#pragma region //variables declaration
    float size_M = d1 * d2 * sizeof(float);
    float size_N = d2 * d1 * sizeof(float);
    float size_P = d3 * d3 * sizeof(float);
    cout<<"size of M: "<<size_M<<endl;
    cout<<"size of N: "<<size_N<<endl;
    cout<<"size of P: "<<size_P<<endl;
    
    dim3 dimBlock(32,32);
    dim3 dimGrid_M(((d2+dimBlock.x-1)/dimBlock.x),((d1+dimBlock.y-1)/dimBlock.y));
    dim3 dimGrid_N(((d1+dimBlock.x-1)/dimBlock.x),((d2+dimBlock.y-1)/dimBlock.y));
    dim3 dimGrid_P(((d3+dimBlock.x-1)/dimBlock.x),((d3+dimBlock.y-1)/dimBlock.y));
#pragma endregion

#pragma region //create and allocate matrix M, N and P
    //allocate dynamic matrix
    double *M, *N, *P; //host matrix

    //in standard memory we have to allocate CPU
    M = (float*)malloc(size_M);
    N = (float*)malloc(size_N);
    P = (float*)malloc(size_P);

    double *dev_M, *dev_N, *dev_P; //device matrix

    cudaMalloc((void**)&dev_M, size_M);
    cudaMalloc((void**)&dev_N, size_N);
    cudaMalloc((void**)&dev_P, size_P);

    cudaMemcpy(dev_M, M, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N, N, size_N, cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_P, P, size_P, cudaMemcpyHostToDevice); forse va chiamato cudamemcpy anche per P
#pragma endregion

#pragma region //init all the matrix with a passed value
    matrixInit<<<dimGrid_M, dimBlock>>>(dev_M,2.0, d1, d2);
    matrixInit<<<dimGrid_N, dimBlock>>>(dev_N,3.0, d2, d1);
    matrixInit<<<dimGrid_P, dimBlock>>>(dev_P,0.0, d3, d3);
#pragma endregion

#pragma region //multiplication operation
    //matrixMulti<<<XXX, YYY>>>(dev_M, dev_N, dev_P, j, k, l); non è ancora chiaro come chiamare correttamente il metodo e cosa siano j, k, j.

    cudaDeviceSynchronize();

    //cudaMemcpy(P, dev_P, size_P, cudaMemcpyDeviceToHost); probabilmetne metodo inutile
#pragma endregion

#pragma region //check for errors (all values should be 3.0f)
    float maxError = 0;
    for (int i = 0; i < M * N; i++)
	    maxError=fmax(maxError, fabs(C[i]-3.0f));
    cout << "Max error: " << maxError << endl;
#pragma endregion

#pragma region //check for errors (all values should be 3.0f)
    float maxError = 0;
    for (int i = 0; i < M * N; i++)
	    maxError=fmax(maxError, fabs(C[i]-12000.0f));
    cout << "Max error: " << maxError << endl;
#pragma endregion

#pragma region //free cuda memory
    cudaFree(dev_M); 
    cudaFree(dev_N); 
    cudaFree(dev_P);
#pragma region

    return 0;
}