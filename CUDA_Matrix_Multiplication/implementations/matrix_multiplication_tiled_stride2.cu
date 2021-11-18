/**
GPGPU assignment 2: Matrix Multiplication in CUDA
    @file matrix_multiplication_serial.cpp
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 03 November 2021 
*A tiled implementation of the matrix multiplication algorithm in CUDA
    using the Unified Memory and the Grid-Stride Loop models.
 - dims of M = 2000x500
 - dims of N = 500x2000
*/

#include<algorithm>
#include<iostream>
#include<math.h>
#include<stdlib.h>
using namespace std;

#define TILE 8


__global__
void matrixInit(float* A, float value, int row, int col)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;

  for (int i = index_x; i < row; i += stride_x)
    for (int j = index_y; j < col; j += stride_y)
        A[j*row+i]=value;
}

__global__ void matrixMulTiledGrid(float* M, float* N, float* P, int rows_M, int cols_M, int rows_N, int cols_N, int rows_P, int cols_P)
{
    float pValue = 0;

    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    int stride_i = blockDim.y * gridDim.y;
    int stride_j = blockDim.x * gridDim.x;
    __shared__ float Ms[TILE][TILE];
    __shared__ float Ns[TILE][TILE];
    
    for (int ph = 0; ph < (TILE + cols_M - 1)/TILE; ph++)
    {
        Ms[threadIdx.y][threadIdx.x] = 0.0;
        Ns[threadIdx.y][threadIdx.x] = 0.0;

        for(int i = row; i < rows_M && ph*TILE + threadIdx.x < cols_M; i+=stride_i)
            Ms[threadIdx.y][threadIdx.x] = M[i*cols_M + ph*TILE + threadIdx.x]; //coalesced
        
        for(int j = col; j < cols_N && ph*TILE + threadIdx.y < rows_N; j+= stride_j)            
            Ns[threadIdx.y][threadIdx.x] = N[(ph*TILE + threadIdx.y)*cols_N + j]; //coalesced

        __syncthreads();

        for (int n = 0; n < TILE; n++)
        pValue += Ms[threadIdx.y][n] * Ns[n][threadIdx.x];

        __syncthreads();
    }
    P[((blockIdx.y * blockDim.y + threadIdx.y)*cols_P) + (blockIdx.x * blockDim.x)+ threadIdx.x] = pValue;
            
}


int main(int argc, char* argv[])
{
   
#pragma region //managing argv & argc & time
    clock_t start, end;

	int blockSize;

    float rows_M = 2000;
    float cols_M = 500;
    float rows_N = 500;
    float cols_N = 2000;

    //rows_P is equal to rows_M
    //cols_P is equal to cols_N

    if(argc != 2){
    	cout<<"No Block Size Declared!"<<endl;
    	return 0;
    }
    
    blockSize = atoi(argv[1]);
    
    if(blockSize!=8 && blockSize!=16 && blockSize!=32){
    	cout<<"Invalid Block Size!"<<endl;
    	return 0;
    }
    
#pragma endregion

#pragma region //variables declaration
    start=clock();

    float size_M = rows_M * cols_M * sizeof(float);
    float size_N = rows_N * cols_N * sizeof(float);
    float size_P = rows_M * cols_N * sizeof(float);
    
    dim3 dimBlock(blockSize,blockSize);
    
    dim3 dimGridM(ceil(cols_M/dimBlock.x), ceil(rows_M/dimBlock.y));  //d1*d2 500x2000
    dim3 dimGridN(ceil(cols_N/dimBlock.x), ceil(rows_N/dimBlock.y));  //d2*d3 2000x500
    dim3 dimGridP(ceil(cols_N/dimBlock.x), ceil(rows_M/dimBlock.y));  //il ceil funzion per variabili float

#pragma endregion

#pragma region //create and allocate matrix A, B and C
    float* M; cudaMallocManaged(&M, size_M);
    float* N; cudaMallocManaged(&N, size_N);
    float* P; cudaMallocManaged(&P, size_P);
#pragma endregion

#pragma region //init all the matrix with a passed value
    matrixInit<<<dimGridM, dimBlock>>>(M, 2.0f, rows_M, cols_M);
    matrixInit<<<dimGridN, dimBlock>>>(N, 3.0f, rows_N, cols_N);
    matrixInit<<<dimGridP, dimBlock>>>(P, 0.0f, rows_M, cols_N);
#pragma endregion

#pragma region //multiplication operation
    matrixMulTiledGrid<<<dimGridP, dimBlock>>>(M, N, P, rows_M, cols_M, rows_N, cols_N, rows_M, cols_N);

    cudaDeviceSynchronize();
#pragma endregion

#pragma region //check for errors (all values should be 3000.0f)
    cout<<"M[0] = "<<M[0]<<endl;
    cout<<"M[last_position] = "<<M[(int)(rows_M*cols_M-1)]<<endl;
    cout<<"N[0] = "<<N[0]<<endl;
    cout<<"N[last_position] = "<<N[(int)(rows_N*cols_N-1)]<<endl;
    cout<<"P[0] = "<<P[0]<<endl;
    cout<<"P[last_position] = "<<P[(int)(rows_M*cols_N-1)]<<endl;

    // int cont=0;
    // for (int i = 0; i < rows_M * cols_N; i++)
    // {
    //     if(P[i]!=12000.0f)
    //         cont++;
    // }
    // cout<<"elementi mancanti: "<<cont<<endl;

    float maxError = 0;
    for (int i = 0; i < rows_M * cols_N; i++)
	    maxError=fmax(maxError, fabs(P[i]-3000.0f));
    cout << "Max error: " << maxError << endl;
#pragma endregion

#pragma region //free cuda memory
    cudaFree(M); 
    cudaFree(N); 
    cudaFree(P);

#pragma region
    end=clock();
    cout << "Exe time: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;
    
    return 0;
}
