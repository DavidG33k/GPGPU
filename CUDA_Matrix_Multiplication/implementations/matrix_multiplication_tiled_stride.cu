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

#define d1 2000
#define d2 500
#define d3 2000
#define TILE 8

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
void matrixMulti(float* dev_M, float* dev_N, float* dev_P)
{
     
     __shared__ float Mds[TILE][TILE];
     __shared__ float Nds[TILE][TILE];     
     
     int tc = threadIdx.x;
     int tr = threadIdx.y;
     int Row = blockIdx.y * TILE + threadIdx.y;
     int Col = blockIdx.x * TILE + threadIdx.x;

     int stride_c = blockDim.x * gridDim.x;
     int stride_r = blockDim.y * gridDim.y;

     float Pvalue = 0.0f;

     for(int ph = 0; ph < (TILE + d2 - 1)/TILE; ++ph)
     {

        Mds[tr][tc] = 0;
        Nds[tr][tc] = 0;

        for (int j = Row; j < d1; j += stride_r)
            for (int i = (ph * TILE + tc); i < d2; i += stride_c)
             {
                Mds[tr][tc] = dev_M[j * d2 + i];
             }

        for (int j = (ph * TILE + tr); j < d2; j += stride_r)
            for (int i = Col; i < d3; i += stride_c)
            { 
                Nds[tr][tc] = dev_N[j * d3 + i];
            }
            
        __syncthreads();

        for(int i = 0; i < TILE; ++i)
            Pvalue += Mds[tr][i] * Nds[i][tc];

        __syncthreads();
     }
     
     for (int j = Row; j < d1; j += stride_r)
            for (int i = Col; i < d3; i += stride_c)
                dev_P[i * d3 + j] = Pvalue;
            
}

int main(int argc, char* argv[])
{
   
#pragma region //managing argv & argc & time
    clock_t start, end;

	int blockSize;

    if(argc != 2){
    	cout<<"no Block Size declared!"<<endl;
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

    float size_M = d1 * d2 * sizeof(float);
    float size_N = d2 * d1 * sizeof(float);
    float size_P = d1 * d3 * sizeof(float);
    
    dim3 dimBlock(blockSize,blockSize);
        
    dim3 dimGridM(ceil(d2/dimBlock.x), ceil(d1/dimBlock.y));  //d1*d2 500x2000
    dim3 dimGridN(ceil(d3/dimBlock.x), ceil(d2/dimBlock.y));  //d2*d3 2000x500
    dim3 dimGridP(ceil(d3/dimBlock.x), ceil(d1/dimBlock.y));  //ceil works
#pragma endregion

#pragma region //create and allocate matrix A, B and C
    float* M; cudaMallocManaged(&M, size_M);
    float* N; cudaMallocManaged(&N, size_N);
    float* P; cudaMallocManaged(&P, size_P);
#pragma endregion

#pragma region //init all the matrix with a passed value
    matrixInit<<<dimGridM, dimBlock>>>(M, 2.0f, d1, d2);
    matrixInit<<<dimGridN, dimBlock>>>(N, 3.0f, d2, d1);
    matrixInit<<<dimGridP, dimBlock>>>(P, 0.0f, d1, d3);
#pragma endregion

#pragma region //multiplication operation
    matrixMulti<<<dimGridP, dimBlock>>>(M, N, P);

    cudaDeviceSynchronize();
#pragma endregion

#pragma region //check for errors (all values should be 3000.0f)
    cout<<"M[0] = "<<M[0]<<endl;
    cout<<"M[last_position] = "<<M[d1*d2-1]<<endl;
    cout<<"N[0] = "<<N[0]<<endl;
    cout<<"N[last_position] = "<<N[d2*d3-1]<<endl;
    cout<<"P[0] = "<<P[0]<<endl;
    cout<<"P[last_position] = "<<P[d1*d3-1]<<endl;

    int cont=0;
    for (int i = 0; i < d1 * d3; i++)
    {
        if(P[i]!=3000.0f)
            cont++;
    }
    cout<<"elementi mancanti: "<<cont<<endl;

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
    end=clock();
    cout << "Exe time: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;
    
    return 0;
}
