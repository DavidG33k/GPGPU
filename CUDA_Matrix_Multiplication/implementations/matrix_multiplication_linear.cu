#include <algorithm>
#include <iostream>
#include <math.h>
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


__global__ void matrixMulti(float* M, float* N, float* P) {
// Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((Row < d1) && (Col < d3)) {
        float Pvalue = 0;
    // each thread computes one element of the block sub-matrix
    for (int k = 0; k < d2; ++k) {
    Pvalue += M[Row*d2+k]*N[k*d2+Col];
    }
    P[Row*d1+Col] = Pvalue;
    }
}



int main(int argc, char* argv[])
{
   
#pragma region //managing argv && argc

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
    cout<<"qui";
#pragma endregion

#pragma region //init all the matrix with a passed value
    matrixInit<<<dimGrid, dimBlock>>>(M, 2.0f, d1, d2);
    matrixInit<<<dimGrid, dimBlock>>>(N, 3.0f, d2, d1);
    matrixInit<<<dimGrid, dimBlock>>>(P, 0.0f, d1, d3);
    
    cudaDeviceSynchronize();
#pragma endregion

cout<<"m[0] = "<<M[0]<<endl;
cout<<"n[0] = "<<N[0]<<endl;

#pragma region //multiplication operation
    matrixMulti<<<dimGrid, dimBlock>>>(M, N, P);

    cudaDeviceSynchronize();
#pragma endregion

#pragma region //check for errors (all values should be 3000.0f)
    cout<<"P[0] = "<<P[0]<<endl;
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
