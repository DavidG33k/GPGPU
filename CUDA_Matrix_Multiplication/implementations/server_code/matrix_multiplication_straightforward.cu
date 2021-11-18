#include<algorithm>
#include<iostream>
#include<math.h>
#include<time.h>
using namespace std;

#define d1 500
#define d2 2000
#define d3 500

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


__global__ void matrixMulti(float* M, float* N, float* P)
{
    // Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    // if(Row<d1 && Col<d3)
    // {
    //     float pValue=0;
    //     for(int k=0; k<d2; ++k)
    //         pValue += M[Row*d2 + k] * N[k*d2 + Col]; 
    //     P[Row*d1 + Col] = pValue;
    // }

    int stride_i = blockDim.y * gridDim.y;
    int stride_j = blockDim.x * gridDim.x;

    for (int i = Row; i < d1; i += stride_i)
    {
        for (int j = Col; j < d3; j += stride_j)
        {
            float pValue = 0;

            // each thread computes one element of the block sub-matrix
            for (int k = 0; k < d2; ++k) {
                pValue += M[i*d2+k] * N[k*d2+j];
            }
            P[i*d1+j] = pValue;
        }
    }
}


int main(int argc, char* argv[])
{
   
#pragma region //managing argv && argc && time
    clock_t start, end;

	int blockSize;

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
    float size_M = d1 * d2 * sizeof(float);
    float size_N = d2 * d1 * sizeof(float);
    float size_P = d1 * d3 * sizeof(float);
    
    dim3 dimBlock(blockSize,blockSize);
        
    dim3 dimGridM(ceil(d2/dimBlock.x), ceil(d1/dimBlock.y));  //d1*d2 500x2000
    dim3 dimGridN(ceil(d3/dimBlock.x), ceil(d2/dimBlock.y));  //d2*d3 2000x500
    dim3 dimGridP(ceil(d3/dimBlock.x), ceil(d1/dimBlock.y));  //ceil works for float numbers
#pragma endregion

#pragma region //create and allocate matrix A, B and C
    float* M; cudaMallocManaged(&M, size_M);
    float* N; cudaMallocManaged(&N, size_N);
    float* P; cudaMallocManaged(&P, size_P);
#pragma endregion

#pragma region //init all the matrix with a passed value
    matrixInit<<<dimGridM, dimBlock>>>(M, 2.0f, d1, d2);
    matrixInit<<<dimGridN, dimBlock>>>(N, 3.0f, d2, d3);
    matrixInit<<<dimGridP, dimBlock>>>(P, 0.0f, d1, d3);
    
    cudaDeviceSynchronize();
#pragma endregion

cout<<"m[0] = "<<M[0]<<endl;
cout<<"n[0] = "<<N[0]<<endl;

#pragma region //multiplication operation
    matrixMulti<<<dimGridP, dimBlock>>>(M, N, P);

    cudaDeviceSynchronize();
#pragma endregion

#pragma region //check for errors (all values should be 3000.0f)
    cout<<"M[0] = "<<M[0]<<endl;
    cout<<"M[last_position] = "<<M[(int)(d1*d2-1)]<<endl;
    cout<<"N[0] = "<<N[0]<<endl;
    cout<<"N[last_position] = "<<N[(int)(d2*d3-1)]<<endl;
    cout<<"P[0] = "<<P[0]<<endl;
    cout<<"P[last_position] = "<<P[(int)(d1*d3-1)]<<endl;

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

    end=clock();
    cout << "Exe time: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;
#pragma region

    return 0;
}
