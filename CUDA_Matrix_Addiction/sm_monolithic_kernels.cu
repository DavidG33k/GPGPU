/**
GPGPU assignment 1: Matrix Addition in CUDA - Standard Memory/Monolithic Kernels version
    @file sm_monolithic_kernels.cu
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 13 October 2021 
*
Let A and B be the matrices of double-precision floating-point numbers to be added,
and C the resulting matrix; Let m = 2^12 and n=2^15 be their number of rows and columns, respectively.
*
Implement four versions of the matrix addition application in CUDA using:
    - Standard-Memory/Monolithic-Kernels.
*/

#include<iostream>
#include<math.h>
using namespace std;

#define M (1 << 12) //m=2^12 = 4096
#define N (1 << 15) //n=2^15 = 32768

__global__
void matrixInit(double* A, double value)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = col + row * N;

    if (col < N && row < M) {
        A[index]=value;
    }
}

__global__
void matrixAdd(double* A, double* B, double* C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = col + row * N;

    if (col < N && row < M) {
        C[index] = A[index] + B[index];
    }
}

void printMatrix(double* A)
{
    for(int i=0; i<M*N; i++)
        cout<<" "<<A[i]<<" ";
}

int main()
{
//variables declaration
    double size = M * N * sizeof(double); //expect a size in bytes
    cout<<"size: "<<size<<endl;

    dim3 dimBlock(32,32);
    dim3 dimGrid(((N+dimBlock.x-1)/dimBlock.x),((M+dimBlock.y-1)/dimBlock.y));

//create and allocate matrix A, B and C
    //allocate dynamic matrix
    double *A, *B, *C; //host matrix

    //in standard memory we have to allocate CPU
    A = (double*)malloc(size);
    B = (double*)malloc(size);
    C = (double*)malloc(size);

    double *dev_A, *dev_B, *dev_C; //device matrix

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

//init all the matrix with a passed value
    matrixInit<<<dimGrid, dimBlock>>>(dev_A,1.0);
    matrixInit<<<dimGrid, dimBlock>>>(dev_B,2.0);
    matrixInit<<<dimGrid, dimBlock>>>(dev_C,0.0);
    cout<<endl<<"M-init done"<<endl;
 
//addiction operation and print results
    cout<<endl<<"add starts"<<endl;
    matrixAdd<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);
    cout<<endl<<"add ends"<<endl;

    // cout<<endl<<"synch starts"<<endl;
    // cudaDeviceSynchronize();
    // cout<<endl<<"synch ends"<<endl;

    cout<<endl<<"final copy starts"<<endl;
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);
    cout<<endl<<"final copy ends"<<endl;
//printing resulting matrix C
    cout<<endl<<"PRINT C final"<<endl;
    //printMatrix(C);

// Check for errors (all values should be 3.0f)	
    float maxError = 0;
    for (int i = 0; i < M * N; i++)
	maxError=fmax(maxError, fabs(C[i]-3.0f));
    cout << endl << "Max error: " << maxError << endl;

//free cuda memory
    cudaFree(dev_A); 
    cudaFree(dev_B); 
    cudaFree(dev_C);
    
    return 0;
}