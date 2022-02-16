#include<iostream>
#include<math.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define dim 100000000
#define block_size 8


__global__
void matrixInit(int* A, int size) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
        A[i] = i;
}

__global__
void matrixInitC(int* A, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        A[i] = 0;
}



__device__ inline
void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0; //index into A
    int j = 0; //index into B
    int k = 0; //index into C

    while ((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        }
        else {
            C[k++] = B[j++];
        }
    }

    if (i == m) {
        for (; j < n; j++) {
            C[k++] = B[j];
        }
    }
    else {
        for (; i < m; i++) {
            C[k++] = A[i];
        }
    }
}

__device__ inline
int co_rank(int k, int* A, int m, int* B, int n) {
    int i = k < m ? k : m; //i = min(k,m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n; //i_low = max(0, k-n)
    int j_low = 0 > (k - m) ? 0 : k - m; //i_low = max(0, k-m)
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ((i - i_low + 1) >> 1); // ceil(i-i_low)/2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else {
            active = false;
        }
    }
    return i;
}


__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int k_curr = tid * ceilf((m + n) / (blockDim.x * gridDim.x));
    int k_next = ((tid + 1) * ceilf((m + n) / (blockDim.x * gridDim.x))) < (m + n) ? ((tid + 1) * ceilf((m + n) / (blockDim.x * gridDim.x))) : (m + n);

    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}








int main()
{
#pragma region //variables declaration
    //int block_size = 8;
    dim3 dimBlock(block_size, 1);
    dim3 dimGrid((((dim*2) + dimBlock.x - 1) / dimBlock.x));
    int size = dim * sizeof(int);
    int size_c = (2 * dim) * sizeof(int);
#pragma endregion

#pragma region //create and allocate matrix A, B and C
    //allocate dynamic matrix
    int* A,* B, * C; //host matrix

    //in standard memory we have to allocate CPU
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size_c);

    int* dev_A, * dev_B, * dev_C; //device matrix

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size_c);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);
#pragma endregion

#pragma region //init all the matrix with a passed value
    matrixInit <<< dimGrid, dimBlock >>> (dev_A, dim);
    matrixInit <<< dimGrid, dimBlock >>> (dev_B, dim);
    matrixInitC <<< dimGrid, dimBlock >>> (dev_C, dim*2);
   
    
#pragma endregion

#pragma region 
    merge_basic_kernel << <dimGrid, dimBlock >> > (dev_A, dim, dev_B, dim, dev_C);

    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_C, size_c, cudaMemcpyDeviceToHost);

    //printing resulting matrix C
    //cout << endl << "MatrixC final" << endl;
    //printMatrix(C);
#pragma endregion

#pragma region //errors check	
    bool sorted = true;

    for (int i=0; i<(dim*2)-1; i++) 
        if (C[i] > C[i+1]) {
            sorted = false;
            break;
        }
    
    if(sorted)
        cout << "Sorted!\n";
    else 
        cout << "Error!\n";

#pragma endregion
    
#pragma region //free cuda memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
#pragma endregion

    return 0;
}