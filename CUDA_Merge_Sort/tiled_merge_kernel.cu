#include<iostream>
#include<math.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define dim 100000000
#define block_size 8
#define TILE 8



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



__global__
void matrixInit(int* A, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        A[i] = i;
}

__global__
void matrixInitC(int* A, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        A[i] = 0;
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


__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C) {
    /* shared memory allocation */
    __shared__ int shareAB[TILE*2];
    
    int* A_S = &shareAB[0]; //shareA is first half of shareAB
    int* B_S = &shareAB[TILE]; //ShareB is second half of ShareAB

    int C_curr = blockIdx.x * ceilf((m + n) / gridDim.x); // starting point of the C subarray for current block
    int C_next = ((blockIdx.x + 1) * ceilf((m + n) / gridDim.x)) < (m + n) ? ((blockIdx.x + 1) * ceilf((m + n) / gridDim.x)) : (m + n); // starting point for next block



    if (threadIdx.x == 0)
    {
        A_S[0] = co_rank(C_curr, A, m, B, n); // Make the block-level co-rank values visible to
        A_S[1] = co_rank(C_next, A, m, B, n); // other threads in the block
    }

    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    
    __syncthreads();

    int counter = 0; //iteration counter
    
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    
    int total_iteration = ceilf((C_length) / TILE); //total iteration
    
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while (counter < total_iteration) {
        /* loading tile-size A and B elements into shared memory */
        for (int i = 0; i < TILE; i += blockDim.x)
        {
            if (i + threadIdx.x < A_length - A_consumed)
            {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }

        for (int i = 0; i < TILE; i += blockDim.x)
        {
            if (i + threadIdx.x < B_length - B_consumed)
            {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }

        __syncthreads();

        int c_curr = threadIdx.x * (TILE / blockDim.x);
        int c_next = (threadIdx.x + 1) * (TILE / blockDim.x);

        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        /* find co-rank for c_curr and c_next */
        int a_min = TILE < A_length - A_consumed ? TILE : A_length - A_consumed;
        int b_min = TILE < B_length - B_consumed ? TILE : B_length - B_consumed;
        
        int a_curr = co_rank(c_curr, A_S, a_min, B_S, b_min);

        int b_curr = c_curr - a_curr;
        
        a_min = TILE < A_length - A_consumed ? TILE : A_length - A_consumed;
        b_min = TILE < B_length - B_consumed ? TILE : B_length - B_consumed;

        int a_next = co_rank(c_next, A_S, a_min, B_S, b_min);

        int b_next = c_next - a_next;

        /* All threads call the sequential merge function */
        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr, C + C_curr + C_completed + c_curr);

        /* Update the A and B elements that have been consumed thus far */
        counter++;
        C_completed += TILE;
        A_consumed += co_rank(TILE, A_S, TILE, B_S, TILE);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}



int main()
{
#pragma region //variables declaration
    
    dim3 dimBlock(block_size, 1);
    dim3 dimGrid((((dim * 2) + dimBlock.x - 1) / dimBlock.x));
    int size = dim * sizeof(int);
    int size_c = (2 * dim) * sizeof(int);
#pragma endregion

#pragma region //create and allocate matrix A, B and C
    //allocate dynamic vectors
    int* A, * B, * C; //host vectors

    //in standard memory we have to allocate CPU
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size_c);

    int* dev_A, * dev_B, * dev_C; //device vectors

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size_c);

    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);
#pragma endregion

#pragma region //init all the matrix with a passed value
    matrixInit <<< dimGrid, dimBlock >>> (dev_A, dim);
    matrixInit <<< dimGrid, dimBlock >>> (dev_B, dim);
    matrixInitC <<< dimGrid, dimBlock >>> (dev_C, dim * 2);
#pragma endregion

#pragma region 
    merge_tiled_kernel << <dimGrid, dimBlock >> > (dev_A, dim, dev_B, dim, dev_C);

    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_C, size_c, cudaMemcpyDeviceToHost);

#pragma endregion

#pragma region //errors checks
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