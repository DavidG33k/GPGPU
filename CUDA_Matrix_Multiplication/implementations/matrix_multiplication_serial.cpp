/**
GPGPU assignment 2: Matrix Multiplication in CUDA
    @file matrix_multiplication_serial.cpp
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 03 November 2021 
*A serial implementation of the matrix multiplication algorithm in C/C++.
 - dims of M = 2000x500
 - dims of N = 500x2000
*/

#include<iostream>
#include<math.h>
#include<time.h>
using namespace std;

#define d1 2000
#define d2 500
#define d3 2000
#define dim 2000*500

void matrixInit(float** A, int m, int n, float value)
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            A[i][j] = value;
}

void matrixMulti(float** M, float** N, float** P)
{
    for(int i=0; i<d1; i++)
        for(int j=0; j<d3; j++)
        {
           float val=0.0;
            for(int k=0; k<d2; k++)
                val += M[i][k] * N[k][j]; 
            P[i][j]=val;
        }  
}

int main()
{
#pragma region //create and allocate matrix M, N and P
    clock_t start_tot, end_tot;

    float **M, **N, **P;

    M = (float**)malloc(d1*sizeof(float));
    for(int i=0; i<d1; ++i)
        M[i]=(float*)malloc(d2*sizeof(float));

    N = (float**)malloc(d2*sizeof(float));
    for(int i=0; i<d2; ++i)
        N[i]=(float*)malloc(d1*sizeof(float));

    P = (float**)malloc(d1*sizeof(float));
    for(int i=0; i<d1; ++i)
        P[i]=(float*)malloc(d1*sizeof(float));

    //tracking the time
    clock_t start, end;
    start_tot=clock();
#pragma endregion

#pragma region //init all the matrix with a passed value
    start=clock();
    matrixInit(M,d1,d2,2.0f);
    matrixInit(N,d2,d1,3.0f);
    matrixInit(P,d1,d1,0.0f);
    end=clock();
    cout << "Init time*3: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;
#pragma endregion

#pragma region //Multiplication operation and print results
    //tracking Multiplication time
    start=clock();
    matrixMulti(M,N,P);
    end=clock();

    cout.precision(100);
    cout << "Multiplication time: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;

#pragma endregion

#pragma region //check for errors (all values should be 3000f)
    cout<<P[0][0]<<endl;
    float maxError=0;
    for(int i=0; i<d1; i++)
	    for(int j=0; j<d1; j++)
	        maxError=fmax(maxError, fabs(P[i][j] - 3000.0f));
    cout << "Max error: " << maxError << endl;
#pragma endregion

#pragma region //delete matrix
    free(M);
    free(N);
    free(P);
#pragma endregion

    end_tot=clock();
    cout << "Exe time: "<<(((double)(end_tot-start_tot))/CLOCKS_PER_SEC)<<" sec"<<endl;

    return 0;
}
