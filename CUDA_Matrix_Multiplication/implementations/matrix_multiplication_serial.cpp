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
#include<assert.h>
using namespace std;

#define rowsM 2000
#define colsM 500
#define rowsN 500
#define colsN 2000
#define dim 2000*500

void matrixInit(float** A, int m, int n, float value)
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            A[i][j] = value;
}

void matrixMulti(float** M, float** N, float** P)
{
    for(int i=0; i<rowsM; i++)
        for(int j=0; j<colsN; j++)
        {
           float val=0.0;
            for(int k=0; k<colsM; k++)
                val += M[i][k] * N[k][j]; 
            P[i][j]=val;
        }  
}

int main()
{
#pragma region //create and allocate matrix M, N and P
    clock_t start_tot, end_tot;

    assert(colsM == rowsN);

    float **M, **N, **P;

    M = (float**)malloc(rowsM*sizeof(float));
    for(int i=0; i<rowsM; ++i)
        M[i]=(float*)malloc(colsM*sizeof(float));

    N = (float**)malloc(colsM*sizeof(float));
    for(int i=0; i<colsM; ++i)
        N[i]=(float*)malloc(rowsM*sizeof(float));

    P = (float**)malloc(rowsM*sizeof(float));
    for(int i=0; i<rowsM; ++i)
        P[i]=(float*)malloc(rowsM*sizeof(float));

    //tracking the time
    clock_t start, end;
    start_tot=clock();
#pragma endregion

#pragma region //init all the matrix with a passed value
    start=clock();
    matrixInit(M,rowsM,colsM,2.0f);
    matrixInit(N,colsM,rowsM,3.0f);
    matrixInit(P,rowsM,rowsM,0.0f);
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

#pragma region //check for errors (all values should be eValue)

    float eValue = 3000.0f;

    cout<<"P[0] = "<<P[0]<<endl;
    cout<<"P[last] = "<<P[rowsM*colsN-1]<<endl;

    float maxError=0;
    for(int i=0; i<rowsM; i++)
	    for(int j=0; j<rowsM; j++)
	        maxError=fmax(maxError, fabs(P[i][j] - eValue));
    cout << "Max error: " << maxError << endl;
#pragma endregion

#pragma region //delete matrix and printind execution time
    free(M);
    free(N);
    free(P);

    end_tot=clock();
    cout << "Exe time: "<<(((double)(end_tot-start_tot))/CLOCKS_PER_SEC)<<" sec"<<endl;
#pragma endregion   

    return 0;
}
