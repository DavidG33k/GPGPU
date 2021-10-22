/**
GPGPU assignment 1: Matrix Addition in CUDA
    @file matrix_addiction_serial.cpp
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 13 October 2021 
*
Let A and B be the matrices of double-precision floating-point numbers to be added,
and C the resulting matrix; Let m = 2^12 and n=2^16 be their number of rows and columns, respectively.
*
Implement a serial version of the matrix addition application in C (for comparison).
Reduce m and n.
*/
#include<iostream>
#include<math.h>
#include<time.h>
using namespace std;

const int m = pow(2,12);
const int n = pow(2,14);

void matrixAlloc(double** A)
{
    for(int i=0; i<m; i++)
        A[i]=new double[n];
}

void matrixInit(double** A, double value)
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            A[i][j] = value;
}

void matrixAdd(double** A, double** B, double** C)
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void printMatrix(double** A)
{
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
            cout<<" "<<A[i][j]<<" ";
        cout<<endl;
    }
}

void deleteMatrix(double** A)
{
    for(int i=0; i<m; i++)
        delete[] A[i];
    delete[] A;
}

int main()
{
    #pragma region //create and allocate matrix A, B and C
    double **A = new double*[m];
    double **B = new double*[m];
    double **C = new double*[m];

    matrixAlloc(A);
    matrixAlloc(B);
    matrixAlloc(C);

    //tracking the time
    clock_t tInit, tAdd;
    #pragma endregion

    #pragma region //init all the matrix with a passed value
    //tracking init time
    tInit = clock();
    matrixInit(A,1);
    matrixInit(B,2);
    matrixInit(C,0);

    cout.precision(10);
    cout << "Init time: "<<tInit<<" clocks in "<<((double)(clock() - tInit))/CLOCKS_PER_SEC<<" sec"<<endl;
    #pragma endregion

    #pragma region //addiction operation and print results
    //tracking addiction time
    tAdd = clock();
    matrixAdd(A,B,C);

    cout.precision(10);
    cout << "Add time: "<<tAdd<<" clocks in "<<((double)(clock() - tAdd))/CLOCKS_PER_SEC<<" sec"<<endl;

    //printing resulting matrix C and resulting time
    cout<<endl<<"PRINT C final"<<endl;
    cout.precision(10);
    cout << "Total time for size="<<m*n<<" : "<<tInit+tAdd<<" clocks in "<<((double)(clock() - (tInit + tAdd)))/CLOCKS_PER_SEC<<" sec"<<endl;
    //printMatrix(C);
    #pragma endregion

    #pragma region //check for errors (all values should be 3.0f)
    float maxError = 0;
    for (int i = 0; i < m; i++)
	for(int j=0; j<n; j++)
		maxError=fmax(maxError, fabs(C[i][j]-3.0f));
    cout << "Max error: " << maxError << endl;
    #pragma endregion

    #pragma region //delete matrix
    deleteMatrix(A);
    deleteMatrix(B);
    deleteMatrix(C);
    #pragma endregion
    
    return 0;
}