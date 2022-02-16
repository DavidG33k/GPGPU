/**
Exam report: Cuda Merge Sort
    @file serial_merge_sort.cu
    @author Canonaco Martina @author Gena Davide
    @version 11 gen 2022
*/

#include <iostream>
#include <assert.h>
#include <vector>
using namespace std;

#define DIM 1000

void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0;  //index into A
    int j = 0;  //index into B
    int k = 0;  //index into C

    // handle the start of A[] and B[]
    while((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    if (i == m) {
        //done with A[] handle remaining B[]
        for (; j < n; j++) {
            C[k++] = B[j];
        }
    } else {
        //done with B[], handle remaining A[]
        for (; i <m; i++) {
            C[k++] = A[i];
        }
    }
}

int initArray (int *array) {
    for(int i=0; i<DIM; i++)
        array[i] = i;
}

int main () {

    int A[DIM];
    int B[DIM];
    int C[DIM*2];

    initArray(A);
    initArray(B);

    merge_sequential(A, DIM, B, DIM, C);

    bool sorted = true;

    for(int i=0; i<DIM*2-1; i++)
        if(C[i] > C[i+1]) {
            sorted = false;
            break;
        }
            
    if(sorted)
        cout << "Sorted!\n";
    else 
        cout << "Error!\n";  

    return 0;
}
