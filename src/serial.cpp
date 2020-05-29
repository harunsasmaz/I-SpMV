// SpMV Code
// Created: 03-12-2019
// Author: Najeeb Ahmad
// Updated: 13-05-2020
// Author: Muhammad Aditya Sasongko

#include "stdc++.h"
#include "../include/common.h"
#include "../include/matrix.h"
#include "../include/mmio.h"

using namespace std;

int main(int argc, char **argv)
{
    if(argc < 3)
    {
    	cout << "Error: Missing arguments\n";
    	cout << "Usage: " << argv[0] << " matrix.mtx\n" << " time_step_count";
    	return EXIT_FAILURE;
    }

    csr_matrix matrix;
    string matrix_name;

    printf("Reading .mtx file\n");
    int retCode = 0;
    int time_steps = atoi(argv[2]);
    matrix_name = argv[1];
    cout << matrix_name << endl;
    
    double *rhs;
    double *result;
    retCode = mm_read_unsymmetric_sparse(argv[1], &matrix.m, &matrix.n, &matrix.nnz,
					 &matrix.csrVal, &matrix.csrRowPtr, &matrix.csrColIdx);
    // Allocate vector rhs
    rhs = (double *)malloc(sizeof(double) * matrix.n);
    // Allocate vector result
    result = (double *)malloc(sizeof(double) * matrix.n);

    if(retCode == -1)
    {
      cout << "Error reading input .mtx file\n";
      return EXIT_FAILURE;
    }
    
    printf("Matrix Rows: %d\n", matrix.m);
    printf("Matrix Cols: %d\n", matrix.n);
    printf("Matrix nnz: %d\n", matrix.nnz);
    coo2csr_in(matrix.m, matrix.nnz, matrix.csrVal, matrix.csrRowPtr, matrix.csrColIdx);
    printf("Done reading file\n");

    // Initialize right-hand-side
    for(int i = 0; i < matrix.n; i++)
        rhs[i] = (double) 1.0/matrix.n;

    clock_t start, end;

    start = clock();

    for(int k = 0; k < time_steps; k++) {
    	for(int i = 0; i < matrix.m; i++)
      {
		    result[i] = 0.0;
		    for(int j = matrix.csrRowPtr[i]; j < matrix.csrRowPtr[i+1]; j++)
	  	  {
	    		result[i] += matrix.csrVal[j] * rhs[matrix.csrColIdx[j]];
	  	  }
      }

    	for(int i = 0; i < matrix.m; i++)
      {
		    rhs[i] = result[i];
      }

    }

    end = clock(); 

    double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
    cout << "Time taken by program is : " << fixed  
         << time_taken << setprecision(5); 
    cout << " sec " << endl; 

    cout << endl;
    
    return EXIT_SUCCESS;
}
