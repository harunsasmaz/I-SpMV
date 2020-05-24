#include "stdc++.h"
#include "../include/common.h"
#include "../include/matrix.h"
#include "../include/mmio.h"
#include <mpi.h>

using namespace std;
#define MASTER 0

int main(int argc, char* argv[]){

    int rank, nprocs, nrows, time_steps;
    int elm_count, my_row;
    int *rowptr, *colptr;
    double *valptr, *local_res, *final_res, *rhs;

    csr_matrix matrix;
    string matrix_name;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int eCounts[nprocs], eDispls[nprocs], rowCounts[nprocs], rowDispls[nprocs];

    if(rank == MASTER)
    {
        if(argc < 3)
        {
            cout << "Error: Missing arguments\n";
            cout << "Usage: " << argv[0] << " matrix.mtx\n" << " time_step_count";
            return EXIT_FAILURE;
        }

        printf("Reading .mtx file\n");
        int retCode = 0;
        time_steps = atoi(argv[2]);
        matrix_name = argv[1];
        cout << matrix_name << endl;
        
        retCode = mm_read_unsymmetric_sparse(argv[1], &matrix.m, &matrix.n, &matrix.nnz,
                        &matrix.csrVal, &matrix.csrRowPtr, &matrix.csrColIdx);
        
        // global total row count to be broadcasted.
        nrows = matrix.m;

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

        // Allocate and init vector rhs to be ready before brodcasting
        rhs = (double *)malloc(sizeof(double) * matrix.n);
        for(int i = 0; i < matrix.n; i++)
            rhs[i] = (double) 1.0/matrix.n;
    }

    // broadcast number of rows/cols in matrix and time steps.
    MPI_Bcast(&nrows, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&time_steps, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // calculate how many row is reserved for each process.
    int row_count = ceil((double)nrows / nprocs);
    // in case of nrows is not divisible by nprocs, last process gets the remaining rows.
    int row_count_last = nrows - ((nprocs - 1) * row_count);
    if(rank == nprocs - 1)
        row_count = row_count_last;

    // keep which process gets how many rows and its starting index.
    for(int i = 0; i < nprocs - 1; i++){
        rowCounts[i] = row_count;
        rowDispls[i] = i * row_count;
    }
    rowCounts[nprocs - 1] = row_count_last;
    rowDispls[nprocs - 1] = (nprocs - 1) * row_count;

    // calculate which process gets how many elements in colptr and valptr.
    if(rank == MASTER){
        for(int i = 0; i < nprocs; i++){
            eCounts[i] = matrix.csrRowPtr[i * row_count + rowCounts[i]] - matrix.csrRowPtr[i * row_count];
            eDispls[i] = matrix.csrRowPtr[i * row_count];
        }
    } else {
        // MASTER has already done this allocation.
        rhs = (double*)malloc(sizeof(double) * nrows);
    }

    // broadcast rhs vector to all.
    MPI_Bcast(rhs, nrows, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // send numbers of nonzero elements per process.
    MPI_Scatter(eCounts, 1, MPI_INT, &elm_count, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // send row ptr to each process
    rowptr = (int*)malloc(sizeof(int) * (row_count + 1));
    MPI_Scatterv(matrix.csrRowPtr, rowCounts, rowDispls, MPI_INT, 
                    rowptr, row_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    // to avoid collision in scatter, last element is added seperately.
    rowptr[row_count] = elm_count;

    // send col indexes to each process
    colptr = (int*)malloc(sizeof(int) * elm_count);
    MPI_Scatterv(matrix.csrColIdx, eCounts, eDispls, MPI_INT, 
                    colptr, elm_count, MPI_INT, MASTER, MPI_COMM_WORLD);

    // send val indexes to each process
    valptr = (double*)malloc(sizeof(double) * elm_count);
    MPI_Scatterv(matrix.csrVal, eCounts, eDispls, MPI_DOUBLE, 
                    valptr, elm_count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);


    // allocate local and final result vectors.
    local_res = (double*)malloc(sizeof(double) * row_count);
    final_res = (double*)malloc(sizeof(double) * nrows);

    clock_t start = clock();

    for(int k = 0; k < time_steps; k++) 
    {
        for(int i = 0; i < row_count; i++)
        {
            local_res[i] = 0.0;
            for(int j = rowptr[i]; j < rowptr[i+1]; j++)
            {
                local_res[i] += valptr[j] * rhs[colptr[j]];
            }
        }

        MPI_Allgatherv(local_res, row_count, MPI_DOUBLE, final_res, rowCounts, rowDispls, MPI_DOUBLE, MPI_COMM_WORLD);

        for(int i = 0; i < nrows; i++)
        {
            rhs[i] = final_res[i];
        }
    }

    clock_t end = clock();

    if(rank == MASTER){
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
        cout << "Time taken by program is : " << fixed  
            << time_taken << setprecision(5); 
        cout << " sec " << endl; 
    }

    MPI_Finalize();
    return 0;
}