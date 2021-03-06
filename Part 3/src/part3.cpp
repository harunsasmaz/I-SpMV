#include <bits/stdc++.h>
#include "../include/common.h"
#include "../include/matrix.h"
#include "../include/mmio.h"
#include <mpi.h>

using namespace std;

#define MASTER 0

MPI_Datatype info_type;

// create an MPI datatype for the struct below.
void create_mpi_datatypes(MPI_Datatype *info_type) {
    MPI_Datatype oldtypes[2];
    MPI_Aint offsets[2], extent;
    int blockcounts[2];

    offsets[0] = 0;
    oldtypes[0] = MPI_INT;
    blockcounts[0] = 4;

    MPI_Type_create_struct(1, blockcounts, offsets, oldtypes, info_type);
    MPI_Type_commit(info_type);
}

// each process needs this information, pack them in a struct
typedef struct info_t {
    int nrows;
    int steps;
    int e_count;
    int e_displs;
} info_t;

void equal_partition(int nprocs, int nrows, int nnz, int* rows, int* counts, int* displs)
{   
    // count nnz for each row.
    int* row_elements = (int*)malloc(sizeof(int) * nrows);
    for(int i = 0; i < nrows; i++){
        row_elements[i] = rows[i + 1] - rows[i];
    }

    // first find a lower bound nnz for processes
    // by applying binary search.
    int low = 0, high = nnz;
    while(low < high)
    {   
        // the range of searching
        int mid = (low + high) / 2;

        // calculate how many process are needed so that each
        // process receives up to nnz elements 
        int sum = 0, need = 0;
        for(int i = 0; i < nrows; i++){
            if (sum + row_elements[i] > mid) {
                sum = row_elements[i];
                need++;
            } else {
                sum += row_elements[i];
            }
        }

        // update the searching range.
        if (need < nprocs)
            high = mid;
        else
            low = mid + 1;
    }

    // split rows by processes according to lower bound.
    int sum = 0, counter = 0;
    for(int i = 0; i < nrows; i++){
        if(sum + row_elements[i] > low){
            sum = row_elements[i];
            counter++;
            counts[counter] = 1;
            displs[counter] = i;
        } else {
            sum += row_elements[i];
            counts[counter]++;
        }
    }
}


int main(int argc, char* argv[]){

    int rank, nprocs, nrows, time_steps;
    int elm_count, elm_displs, row_count;
    int *rowptr, *colptr;
    double *valptr, *local_res, *final_res, *rhs;

    info_t* infos;
    info_t my_info;

    csr_matrix matrix;
    string matrix_name;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    create_mpi_datatypes(&info_type);

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
        rhs = (double *)malloc(sizeof(double) * nrows);
        for(int i = 0; i < nrows; i++)
            rhs[i] = (double) 1.0/nrows;
        
        // initially set to 0 for safe parameter passing.
        rowDispls[0] = 0;
        rowCounts[0] = 0;
        // partition rows with load balancing.
        equal_partition(nprocs, nrows, matrix.nnz, matrix.csrRowPtr, rowCounts, rowDispls);
        
        // keep which process gets how many elements from which index and fill needed information.
        infos = (info_t*)malloc(sizeof(info_t) * nprocs);
        for(int i = 0; i < nprocs; i++){
            infos[i].nrows = nrows;
            infos[i].steps = time_steps;
            infos[i].e_count = eCounts[i] = matrix.csrRowPtr[rowDispls[i] + rowCounts[i]] - matrix.csrRowPtr[rowDispls[i]];
            infos[i].e_displs = eDispls[i] = matrix.csrRowPtr[rowDispls[i]];
        }

    }

    // scatter needed information to all processes
    MPI_Scatter(infos, 1, info_type, &my_info, 1, info_type, MASTER, MPI_COMM_WORLD);
    nrows = my_info.nrows;
    time_steps = my_info.steps;
    elm_count = my_info.e_count;
    elm_displs = my_info.e_displs;

    // broadcast row counts and displs.
    MPI_Bcast(rowCounts, nprocs, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(rowDispls, nprocs, MPI_INT, MASTER, MPI_COMM_WORLD);

    // each process knows how many rows it gets.
    row_count = rowCounts[rank];
    // only master allocated rhs before, the rest should allocate too.
    if(rank != MASTER)
        rhs = (double*)malloc(sizeof(double) * nrows);
        
    // broadcast rhs vector to all.
    MPI_Bcast(rhs, nrows, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    
    // send row ptr to each process
    rowptr = (int*)malloc(sizeof(int) * (row_count + 1));
    MPI_Scatterv(matrix.csrRowPtr, rowCounts, rowDispls, MPI_INT, 
                    rowptr, row_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    // to avoid collision in scatter, last element is added seperately.
    rowptr[row_count] = elm_count + rowptr[0];
    
    // send col indexes to each process
    colptr = (int*)malloc(sizeof(int) * elm_count);
    MPI_Scatterv(matrix.csrColIdx, eCounts, eDispls, MPI_INT, 
                    colptr, elm_count, MPI_INT, MASTER, MPI_COMM_WORLD);
                    
    // send val indexes to each process
    valptr = (double*)malloc(sizeof(double) * elm_count);
    MPI_Scatterv(matrix.csrVal, eCounts, eDispls, MPI_DOUBLE, 
                    valptr, elm_count, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // allocate local and final result vectors.
    local_res = (double*)malloc(sizeof(double) * nrows);
    final_res = (double*)malloc(sizeof(double) * nrows);
    double a; // temp result of multiplication loop.
    int x,y; // start and end indexes of multiplication loop.

    clock_t start = clock();
    for(int k = 0; k < time_steps; k++) 
    {   
        for(int i = 0; i < row_count; i++)
        {
            a = 0.0;
            x = rowptr[i];
            y = rowptr[i+1];
            for(int j = x; j < y; j++)
            {   
                a += valptr[j - elm_displs] * rhs[colptr[j - elm_displs]];
            }
            local_res[i] = a;
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
        cout << "Time taken by processes is : " << fixed  
            << time_taken << setprecision(5); 
        cout << " sec " << endl;

        // for debugging the result: 
        
        // for(int i = 0; i < nrows; i++)
        //     cout << rhs[i] << endl;
    }


    free(rowptr);
    free(colptr);
    free(valptr);
    free(local_res);
    free(final_res);
    free(rhs);
    MPI_Finalize();
    return 0;
}
