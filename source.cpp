#include "stdc++.h"
#include "../include/common.h"
#include "../include/matrix.h"
#include "../include/mmio.h"
#include <mpi.h>

using namespace std;
#define MASTER 0
#define split "row"

MPI_Datatype mpi_info;

typedef struct {
    int N,              /* row dim of matrix */
        nnz,            /* number of non-zero elements */
        nz_proc,        /* number of matrix elements for each process */
        nz_start_idx,   /* first matrix element for each process */
        row_count,      /* number of rows to process for each process */
        row_start_idx;  /* first row processed for each process */ 
}info_t;

void make_mpi_struct(MPI_Datatype* type)
{
    MPI_Datatype oldtypes[2];
    MPI_Aint offsets[2], extent;
    int blockcounts[2];

    offsets[0] = 0;
    oldtypes[0] = MPI_INT;
    blockcounts[0] = 6;

    MPI_Type_create_struct(1, blockcounts, offsets, oldtypes, type);
    MPI_Type_commit(type);
}

void split_by_row(info_t *info, int nprocs, int* row_ptr)
{
    double chunk = ((double) info[MASTER].N) / nprocs;

    int last, counter = 0;
    for(int i = 0; i < nprocs; i++){

        info[i].nz_proc = 0;
        info[i].nz_start_idx = i;

        last = (int)((i + 1) * chunk - 1);

        info[i].row_start_idx = (int)(i * chunk);
        info[i].row_count = last - info[i].row_start_idx + 1;

        while (counter < info[0].nnz && row_ptr[i] <= last) {
            info[i].nz_proc++; 
            counter++;
        }
    }

    info[nprocs - 1].nz_proc += info[0].nnz - counter; 
}

void split_load_balanced(info_t *info, int nprocs, int* row_ptr);

double* multiply(int rank, int nprocs, info_t *all_info, int* row_ptr, 
    int* col_ptr, double* val_ptr, double* rhs)
{
    info_t info;
    double* final_res;

    int *counts, *displs;

    MPI_Scatter(all_info, 1, mpi_info, &info, 1, mpi_info, MASTER, MPI_COMM_WORLD);

    double* local_res;
    if(rank != MASTER){
        rhs = (double*)malloc(info.N * sizeof(double));
        row_ptr = (int*)malloc(info.nz_proc * sizeof(int));
        col_ptr = (int*)malloc(info.nz_proc * sizeof(int));
        val_ptr = (double*)malloc(info.nz_proc * sizeof(double));
    }

    MPI_Bcast(rhs, info.N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    if(rank == MASTER){
        counts = (int*)malloc(nprocs * sizeof(int));
        displs = (int*)malloc(nprocs * sizeof(int));
        for (int i = 0; i < nprocs; i++) {
            counts[i] = all_info[i].nz_proc;
            displs[i] = all_info[i].nz_start_idx;
        }
    }

    MPI_Scatterv(row_ptr, counts, displs, MPI_INT, row_ptr, 
                    info.nz_proc, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(col_ptr, counts, displs, MPI_INT, col_ptr, 
                    info.nz_proc, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(val_ptr, counts, displs, MPI_DOUBLE, row_ptr, 
                    info.nz_proc, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // TODO: Do the multiplication here

   
    final_res = (double*)malloc(info.N * sizeof(double));
    for(int i = 0; i < nprocs; i++){
        counts[i] = all_info[i].row_count;
        displs[i] = all_info[i].row_start_idx;
    }

    MPI_Allgatherv(local_res, info.row_count, MPI_DOUBLE, final_res, counts, 
                displs, MPI_DOUBLE, MPI_COMM_WORLD);
    

    return final_res;
    
}


int main (int argc, char* argv[])
{
    int rank, size;
    double *rhs, *result;
    info_t *all_info;
    csr_matrix matrix;
    string matrix_name;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    make_mpi_struct(&mpi_info);

    if(rank == MASTER){

        if(argc < 3)
        {
            cout << "Error: Missing arguments\n";
            cout << "Usage: " << argv[0] << " matrix.mtx\n" << " time_step_count";
            return EXIT_FAILURE;
        }

        printf("Reading .mtx file\n");
        int retCode = 0;
        int time_steps = atoi(argv[2]);
        matrix_name = argv[1];
        cout << matrix_name << endl;
        
        double *rhs;
        double *result;
        retCode = mm_read_unsymmetric_sparse(argv[1], &matrix.m, &matrix.n, &matrix.nnz,
                        &matrix.csrVal, &matrix.csrRowPtr, &matrix.csrColIdx);

        if(retCode == -1)
        {
            cout << "Error reading input .mtx file\n";
            return EXIT_FAILURE;
        }

        // Allocate vector rhs
        rhs = (double *)malloc(sizeof(double) * matrix.n);
        
        printf("Matrix Rows: %d\n", matrix.m);
        printf("Matrix Cols: %d\n", matrix.n);
        printf("Matrix nnz: %d\n", matrix.nnz);
        coo2csr_in(matrix.m, matrix.nnz, matrix.csrVal, matrix.csrRowPtr, matrix.csrColIdx);
        printf("Done reading file\n");

        all_info = (info_t*)malloc(size * sizeof(info_t));
        all_info[MASTER].N = matrix.m; // equal to number of rows.
        all_info[MASTER].nnz = matrix.nnz;

        for(int i = 1; i < size; i++)
            all_info[i] = all_info[MASTER];

        for(int i = 0; i < matrix.n; i++)
            rhs[i] = (double) 1.0/matrix.n;

        if(split == "row")
            split_by_row(all_info, size, matrix.csrRowPtr);
        else
            split_load_balanced(all_info, size, matrix.csrRowPtr);
        
    }

    result = (double *)malloc(sizeof(double) * matrix.n);

    clock_t start = clock();
    result = multiply(rank, size, all_info, matrix.csrRowPtr, matrix.csrColIdx, matrix.csrVal, rhs);
    clock_t end = clock();

    if(rank == MASTER){
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
        cout << "Time taken by program is : " << fixed << time_taken << setprecision(5); 
        cout << " sec " << endl;  
    }
    
    return EXIT_SUCCESS;

}

// ====================== BONUS ======================== // 

void split_load_balanced(info_t *all_info, int nprocs, int* row_ptr)
{
    int* nnz_per_row = (int*)calloc(all_info[MASTER].N, sizeof(int));
    for (int i = 0; i < all_info[MASTER].nnz; i++) 
    {
        nnz_per_row[row_ptr[i]]++;
    }

    int low = 0, high = all_info[MASTER].nnz;
    while(low < high)
    {
        int mid = (low + high) / 2;
        int sum = 0, procs_need = 0;
        for(int i = 0; i < all_info[MASTER].N; i++)
        {
            if(sum + nnz_per_row[i] > mid)
            {
                sum = nnz_per_row[i];
                procs_need++;
            } 
            else sum += nnz_per_row[i];
        }

        if(procs_need <= nprocs - 1)
            high = mid;
        else
            low = mid + 1;
    }

    all_info[MASTER] = (info_t) {all_info[MASTER].N, all_info[MASTER].nnz, 0, 0, 0, 0};

    int sum = 0, total_sum = 0, k = 0;
    for (int i = 0; i < all_info[0].N; i++) {
        if (sum + nnz_per_row[i] > low) {
            sum = nnz_per_row[i];
            k++;

            /* update process info */
            all_info[k].row_count = 1;
            all_info[k].row_start_idx = i;
            all_info[k].nz_proc = nnz_per_row[i];
            all_info[k].nz_start_idx = total_sum;
        }
        else {
            sum += nnz_per_row[i];
            
            /* update process info */
            all_info[k].row_count++;
            all_info[k].nz_proc += nnz_per_row[i];
        }
        total_sum += nnz_per_row[i];
    }

}