# Iterative SpMV solver

## Author: Harun Sasmaz

Iterative Sparse Matrix - Dense Vector Multiplication program with 1D row partition.

> Unsymmetric COO format matrices are used to read by mmio.h and are converted to CSR format.

> Used MPI version: mpich v3.2.1

There are three parts:

Part I:

> Partition by equal rows to each process.

Part II:

> Hybrid MPI + OpenMP implementation

Part III:

> Load balanced row partitioning

Compile:

> First put the version you want to test in src, then call "make"

Test:

> mpirun -np <NUM_PROCS> build/spmv <test_matrix> <iteration_count>

Results:

Provided results are obtained by the matrices at;

> <https://sparse.tamu.edu/Janna/Cube_Coup_dt6>

> <https://sparse.tamu.edu/Janna/Flan_1565>
