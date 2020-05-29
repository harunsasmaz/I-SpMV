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