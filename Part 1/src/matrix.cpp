// matrix.cpp
#include "../include/matrix.h"
#include "../include/common.h"

csr_matrix::csr_matrix(ind_type * const csrRowPtr,
             		   ind_type * const csrColIdx,
             		   val_type * const csrVal,
             		   const sz_type m,
             	     const sz_type n,
             		   const sz_type nnz,
             		   const std::string &name)
    : csrRowPtr(csrRowPtr), csrColIdx(csrColIdx), csrVal(csrVal), m(m), n(n), nnz(nnz), name(name) {}

csc_matrix::csc_matrix(ind_type * const cscRowIdx,
                   ind_type * const cscColPtr,
                   val_type * const cscVal,
                   const sz_type m,
                   const sz_type n,
                   const sz_type nnz,
                   const std::string &name)
    : cscRowIdx(cscRowIdx), cscColPtr(cscColPtr), cscVal(cscVal), m(m), n(n), nnz(nnz), name(name) {}

csr_matrix::csr_matrix()
{

}

csc_matrix::csc_matrix()
{
  
}

csr_matrix::~csr_matrix()
{
  //free(csrRowPtr);
  //free()
}

