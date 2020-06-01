// matrix.h
#pragma once

#include "common.h"

struct csr_matrix 
{
  ind_type * csrRowPtr;
  ind_type * csrColIdx;
  val_type * csrVal;
  sz_type m;
  sz_type n;
  sz_type nnz;
  std::string name;

  csr_matrix();
  ~csr_matrix();
  csr_matrix(ind_type * csrRowPtr,
             ind_type * csrColIdx,
             val_type * csrVal,
             sz_type m,
             sz_type n,
             sz_type nnz,
             const std::string &name);
};

struct csc_matrix
{
  ind_type * cscRowIdx;
  ind_type * cscColPtr;
  val_type * cscVal;
  sz_type m;
  sz_type n;
  sz_type nnz;
  std::string name;

  csc_matrix();
  csc_matrix(ind_type * cscRowIdx,
             ind_type * cscColPtr,
             val_type * cscVal,
             sz_type m,
             sz_type n,
             sz_type nnz,
             const std::string &name);
};

