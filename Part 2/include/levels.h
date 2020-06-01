// levels.h
// Level set calculations and data structures
// Created: 05-12-2019
// Author: Najeeb Ahmad

#ifndef __HEADER_LEVELS__
#define __HEADER_LEVELS__

#include "common.h"
#include "matrix.h"

struct level_info
{
	int nlevel;
	int parallelism_min;
	int parallelism_max;
	int parallelism_avg;
	ind_type *levelPtr;
	ind_type *levelItem;
	ind_type *levelItemNewRowIdx;    // Stores new numbering of matrix rows
	                           // after level set re-ordering.
	                           // e.g. levelItemNewRowIdx[11] = 6 means old
	                           // row 11 is now 6 in new ordering
    ind_type *num_levelRows;   // Number of level rows
	ind_type *num_levelNnzs;   // Number of level nnzs
	ind_type *num_indegree;    // Row indegree
	ind_type *num_outdegree;   // Row outdegree
	ind_type *level_of_row;    // Level number of each row
	ind_type *level_delta;     // Change in level
	float *parallelism;        // parallelism
	float *avg_row_nnzs;       // average row nnzs
	float *avg_col_nnzs;       // average column nnzs
	float *avg_level_delta;    // average level delta
	float *col_center;         // center of column distribution
	ind_type *nnz_per_row;     // nnzs per row
	ind_type *nnz_per_col;     // nnz per column  
	ind_type *cum_Rows;
	ind_type *cum_Nnzs;
};

/*int findlevel_csr(hyb_csr_matrix *tri_mat, level_info *lvl_info)
{
	if(tri_mat->TopTri.m != tri_mat->TopTri.n)
	{
		printf("The matrix is not square. Exiting!\n");
		return EXIT_FAILURE;
	}
}*/

#endif

