#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=spmv-jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --output=spmv-job.out

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Job...!"
echo "==============================================================================="
echo "Running compiled binary..."

#serial version
lscpu
echo "Serial version..."
build/spmv_serial Cube_Coup_dt6/Cube_Coup_dt6.mtx 20
build/spmv_serial Flan_1565/Flan_1565.mtx 20

#parallel version
echo "Parallel version with 1 process"
mpirun -np 1 build/spmv Cube_Coup_dt6/Cube_Coup_dt6.mtx 20

echo "Parallel version with 2 processes"
mpirun -np 2 build/spmv Cube_Coup_dt6/Cube_Coup_dt6.mtx 20

echo "Parallel version with 4 processes"
mpirun -np 4 build/spmv Cube_Coup_dt6/Cube_Coup_dt6.mtx 20

echo "Parallel version with 8 processes"
mpirun -np 8 build/spmv Cube_Coup_dt6/Cube_Coup_dt6.mtx 20

echo "Parallel version with 16 processes"
mpirun -np 16 build/spmv Cube_Coup_dt6/Cube_Coup_dt6.mtx 20

#echo "Parallel version with 16 threads"
#export OMP_NUM_THREADS=16
#export KMP_AFFINITY=verbose,granularity=fine,compact
#./life_parallel
