#!/bin/bash

#SBATCH -J testofpi
#SBATCH -p general
#SBATCH -o out_1n2tasks.txt
#SBATCH -e err.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH -A c00698
start_time=$(date +%s%N) 
start_time_ms=$((start_time/1000000)) 
module load intel
module load python
mpiexec python ./llgpy_mpi.py
end_time=$(date +%s%N) 
end_time_ms=$((end_time/1000000)) 
execution_time_ms=$((end_time_ms - start_time_ms))
echo "Execution time: ${execution_time_ms} ms"
