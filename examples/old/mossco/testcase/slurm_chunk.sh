#!/bin/bash
#SBATCH --job-name=redmod_uq
#SBATCH --comment="RedMod UQ"
#SBATCH --partition=pCluster   # Specify partition name
#SBATCH --output=uqjob.out
#SBATCH --error=uqjob.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=03:00:00
#SBATCH --account=KST

module load compilers/intel
module load intelmpi

number_of_experiments=15625
#runs_per_task=326 # 1 node
#runs_per_task=1954 # 8 nodes
runs_per_task=$3 # 10 nodes
runid=$1
chunknum=$2

./run_chunk.sh ${chunknum} ${runs_per_task} ${number_of_experiments} ${runid}


