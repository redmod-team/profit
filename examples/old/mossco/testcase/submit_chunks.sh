#!/bin/bash

runid=$1
numchunks=$2
num_experiments=15625
# to round up, add numchunks-1 to num_experiments
numgross=$(expr ${num_experiments} + $numchunks - 1)
tasks_per_chunk=$(expr ${numgross} / $numchunks)

for (( i=0; i<$numchunks; i++)) ; do
 sbatch --error="uqjob_$i.err" --output="uqjob_$i.out" slurm_chunk.sh $runid $i $tasks_per_chunk
done
