#!/bin/bash
# run jobs in run directories, depending on
# slurm array task. Each task gets one node,
# the number of simulations per task (=per node),
# and the total number of simulations

task=$1
num_per_task=$2
total_num=$3
runid=$4

rundir=/gpfs/work/hofmeist/redmod-results/${runid}
jobcommand=/gpfs/home/hofmeist/redmod/testcases/sediment1d/sediment_io

# get start and end number of simulations
startn=$(expr $task \* $num_per_task)
endn=$(expr $startn + $num_per_task - 1)
if [ "$endn" -ge "$total_num" ] ; then
  endn=$(expr $total_num - 1)
fi

# start the simulations
running=0
for num in $(seq $startn $endn) ; do
  echo "(cd ${rundir}/$num ; $jobcommand)"
  (cd ${rundir}/$num ; rm -f PET* ; $jobcommand ) &> ${rundir}/${num}/log_${num}.txt &
  running=$(expr $running + 1)
  if [ "$running" -eq "48" ] ; then
    wait
    running=0
  fi
done

wait
