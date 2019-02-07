from math import ceil
import os

class slurm_backend():
  '''
  Slurm backend class:
  A slurm run-script using job arrays is
  created from the configuration.
  '''

  def call_run():
    os.system('sbatch slurm_uq_job.sh')


  def write_slurm_scripts(tasks_per_node=36, account='xy0123'):

    # first write script for single node
    jobcommand='./sediment_io' # to be taken from config

    with open('run_uq_job.sh','w') as f:
      f.write('''#!/bin/bash
# run jobs in run directories, depending on
# slurm array task. Each task gets one node,
# the number of simulations per task (=per node),
# and the total number of simulations

task=$1
num_per_task=$2
total_num=$3

# write jobcommand
#currdir=`pwd`
#jobcommand="$currdir/sediment_io"
jobcommand=%s

# get start and end number of simulations
startn=$(expr $task \* $num_per_task)
endn=$(expr $startn + $num_per_task - 1)
if [ "$endn" -ge "$total_num" ] ; then
  endn=$(expr $total_num - 1)
fi

# start the simulations
for num in $(seq $startn $endn) ; do
  echo "(cd run/$num; $jobcommand) &"
  (cd run/$num; $jobcommand) &
done

wait'''%(jobcommand))

      # now write slurm batch script

      num_experiments=125 # todo: get from config
      num_nodes=ceil(num_experiments/tasks_per_node)

      with open('slurm_uq_job.sh','w') as f:
        f.write('''#!/bin/bash
#SBATCH --job-name=redmod_uq
#SBATCH --comment="RedMod UQ"
#SBATCH --partition=compute2   # Specify partition name
#SBATCH --output=uqjob-%%A_%%a.out
#SBATCH --error=uqjob-%%A_%%a.err
#SBATCH --nodes=1
#SBATCH --array=0-%d
#SBATCH --time=00:20:00
#SBATCH --account=%s

number_of_experiments=%d
tasks_per_node=%d


./run_job.sh ${SLURM_ARRAY_TASK_ID} ${tasks_per_node} ${number_of_experiments}
'''%(num_nodes,account,num_experiments,tasks_per_node))

      # now output use string
      print('  run job using:\nsbatch slurm_uq_job.sh')
