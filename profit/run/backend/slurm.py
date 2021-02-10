"""TODO: Refactor"""
from math import ceil
import os


class slurm_backend():
  '''
  Slurm backend class:
  A slurm run-script using job arrays is
  created from the configuration.
  '''

  def call_run(self):
    os.system('sbatch slurm_uq_job.sh')


  def write_slurm_scripts(self, num_experiments=125, slurm_config={}, jobcommand='./sediment_io'):

    # set slurm data, first set defaults:
    account = 'xy123'
    time = '00:10:00'
    job_name = 'redmod_uq'
    partition = 'compute'
    tasks_per_node = 36

    if 'account' in slurm_config: account = slurm_config['account']
    if 'time' in slurm_config: time = slurm_config['time']
    if 'tasks_per_node' in slurm_config: tasks_per_node = slurm_config['tasks_per_node']
    if 'partition' in slurm_config: partition = slurm_config['partition']
    if 'job_name' in slurm_config: job_name = slurm_config['job_name']

    # first write script for single node

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
      os.chmod('run_uq_job.sh',509) # make run script executable

      # now write slurm batch script

      num_nodes=ceil(num_experiments/tasks_per_node)

      with open('slurm_uq_job.sh','w') as f:
        f.write('''#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --comment="RedMod UQ"
#SBATCH --partition=%s   # Specify partition name
#SBATCH --output=uqjob-%%A_%%a.out
#SBATCH --error=uqjob-%%A_%%a.err
#SBATCH --nodes=1
#SBATCH --array=0-%d
#SBATCH --time=%s
#SBATCH --account=%s

number_of_experiments=%d
tasks_per_node=%d


./run_uq_job.sh ${SLURM_ARRAY_TASK_ID} ${tasks_per_node} ${number_of_experiments}
'''%(job_name,partition,num_nodes,time,account,num_experiments,tasks_per_node))

      # now output use string
      print('  run job using:\nsbatch slurm_uq_job.sh')
