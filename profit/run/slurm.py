""" Scheduling jobs on a cluster with SLURM

targeted towards aCluster@tugraz.at
"""

from .runner import Runner

import subprocess
from time import sleep, time
import os


@Runner.register('slurm')
class SlurmRunner(Runner):
    def __init__(self, interface_class, config, base_config):
        super().__init__(interface_class, config, base_config)
        if self.config['custom']:
            if not os.path.exists(self.config['path']):
                self.logger.error(f'flag for custom script is set, but could not be found at '
                                  f'specified location {self.config["path"]}')
                self.logger.debug(f'cwd = {os.getcwd()}')
                self.logger.debug(f'ls = {os.listdir(os.path.dirname(self.config["path"]))}')
                raise FileNotFoundError(f'could not find {self.config["path"]}')
        else:
            self.generate_script()

    def spawn_run(self, params=None, wait=False):
        super().spawn_run(params, wait)  # fill data with params
        env = self.env.copy()
        env['PROFIT_RUN_ID'] = str(self.next_run_id)
        submit = subprocess.run(['sbatch', self.config['path'], '--parsable'],
                                cwd=self.base_config['run_dir'], env=env, capture_output=True, text=True, check=True)
        job_id = submit.stdout.split(';')[0].strip()
        self.runs[self.next_run_id] = job_id
        if wait:
            self.wait_for(self.next_run_id)

    def wait_for(self, run_id):
        poll_time = time()
        while run_id in self.runs:
            self.check_runs()
            sleep(self.config['sleep'])
            if time() - poll_time > self.config['poll']:
                self.check_runs(poll=True)
                poll_time = time()
                sleep(self.config['sleep'])

    def check_runs(self, poll=False):
        """ check the status of runs via the interface, poll only when specified """
        self.interface.poll()
        for run_id in list(self.runs):
            if self.interface.internal['DONE'][run_id]:
                del self.runs[run_id]
        if poll:
            acct = subprocess.run(['sacct', f'--name={self.config["job_name"]}', '--brief', '--parsable2'],
                                  capture_output=True, text=True, check=True)
            lookup = {job: run for run, job in self.runs.items()}
            for line in acct.stdout.split('\n'):
                job_id, state = line.split('|')[:2]
                if job_id in lookup:
                    if not (state.startswith('RUNNING') or state.startswith('PENDING')):
                        del self.runs[lookup[job_id]]

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: slurm
        parallel: 1         # maximum number of simultaneous runs (for spawn array)
        sleep: 0            # number of seconds to sleep while (internally) polling
        poll: 600           # number of seconds between external polls (to catch failed runs), use with care!
        path: slurm.bash    # the path to the generated batch script (relative to the base directory)
        custom: false       # whether a custom batch script is already provided at 'path'
        job_name: profit    # the name of the submitted jobs
        """
        if 'parallel' not in config:
            config['parallel'] = 1
        if 'sleep' not in config:
            config['sleep'] = 0
        if 'poll' not in config:
            config['poll'] = 600
        if 'path' not in config:
            config['path'] = 'slurm.bash'
        # convert path to absolute path
        if not os.path.isabs(config['path']):
            config['path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['path']))
        if 'custom' not in config:
            config['custom'] = False
        if 'job_name' not in config:
            config['job_name'] = 'profit'

    def generate_script(self):
        text = f"""\
#!/bin/bash
# automatically generated SLURM batch script for running simulations with proFit
# see https://github.com/redmod-team/profit

#SBATCH --job-name={self.config['job_name']}

if [[ -n $SLURM_ARRAY_TASK_ID ]]; then
    export PROFIT_ARRAY_ID=$SLURM_ARRAY_TASK_ID
fi
        
{'profit-worker' if not self.run_config['custom'] else self.run_config['command']}
"""
        with open(self.config['path'], 'w') as file:
            file.write(text)
