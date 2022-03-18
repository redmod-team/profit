""" Scheduling runs on a cluster with SLURM

* targeted towards aCluster@tugraz.at
* each run is submitted as a job using a slurm batch script
* run arrays are submitted as a slurm job array
* by default completed runs are recognised by the interface, but the scheduler is polled as a fallback (less often)
"""

from .runner import Runner

import subprocess
from time import sleep, time
import os
from tqdm import trange


@Runner.register('slurm')
class SlurmRunner(Runner):
    """ Runner which submits each run as a job to the SLURM scheduler on a cluster

    - generates a slurm batch script with the given configuration
    - can also be used with a custom script
    - supports OpenMP
    - tries to minimize overhead by using job arrays if possible
    - polls the scheduler only at longer intervals
    """
    def __init__(self, interface_class, config, base_config):
        super().__init__(interface_class, config, base_config)
        self.env['SBATCH_EXPORT'] = 'ALL'
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
        self.logger.info(f'schedule run {self.next_run_id:03d} via Slurm')
        self.logger.debug(f'wait = {wait}, params = {params}')
        env = self.env.copy()
        env['PROFIT_RUN_ID'] = str(self.next_run_id)
        submit = subprocess.run(['sbatch', '--parsable', self.config['path']],
                                cwd=self.base_config['run_dir'], env=env, capture_output=True, text=True, check=True)
        job_id = submit.stdout.split(';')[0].strip()
        self.runs[self.next_run_id] = job_id
        if wait:
            self.wait_for(self.next_run_id)
        self.next_run_id += 1

    def spawn_array(self, params_array, blocking=True):
        self.logger.info(f'schedule array {self.next_run_id} - {self.next_run_id + len(params_array) - 1} via slurm')
        self.fill(params_array, offset=self.next_run_id)
        env = self.env.copy()
        env['PROFIT_RUN_ID'] = str(self.next_run_id)
        array_str = f'--array=0-{len(params_array) - 1}'
        if self.config['parallel'] is not None:
            array_str += f'%{self.config["parallel"]}'
        submit = subprocess.run(['sbatch', '--parsable', array_str, self.config['path']],
                                cwd=self.base_config['run_dir'], env=env, capture_output=True, text=True, check=True)
        job_id = submit.stdout.split(';')[0].strip()
        for i in range(len(params_array)):
            self.runs[self.next_run_id + i] = f'{job_id}_{i}'
        if blocking:
            self.wait_for_all([self.next_run_id + i for i in range(len(params_array))], show_tqdm=True)
        self.next_run_id += len(params_array)

    def wait_for_all(self, run_ids, show_tqdm=False):
        """wait until all specified runs have completed"""
        poll_time = time()
        if show_tqdm:
            progress = trange(len(run_ids))
        while len([i for i in run_ids if i in self.runs]):
            self.check_runs()
            sleep(self.config['sleep'])
            # poll the scheduler after a longer period
            if time() - poll_time > self.config['poll']:
                self.check_runs(poll=True)
                poll_time = time()
                sleep(self.config['sleep'])
            if show_tqdm:
                progress.update(len([i for i in run_ids if i not in self.runs]) - progress.n)

    def wait_for(self, run_id: int):
        """wait until the specified run has completed"""
        self.wait_for_all([run_id])

    def check_runs(self, poll: bool = False):
        """check the status of runs via the interface, poll only when specified"""
        # ask interface and remove all completed runs
        self.interface.poll()
        for run_id in list(self.runs):
            if self.interface.internal['DONE'][run_id]:
                self.delete_run(run_id)
        # poll the slurm scheduler to check for crashed runs
        if poll:
            acct = subprocess.run(['sacct', f'--name={self.config["options"]["job-name"]}', '--brief', '--parsable2'],
                                  capture_output=True, text=True, check=True)
            lookup = {job: run for run, job in self.runs.items()}
            for line in acct.stdout.split('\n'):
                if len(line) < 2:
                    continue
                job_id, state = line.split('|')[:2]
                if job_id in lookup:
                    run_id = lookup[job_id]
                    if not (state.startswith('RUNNING') or state.startswith('PENDING')):
                        # job has crashed or completed -> check backup
                        if self.check_backup(run_id):
                            self.delete_run(run_id)
                        else:  # failed / crashed
                            self.failed[run_id] = self.runs.pop(run_id)

    def cancel_all(self):
        from re import split
        ids = set()
        for run_id in self.runs:
            ids.add(split(r'[_.]', self.runs[run_id])[0])
        for job_id in ids:
            subprocess.run(['scancel', job_id])
        self.failed = self.runs
        self.runs = {}

    def delete_run(self, run_id: int):
        """helper: delete run from runs and remove slurm-stdout

        :param run_id: which run to delete, is a key in ``self.runs``
        """
        if self.run_config['clean'] and self.interface.internal['DONE'][run_id]:
            path = os.path.join(self.base_config['run_dir'], f'slurm-{self.runs[run_id]}.out')
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        del self.runs[run_id]

    def clean(self):
        """remove generated scripts and any slurm-stdout-files which match ``slurm-*.out``"""
        from re import match
        super().clean()
        if not self.config['custom'] and os.path.exists(self.config['path']):
            os.remove(self.config['path'])
        for direntry in os.scandir(self.base_config['run_dir']):
            if direntry.is_file() and match(r"slurm-(\d+)\.out", direntry.name) is not None:
                job_id = match(r"slurm-(\d+)\.out", direntry.name).groups()[0]
                if job_id not in self.failed.values():  # do not remove failed runs
                    os.remove(os.path.join(self.base_config['run_dir'], direntry.path))

    def generate_script(self):
        text = f"""\
#!/bin/bash
# automatically generated SLURM batch script for running simulations with proFit
# see https://github.com/redmod-team/profit
"""
        for key, value in self.config['options'].items():
            if value is not None:
                text += f"\n#SBATCH --{key}={value}"

        text += """
#SBATCH --ntasks=1
"""
        if self.config['cpus'] == 'all':
            text += """
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=$SLURM_CPUS_ON_NODE"""
        elif self.config['cpus'] > 1:
            text += f"""
#SBATCH --cpus-per-task={self.config['cpus']}"""

        if self.config['OpenMP']:
            text += """
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export OMP_PLACES=threads"""

        text += f"""
if [[ -n $SLURM_ARRAY_TASK_ID ]]; then
    export PROFIT_ARRAY_ID=$SLURM_ARRAY_TASK_ID
fi
export PROFIT_RUNNER_ADDRESS=$SLURM_SUBMIT_HOST

{self.config["prefix"]} {'profit-worker' if not self.run_config['custom'] else self.run_config['command']}
"""
        with open(self.config['path'], 'w') as file:
            file.write(text)
