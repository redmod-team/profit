# mossco example (sediment1d)

This configuration runs an uncertainty quantification for a variation of three model parameters in the seasonal cycling of nutrients in marine sediments. The single simulations run for a couple of minutes. SLURM is used to distribute the simulations on a cluster.

The configuration of the UQ is read in from the configuration yaml uq.yaml

> mkdir template # or copy template from the mossco setups. Parameters to be changed have to appear in curly brackets, e.g. parameter value for "rLabile" has to be "{rLabile}" in the parameter/nml file.
> python ../../../profit/profit/main.py pre uq.yaml

Running the job is prepared by

> python ../../../profit/profit/main.py run uq.yaml
> sbatch slurm_uq_job.sh

Postprocessing is done in one single python script:

> python postprocess.py ./run


