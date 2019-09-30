# mossco testcase (sediment1d)

This configuration runs an uncertainty quantification for a variation of more than three  model parameters in the seasonal cycling of nutrients in marine sediments. The single simulations run for a couple of minutes. SLURM is used to distribute the simulations on a cluster.

The configuration of the UQ is read in from the configuration yaml uq.yaml

> python ../../../suruq/suruq/main.py pre uq.yaml

Running the job is prepared by

> mkdir template # or copy template from the mossco setups
> python ../../../suruq/suruq/main.py run uq.yaml

get number of experiments from uq setup and add to submit_chunks.sh
then submit job based on run-ID and number of chunks (which is a number of contributing nodes)

> ./submit_chunks.sh runid number-of-chunks



