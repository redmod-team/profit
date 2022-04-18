.. _cluster:

Cluster Support
===============

proFit is desiged to schedule simulations on a cluster. As of v0.4 only the `slurm <slurm.schedmd.com>`_ scheduler is
supported. If you require a different scheduler consider contributing to proFit. All configuration is done as usual in
the study's configuration file (usually ``profit.yaml``). Using a provided script is also supported.
It is recommended to use the ``zeromq`` Interface as it is designed to be used with distributed Workers.

You can also start ``profit ui`` on the cluster and connect to it remotely using *ssh port forwarding*. The UI is
usually started on port 8050.

Example Configuration
---------------------

.. code-block:: yaml

    run:
        command: ./simulation
        runner:
            class: slurm
            OpenMP: true
            cpus: all
            options:
                job-name: profit-sim
                mem-per-cpu: 2G
        interface:
            class: zeromq
            port: 9100

Many clusters require specific options for each job like ``account``, ``mem`` or ``time``. These can be easily added to
using the ``runner/options`` dictionary, where the key has to be a valid option for the slurm batch script. However some
options (``cpus-per-task`` and ``ntasks`` as well as ``nodes`` and ``exclusive`` if ``cpus: all`` is set) are already
set internally.

Troubleshooting
---------------

- Each Worker writes a log file (usually into the `study/log` directory).
- A failed run is detected but is usually just missing from the output data, which causes the output to be misaligned
  with the input data. The current workaround is to delete the relevant lines from the input file manually.
- ``profit run`` can be started from a login node (it shouldn't use many resources) but sometimes ``zeromq`` can't
  connect from a worker node to the login node. Try ``srun profit run`` instead, as proFit will detect the correct host
  unless the ``connect`` address is overridden.

For more information and to submit the bugs you encountered visit the
`Issue Tracker on GitHub <https://github.com/redmod-team/profit/issues>`_.

Available Options
-----------------

.. autoraw:: profit.run.slurm.SlurmRunner.handle_config

.. autoraw:: profit.run.zeromq.ZeroMQRunnerInterface.handle_config
