Run Systems
===========

example configurations
----------------------

.. code-block:: yaml

    run:
        command: ./simulation
        runner: local
        interface: memmap
        pre:
            class: template
            path: ../template_dir
            param_files: input.txt
        post:
            class: numpytxt
            path: output.txt
            names: "f g"

cluster
-------

.. code-block:: yaml

    run:
        command: ./simulation
        runner:
            class: slurm

custom script
~~~~~~~~~~~~~

.. code-block:: yaml

            custom: true
            path: slurm.bash
            options:
                job-name: profit-worker
                mem: 2G

OpenMP
~~~~~~

.. code-block:: yaml

            OpenMP: true
            cpus: 16

full configuration
------------------

.. code-block:: yaml

    run:
        command: ./simulation
        time: false
        clean: true
        stdout: stdout
        stderr: null
        time: false
        clean: true
        include: []
        runner:
            class: local
            parallel: 1
            sleep: 0
            fork: true
        interface:
            class: zeromq
            transport: tcp
            port: 9000
            timeout: 2500
            retries: 3
            retry-sleep: 1
        pre:
            class: template
            path: ../template_dir
            param_files: input.txt
        post:
            class: numpytxt
            path: output.txt
            names: "f g"


custom components
-----------------

.. code_block:: yaml

        include: custom_components.py
        post: identifier

.. code_block:: yaml

        include: custom_components.py
        post:
            class: identifier
            custom_option: 123

