# Philosophy
## Directories & Paths

* for each parameter study, a *base directory* with the configuration file and the simulation, 
as well as the template directory or other supplementary files, is required.
* the output (generated data and model) will also be saved in the *base dir*
* logs and temporary data will be stored in a *temporary directory*, which is per default the same as the *base dir*
* with the `command` Worker and `template` Preprocessor, individual *run directories* will be created for each Worker/simulation run
* the content of the *run dirs* will be created according to the defined *template dir* (located relative to the *base dir*)

## Components & Sections

* the `run` configuration contains a few global options and three subsections: `runner`, `interface` & `worker`
* the `command` Worker has another two subsections: `pre` and `post`
* the options in each section are mostly the same as the arguments for `__init__`, 
some components have additional arguments which are set during the program flow (e.g. the `run_id` for the Worker)
* the `from_config` classmethod is defined for each component and selects the correct implementation
* `from_config` should also pass on any options which are needed further down (e.g. paths)
* if a Component takes sub-Components as arguments it should also support a config-dictionary
 

# The `run` configuration

```yaml
run:
    clean: True
    debug: False
    runner:
        class: local
        parallel: all
        logfile: runner.log  # tmp dir
        command: profit-worker
    interface:
        class: memmap
        path: interface.npy  # tmp dir
    worker:
        class: command
        command: ./simulation  # individual run dir
        stdout: stdout
        stderr: ~
        log_path: log  # tmp dir
        pre:
            class: template
            path: template  # base dir -> defines template dir -> individual run dir
            param_files: ~ 
        post:
            class: numpytxt
            path: stdout  # individual run dir
            names: ~
            options:
                deletechars: ""
```

Short:
```
run:
    runner:
        class: local
        parallel: all
    interface: memmap
    worker:
        class: command
        command: ./simulation  # individual run dir
        pre:
            class: template
            path: template  # base dir -> defines template dir -> individual run dir
            param_files: ~ 
        post:
            class: numpytxt
            path: stdout  # individual run dir
            names: ~
            options:
                deletechars: ""

```
Alternative:
```
run:
    class: local
    parallel: all 
    interface: memmap
    worker:
        class: command
        command: ./simulation  # individual run dir
        pre:
            class: template
            path: template  # base dir -> defines template dir -> individual run dir
            param_files: ~ 
        post:
            class: numpytxt
            path: stdout  # individual run dir
            names: ~
            options:
                deletechars: ""

```


---
```yaml
include: my_simulation.py  # base dir
run:
    clean: True
    debug: False
    log_path: log  # run dir
    runner:
        class: fork
        parallel: all
        logfile: runner.log 
    interface:
        class: memmap
        path: interface.npy  # run dir
    worker: my_simulation  # label of wrapped Worker
```

```yaml
run:
    clean: True
    debug: False
    runner:
        class: slurm
        parallel: ~
        logfile: runner.log
        sleep: 1  # s
        cpus: 1
        OpenMP: False
        custom: False
        path: slurm.bash  # tmp dir
        options:
            job-name: profit
        command: srun profit-worker 
    interface:
        class: zeromq
        transport: tcp
        port: 9000
        address: ~
        connection: ~
        bind: ~
        timeout: 2500  # ms
        retries: 3
        retry-sleep: 1  # s 
    worker:
        class: command
        log_path: log  # tmp dir
        command: ./simulation  # individual run dir = template dir
        stdout: stdout
        stderr: ~
        pre:
            class: template
            path: template  # base dir -> defines template dir -> individual run dir
            param_files: ~ 
        post:
            class: numpytxt
            path: stdout  # individual run dir
            names: ~
            options:
                deletechars: ""
```