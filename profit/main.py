"""proFit main script.

This script is called when running the `profit` command inside a shell.
"""

import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from platform import python_version

from profit.config import BaseConfig
from profit.util import safe_path
from profit.util.variable import VariableGroup, Variable
from profit.defaults import base_dir as default_base_dir, config_file as default_config_file
from profit.run import Runner

YES = False  # always answer 'y'


def main():
    """Main command line interface"""

    # Get parameters from shell input
    parser = ArgumentParser(usage='profit <mode> (base-dir)',
                            description="Probabilistic Response Model Fitting with Interactive Tools",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('mode',  # ToDo: subparsers?
                        metavar='mode',
                        choices=['run', 'fit', 'ui', 'clean'],
                        help='run   ... start simulation runs\n'
                             'fit   ... fit data with Gaussian Process\n'
                             'ui    ... visualise results\n'
                             'clean ... remove run directories and input/output files')
    parser.add_argument('base_dir',
                        metavar='base-dir',
                        help='path to config file (default: current working directory)',
                        default=default_base_dir, nargs='?')

    from profit import __version__  # delayed to prevent cyclic import
    print(f"proFit {__version__} for python {python_version()}")
    args = parser.parse_args()

    print(args)

    # Instantiate Config from the given file
    config_file = safe_path(args.base_dir, default=default_config_file)
    config = BaseConfig.from_file(config_file)

    sys.path.append(config['base_dir'])

    # Create variables
    variables = VariableGroup(config['ntrain'])
    vars = []
    for k, v in config['variables'].items():
        if isinstance(v, str):
            vars.append(Variable.create_from_str(k, (config['ntrain'], 1), v))
        else:
            vars.append(Variable.create(**v))
    variables.add(vars)

    # Select mode
    if args.mode == 'run':
        from tqdm import tqdm
        from profit.util import check_ndim
        from profit.util.file_handler import FileHandler

        runner = Runner.from_config(config['run'], config)  # Instantiate the runner
        if config['active_learning']['resume_from'] is not None:
            X = FileHandler.load(config['files']['input'])
            y = FileHandler.load(config['files']['output'])
            for v in variables.list:
                if v.name in X.dtype.names:
                    v.value = X[v.name]
                else:
                    v.value = y[v.name]
        FileHandler.save(config['files']['input'], variables.named_input)  # Save variables to input file

        # Check if active learning needs to be done instead of executing a normal run
        if 'activelearning' in (v.kind.lower() for v in variables.list):
            from profit.al.active_learning import ActiveLearning
            from profit.sur.sur import Surrogate

            runner.fill(variables.named_input)  # Prepare runner with input variables
            al = ActiveLearning.from_config(runner, variables, config['active_learning'], config)  # Instantiate the active learning algorithm
            try:
                save_intermediate = {'model_path': config['fit']['save'] or config['fit']['load'],
                                     'input_path': config['files']['input'],
                                     'output_path': config['files']['output']}
                if config['active_learning']['resume_from']:
                    resume_from = config['active_learning']['resume_from']
                    runner.fill_output(variables.named_output)
                    runner.next_run_id = resume_from
                    al.learn(resume_from=resume_from, save_intermediate=save_intermediate)
                else:
                    al.warmup(save_intermediate=save_intermediate)  # Execute warmup cycles
                    al.learn(save_intermediate=save_intermediate)  # Execute main learning loop
                FileHandler.save(config['files']['input'], variables.named_input)  # Save learned input variables
            finally:
                runner.cancel_all()  # Close all run processes
            if config['active_learning']['algorithm']['save']:
                al.save(config['active_learning']['algorithm']['save'])  # Save AL model
        else:
            # Normal (parallel) run
            params_array = [row[0] for row in variables.named_input]  # Structured array of input values
            try:
                runner.spawn_array(tqdm(params_array), blocking=True)  # Start runs
            finally:
                runner.cancel_all()  # Close all run processes

                # Write runner output data into variables
                for key in runner.output_data.dtype.names:
                    variables[key].value = check_ndim(runner.output_data[key])

        if config['run']['clean']:
            runner.clean()

        # Format output data for txt file and save
        formatted_output_data = variables.formatted_output \
            if config['files']['output'].endswith('.txt') else variables.named_output
        FileHandler.save(config['files']['output'], formatted_output_data)

    elif args.mode == 'fit':
        from numpy import arange, hstack, meshgrid
        from profit.util.file_handler import FileHandler
        from profit.sur.sur import Surrogate

        sur = Surrogate.from_config(config['fit'], config)  # Instantiate surrogate model

        # Train model if not yet trained
        if not sur.trained:
            x = FileHandler.load(config['files']['input'])
            y = FileHandler.load(config['files']['output'])
            x = hstack([x[key] for key in x.dtype.names])  # Convert structured to normal array
            y = hstack([y[key] for key in y.dtype.names])

            sur.train(x, y)

        if config['fit']['save']:
            sur.save_model(config['fit']['save'])  # Save surrogate model
        if config['ui']['plot']:
            # Make a simple plot of data and surrogate model
            # TODO: Rename to 'simple_plot' and introduce more options
            try:
                # Get prediction range from input or infer from training data
                xpred = [arange(minv, maxv, step) for minv, maxv, step in config['ui']['plot'].get('Xpred')]
                xpred = hstack([xi.flatten().reshape(-1, 1) for xi in meshgrid(*xpred)])
            except AttributeError:
                xpred = None
            sur.plot(xpred, independent=config['independent'], show=True)

    elif args.mode == 'ui':
        from profit.ui import init_app
        app = init_app(config)
        app.run_server(debug=True)  # Start Dash server on localhost

    elif args.mode == 'clean':
        from shutil import rmtree
        from os import path, remove
        run_dir = config['run_dir']

        question = "Are you sure you want to remove the run directories in {} " \
                   "and input/output files? (y/N) ".format(config['run_dir'])
        if YES:
            print(question + 'y')
        else:
            answer = input(question)
            if not answer.lower().startswith('y'):
                print('exit...')
                sys.exit()
        # Remove single run directories
        for krun in range(config['ntrain']):
            single_run_dir = path.join(run_dir, f'run_{krun:03d}')
            if path.exists(single_run_dir):
                rmtree(single_run_dir)

        # Remove input and output files
        if path.exists(config['files']['input']):
            remove(config['files']['input'])
        if path.exists(config['files']['output']):
            remove(config['files']['output'])

        # Cleanup runner
        runner = Runner.from_config(config['run'], config)
        runner.clean()
        try:
            # Remove log
            rmtree(config['run']['log_path'])
        except FileNotFoundError:
            pass
        if path.exists('runner.log'):
            remove('runner.log')


if __name__ == '__main__':
    main()
