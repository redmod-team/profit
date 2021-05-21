"""proFit main script.

This script is called when running the `profit` command.
"""

from os import getcwd
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
import logging

from profit.config import Config
from profit.util import safe_path_to_file, safe_str

from profit.run import Runner

yes = False  # always answer 'y'


def fill_uq(self, krun, content):
    params_fill = SafeDict()
    kp = 0
    for item in self.params:
        params_fill[item] = self.eval_points[kp, krun]
        kp = kp+1
    return content.format_map(params_fill)


def main():
    """
    Main command line interface
    sys.argv is an array whose values are the entered series of command
    (e.g.: sys.argv=['profit','run', '--active-learning', '/home/user/example'])
    """

    """ Get parameters from argv """
    parser = ArgumentParser(usage='profit <mode> (base-dir)',
                            description="Probabilistic Response Model Fitting with Interactive Tools",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('mode',  # ToDo: subparsers?
                        metavar='mode',
                        choices=['run', 'fit', 'ui', 'clean'],
                        help='run ... start simulation runs \n'
                             'fit ... fit data with Gaussian Process \n'
                             'ui ... visualise results \n'
                             'clean ... remove run directories and input/output files')
    parser.add_argument('base_dir',
                        metavar='base-dir',
                        help='path to config file (default: current working directory)',
                        default=getcwd(), nargs='?')
    args = parser.parse_args()

    print(args)

    """ Instantiate Config class from the given file """
    config_file = safe_path_to_file(args.base_dir, default='profit.yaml')
    config = Config.from_file(config_file)

    sys.path.append(config['base_dir'])

    if args.mode == 'run':
        from tqdm import tqdm
        from profit.pre import get_eval_points, write_input
        from profit.util import save

        runner = Runner.from_config(config['run'], config)

        eval_points = get_eval_points(config)
        write_input(config['files']['input'], eval_points)

        if 'activelearning' in (safe_str(v['kind']) for v in config['input'].values()):
            from profit.fit import ActiveLearning
            from profit.sur.sur import Surrogate
            runner.fill(eval_points)
            if 'active_learning' not in config:
                config['active_learning'] = {}
            ActiveLearning.handle_config(config['active_learning'], config)
            al = ActiveLearning.from_config(runner, config['active_learning'], config)
            try:
                al.run_first()
                al.learn()
            finally:
                runner.cancel_all()
            if config['fit'].get('save'):
                al.save(config['fit']['save'])
        else:
            params_array = [row[0] for row in eval_points]
            try:
                runner.spawn_array(tqdm(params_array), blocking=True)
            finally:
                runner.cancel_all()

        if config['run']['clean']:
            runner.clean()

        if config['files']['output'].endswith('.txt'):
            data = runner.structured_output_data
            save(config['files']['output'], data.reshape(data.size, 1))
        else:
            save(config['files']['output'], runner.output_data)

    elif args.mode == 'fit':
        from numpy import arange, hstack, meshgrid
        from profit.util import load
        from profit.sur.sur import Surrogate

        sur = Surrogate.from_config(config['fit'], config)

        if not sur.trained:
            x = load(config['files']['input'])
            y = load(config['files']['output'])
            x = hstack([x[key] for key in x.dtype.names])
            y = hstack([y[key] for key in y.dtype.names])

            sur.train(x, y)

        if config['fit'].get('save'):
            sur.save_model(config['fit']['save'])
        if config['fit'].get('plot'):
            try:
                xpred = [arange(minv, maxv, step) for minv, maxv, step in config['fit']['plot'].get('Xpred')]
                xpred = hstack([xi.flatten().reshape(-1, 1) for xi in meshgrid(*xpred)])
            except AttributeError:
                xpred = None
            sur.plot(xpred, independent=config['independent'], show=True)

    elif args.mode == 'ui':
        from profit.ui import init_app
        app = init_app(config)
        app.run_server(debug=True)

    elif args.mode == 'clean':
        from shutil import rmtree
        from os import path, remove
        run_dir = config['run_dir']

        question = "Are you sure you want to remove the run directories in {} " \
                   "and input/output files? (y/N) ".format(config['run_dir'])
        if yes:
            print(question + 'y')
        else:
            answer = input(question)
            if not answer.lower().startswith('y'):
                print('exit...')
                sys.exit()

        for krun in range(config['ntrain']):
            single_run_dir = path.join(run_dir, f'run_{krun:03d}')
            if path.exists(single_run_dir):
                rmtree(single_run_dir)
        if path.exists(config['files']['input']):
            remove(config['files']['input'])
        if path.exists(config['files']['output']):
            remove(config['files']['output'])

        runner = Runner.from_config(config['run'], config)
        runner.clean()
        try:
            rmtree(config['run']['log_path'])
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()
