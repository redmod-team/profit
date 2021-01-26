#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

from os import getcwd, path, chdir
import sys

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    from profit.util import tqdm_surrogate as tqdm

from profit.config import Config
from profit.util import safe_path_to_file
from profit.util.io import read_input, collect_output

yes = False  # always answer 'y'


def fit(x, y):
    from profit.sur.backend.gp import GPySurrogate
    fresp = GPySurrogate()
    fresp.train(x, y)
    return fresp


def fill_uq(self, krun, content):
    params_fill = SafeDict()
    kp = 0
    for item in self.params:
        params_fill[item] = self.eval_points[kp, krun]
        kp = kp+1
    return content.format_map(params_fill)


def print_usage():
    # TODO: Add options like active learning on/off: Usage: profit <mode> (--option) (base-dir)
    print("Usage: profit <mode> (base-dir)")
    print("Modes:")
    print("pre  ... prepare simulation runs based on templates")
    print("run  ... start simulation runs")
    print("collect ... collect simulation output")
    print("fit ... fit data with Gaussian Process")


def main():
    """
    Main command line interface
    sys.argv is an array whose values are the entered series of command
    (e.g.: sys.argv=['profit','run', '--active-learning', '/home/user/example'])
    """

    """ Get parameters from argv """
    print(sys.argv)
    if len(sys.argv) < 2:
        print_usage()
        return
    if len(sys.argv) < 3:
        base_dir_path = getcwd()
    elif len(sys.argv) < 4:
        # TODO: add options or everything in yaml?
        base_dir_path = sys.argv[2]
    else:
        print_usage()
        return

    """ Instantiate Config class from the given file """
    config_file = safe_path_to_file(base_dir_path, default='profit.yaml')
    config = Config.from_file(config_file)

    sys.path.append(config['base_dir'])

    if sys.argv[1] == 'pre':
        from profit.pre import fill_run_dir, get_eval_points

        """ Get input points ready to fill run directory """
        eval_points = get_eval_points(config)

        try:
            fill_run_dir(eval_points, template_dir=config['template_dir'],
                         run_dir=config['run_dir'], overwrite=False)
        except RuntimeError:
            question = "Warning: Run directories in {} already exist " \
                       "and will be overwritten. Continue? (y/N) ".format(config['run_dir'])
            if yes:
                print(question+'y')
            else:
                answer = input(question)
                if not answer.lower() == 'y':
                    exit()

            fill_run_dir(eval_points, template_dir=config['template_dir'],
                         run_dir=config['run_dir'], overwrite=True)

    elif sys.argv[1] == 'run':
        from profit.run import LocalCommand

        # TODO: Include options (in call or in config file) which run backend should be used.
        print(read_input(config['run_dir']))
        try:
            run = LocalCommand(config['run']['cmd'], config['run']['ntask'],
                               run_dir=config['run_dir'], base_dir=config['base_dir'])
            run.start()
        except KeyError:
            raise RuntimeError("No 'run' entry in profit.yaml")
        except FileNotFoundError:
            # TODO: Error occurs in single threads and is written to stderr.
            #       Make it easier for the user to recognise this error
            pass

    elif sys.argv[1] == 'collect':

        try:
            collect_output(config)
        except ImportError:
            question = "Interface could not be imported. Try with default interface? (y/N)"
            if yes:
                print(question+'y')
            else:
                answer = input(question)
                if not answer.lower() == 'y':
                    exit()

            collect_output(config, default_interface=True)

    elif sys.argv[1] == 'fit':
        from numpy import loadtxt
        from h5py import File #h5py lets you store huge amounts of numerical data, and easily manipulate that data from NumPy.
        x = loadtxt('input.txt')
        y = loadtxt('output.txt')
        fresp = fit(x, y)
        with File('profit.hdf5', 'w') as h5f: #creates a file under the name of 'profit.hdf5' having the h5f format with the following information:
            h5f['xtrain'] = fresp.xtrain
            h5f['ytrain'] = fresp.ytrain
            h5f['yscale'] = fresp.yscale
            h5f['ndim'] = fresp.ndim
            #h5f['variables'] = [
            #    v.numpy() for v in fresp.m.variables]

    elif sys.argv[1] == 'ui':
        from profit.ui import app
        app.app.run_server(debug=True)

    else:
        print_usage()
        return


if __name__ == '__main__':
    main()
