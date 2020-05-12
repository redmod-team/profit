#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

import os
import importlib
from os import path, mkdir, walk

import sys
# import chaospy as cp
from collections import OrderedDict

try:
    from tqdm import tqdm
    use_tqdm = True
except:
    use_tqdm = False

    def tqdm(x):
        return x

import profit
from profit.config import Config
from profit.util import get_eval_points
#from profit.uq.backend import ChaosPy
#from profit.sur.backend import gp
#from inspect import signature
#from post import Postprocessor, evaluate_postprocessing

yes = False  # always answer 'y'


def fit(x, y):
    from profit.sur.backend.gp import GPFlowSurrogate
    fresp = GPFlowSurrogate()
    fresp.train(x, y)
    return fresp


def read_input(base_dir):
    from profit.util import load_txt
    data = load_txt(os.path.join(base_dir, 'input.txt'))
    return data.view((float, len(data.dtype.names))).T


def pre(self):
    write_input()
#        if(not isinstance(run.backend, run.PythonFunction)):
    if not path.exists(self.template_dir):
        print("Error: template directory {} doesn't exist.".format(self.template_dir))
    fill_run_dir()


def fill_uq(self, krun, content):
    params_fill = SafeDict()
    kp = 0
    for item in self.params:
        params_fill[item] = self.eval_points[kp, krun]
        kp = kp+1
    return content.format_map(params_fill)


def fill_template(self, krun, out_dir):
    for root, dirs, files in walk(out_dir):
        for filename in files:
            if not self.param_files or filename in self.param_files:
                filepath = path.join(root, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    #content = content.format_map(SafeDict(params))
                    content = self.fill_uq(krun, content)
                with open(filepath, 'w') as f:
                    f.write(content)


def print_usage():
    print("Usage: profit <mode> (base-dir)")
    print("Modes:")
    print("pre  ... prepare simulation runs based on templates")
    print("run  ... start simulation runs")
    print("collect ... collect simulation output")


def main():
    print(sys.argv)
    if len(sys.argv) < 2:
        print_usage()
        return

    if len(sys.argv) < 3:
        config_file = os.path.join(os.getcwd(), 'profit.yaml')
    else:
        config_file = os.path.abspath(sys.argv[2])

    config = Config()
    config.load(config_file)

    sys.path.append(config['base_dir'])

    if(sys.argv[1] == 'pre'):
        eval_points = get_eval_points(config)

        try:
            profit.fill_run_dir(eval_points, template_dir=config['template_dir'],
                                run_dir=config['run_dir'], overwrite=False)
        except RuntimeError:
            question = ("Warning: Run directories in {} already exist "
                        "and will be overwritten. Continue? (y/N) ").format(config['run_dir'])
            if (yes):
                print(question+'y')
            else:
                answer = input(question)
                if (not yes) and not (answer == 'y' or answer == 'Y'):
                    exit()

            profit.fill_run_dir(eval_points, template_dir=config['template_dir'],
                                run_dir=config['run_dir'], overwrite=True)

    elif(sys.argv[1] == 'run'):
        print(read_input(config['base_dir']))
        if config['run']:
            run = profit.run.LocalCommand(config['run'])
            run.start()
        else:
            raise RuntimeError('No "run" entry in profit.yaml')

    elif(sys.argv[1] == 'collect'):
        from numpy import array, empty, nan, savetxt
        from .util import save_txt
        spec = importlib.util.spec_from_file_location('interface',
                                                      config['interface'])
        interface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interface)
        data = empty((config['ntrain'], len(config['output'])))
        for krun in range(config['ntrain']):
            run_dir_single = os.path.join(config['run_dir'], str(krun))
            os.chdir(run_dir_single)
            try:
                data[krun,:] = interface.get_output()
            except:
                data[krun,:] = nan
        os.chdir(config['base_dir'])
        savetxt('output.txt', data, header=' '.join(config['output']))

    elif(sys.argv[1] == 'fit'):
        from numpy import loadtxt
        from h5py import File
        x = loadtxt('input.txt')
        y = loadtxt('output.txt')
        fresp = fit(x, y)
        with File('profit.hdf5', 'w') as h5f:
            h5f['xtrain'] = fresp.xtrain
            h5f['ytrain'] = fresp.ytrain
            h5f['yscale'] = fresp.yscale
            h5f['ndim'] = fresp.ndim
            h5f['variables'] = [
                v.numpy() for v in fresp.m.variables]

    elif(sys.argv[1] == 'ui'):
        from profit.ui import app
        app.app.run_server(debug=True)

    else:
        print_usage()
        return


if __name__ == '__main__':
    main()
