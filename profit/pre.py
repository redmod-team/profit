import os
from shutil import copytree, rmtree, ignore_patterns
from profit import util


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def rec2dict(rec):
    return {name: rec[name] for name in rec.dtype.names}


def write_input(eval_points, out_dir=''):
    """ Write input.txt with parameter combinations to directory 'out_dir' """

    from numpy import array, savetxt
    filename = os.path.join(out_dir, 'input.txt')
    if isinstance(eval_points, dict):
        savetxt(filename,
                array(list(eval_points.values())).T,
                header=' '.join(eval_points.keys()))
    else:
        util.save_txt(filename, eval_points)


def fill_run_dir(eval_points, template_dir='template/', run_dir='run/',
                 overwrite=False):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        from profit.util import tqdm_surrogate as tqdm

    nrun = len(eval_points)
    kruns = tqdm(range(nrun))  # run with progress bar

    for krun in kruns:

        # .zfill(3) is an option that forces krun to have 3 digits
        run_dir_single = os.path.join(run_dir, str(krun).zfill(3))
        if os.path.exists(run_dir_single):
            if overwrite:
                rmtree(run_dir_single)
            else:
                raise RuntimeError('Run directory not empty: {}'.format(run_dir_single))
        copy_template(template_dir, run_dir_single)

        fill_template(run_dir_single, eval_points[krun])

    write_input(eval_points, run_dir)  # place input.txt with all eval_points in run directory


def copy_template(template_dir, out_dir, dont_copy=None):
    """ TODO: explain dont_copy patterns """

    if dont_copy:
        copytree(template_dir, out_dir, ignore=ignore_patterns(*dont_copy))
    else:
        copytree(template_dir, out_dir)


def fill_template(out_dir, params, param_files=None):
    """ TODO: Explain param_files """
    for root, dirs, files in os.walk(out_dir):
        for filename in files:
            if not param_files or filename in param_files:
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    # TODO: check if something has to be done here, in case of IndependentRange variables
                    content = content.format_map(SafeDict(rec2dict(params)))
                with open(filepath, 'w') as f:
                    f.write(content)


def get_eval_points(config):
    from collections import OrderedDict
    from profit.util import variable_kinds
    import numpy as np
    import re

    inputs = OrderedDict()
    for k, v in config['variables'].items():

        # Treat an independent variable with range (ndependentRange(0, 1)) as input of type Linear
        if v['kind'] not in ('Output', 'Independent'):
            inputs[k] = v

            # Process data types
            if not 'dtype' in v.keys():
                v['dtype'] = 'float64'

    npoints = config['ntrain']
    dtypes = [(key, inputs[key]['dtype']) for key in inputs.keys()]

    eval_points = np.zeros(npoints, dtype=dtypes)

    for n, (k, v) in enumerate(inputs.items()):

        # match word(int_or_float, int_or_float, optional_int_or_float)
        mat = re.match(r'(\w+)\((\d+(?:\.\d+)?),.(\d+(?:\.\d+)?),?.?(\d+(?:\.\d+)?)?\)', v['kind'])
        kind = mat.group(1)
        inputs = tuple(float(entry) for entry in mat.groups()[1:] if entry is not None)

        try:
            func = getattr(variable_kinds, util.safe_str(kind))
        except AttributeError:
            raise AttributeError("Variable kind not defined.\n"
                                 "Valid Functions: {}".format(util.get_class_methods(variable_kinds)))
        x = func(*inputs, npoints)

        if np.issubdtype(eval_points[k].dtype, np.integer):
            eval_points[k] = np.round(x)
        else:
            eval_points[k] = x

    # TODO: if IndependentRange(...) is present, shape must be changed. Discuss with @kristophny
    #       for a single run
    #     # r       u       v
    #       0       4.7     3.7
    #       1       4.7     3.7
    #       2       4.7     3.7
    #       ...

    return eval_points
