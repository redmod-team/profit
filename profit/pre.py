import os
from shutil import copytree, rmtree, ignore_patterns
from profit import util


def rec2dict(rec):
    return {name: rec[name] for name in rec.dtype.names}


def write_input(eval_points, filename='input.txt'):
    """ Create input file with parameter combinations. """
    util.save(eval_points, filename)


def fill_run_dir(eval_points, template_dir='template/', run_dir='run/',
                 overwrite=False):
    """ Fill each run directory with input data according to template format. """

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        def tqdm(x): return x

    kruns = tqdm(range(eval_points.size))  # run with progress bar

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
                    # Escape '{*}' for e.g. json templates by replacing it with '{{*}}'.
                    # Variables then have to be declared as '{{*}}' which is replaced by a single '{*}'.
                    if '{{' in content:
                        content = content.replace('{{', '§').replace('}}', '§§')\
                            .replace('{', '{{').replace('}', '}}').replace('§§', '}').replace('§', '{')
                    content = content.format_map(util.SafeDict(rec2dict(params)))
                with open(filepath, 'w') as f:
                    f.write(content)


def get_eval_points(config):
    """ Create input data as numpy array from config information.
    Use corresponding variable kinds (e.g. Uniform, Normal, Independent, etc.)
    """

    import numpy as np

    inputs = config['input']

    npoints = config['ntrain']
    dtypes = [(key, inputs[key]['dtype']) for key in inputs.keys()]

    eval_points = np.zeros((npoints, 1), dtype=dtypes)

    for n, (k, v) in enumerate(inputs.items()):
        eval_points[k] = np.round(v['range']) if np.issubdtype(eval_points[k].dtype, np.integer) else v['range']

    return eval_points
