import os
from shutil import copytree, rmtree, ignore_patterns
from profit.profit import util, tqdm

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def rec2dict(rec):
    return {name:rec[name] for name in rec.dtype.names}

def write_input(eval_points, out_dir=''):
    '''
    write input.txt with parameter combinations to
    directory "out_dir"
    '''
    from numpy import array, savetxt
    filename = os.path.join(out_dir, 'input.txt')
    if isinstance(eval_points, dict):
        savetxt(filename,
                array(list(eval_points.values())).T,
                header=' '.join(eval_points.keys()))
    else:
        util.save_txt(filename, eval_points)


def fill_run_dir(eval_points, template_dir='template/', run_dir='run/',
                 overwrite = False):
    nrun = len(eval_points)
    kruns = tqdm(range(nrun))

    write_input(eval_points)

    for krun in kruns:
        run_dir_single = os.path.join(run_dir, str(krun).zfill(3)) #.zfill(3) is an option that forces krun to have 3 digits
        if os.path.exists(run_dir_single):
            if overwrite:
                rmtree(run_dir_single)
            else:
                raise RuntimeError('Run directory not empty: ' + run_dir_single)
        copy_template(template_dir, run_dir_single)

        fill_template(run_dir_single, eval_points[krun])

def copy_template(template_dir, out_dir, dont_copy=None):
    if dont_copy:
        copytree(template_dir, out_dir, ignore=ignore_patterns(*dont_copy))
    else:
        copytree(template_dir, out_dir)

def fill_template(out_dir, params, param_files=None):
    for root, dirs, files in os.walk(out_dir):
        for filename in files:
            if not param_files or filename in param_files:
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    content = content.format_map(SafeDict(rec2dict(params)))
                with open(filepath, 'w') as f:
                    f.write(content)
