import numpy as np
import os

def write_input(eval_points, run_dir='run/'):
    '''
    write input.txt with parameter combinations to
    directory "run_dir"
    '''
    np.savetxt(os.path.join(run_dir, 'input.txt'), 
            np.array(list(eval_points.values())).T, 
            header=' '.join(eval_points.keys()))


def fill_run_dir(eval_points, template_dir='template/', run_dir='run/'):
    nrun = eval_points.shape[1]
    kruns = tqdm(range(nrun))

    for krun in kruns:
        run_dir_single = path.join(run_dir, str(krun))
        if path.exists(run_dir_single):
            rmtree(run_dir_single)
        copy_template(template_dir, run_dir_single)

        for item in self.params:
            params_fill[item] = self.eval_points[kp, krun]
        fill_template(run_dir_single, params)


def fill_template(out_dir, params, param_files=None):
    for root, dirs, files in walk(out_dir):
        for filename in files:
            if not param_files or filename in param_files:
                filepath = path.join(root, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    content = content.format_map(SafeDict(params))
                with open(filepath, 'w') as f:
                    f.write(content)
