from os import path, chdir, listdir
from .util import load_txt


def read_input(run_dir):
    data = load_txt(path.join(run_dir, 'input.txt'))
    return data.view((float, len(data.dtype.names))).T


def collect_output(config, default_interface=False):
    from numpy import zeros, nan, savetxt
    from importlib.util import spec_from_file_location, module_from_spec
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        from .util import tqdm_surrogate as tqdm

    if not default_interface:
        try:
            spec = spec_from_file_location('interface', config['interface'])
            interface = module_from_spec(spec)
            spec.loader.exec_module(interface)
        except FileNotFoundError:
            raise ImportError("Could not load interface {}".format(config['interface']))
    else:
        try:
            name = [f for f in listdir(path.join(config['run_dir'], '000')) if f.endswith('out')][0]
        except IndexError:
            name = None
        interface = DefaultInterface(name)

    data = zeros((config['ntrain'], len(config['output'])))

    kruns = tqdm(range(config['ntrain']))
    for krun in kruns:
        # .zfill(3) is an option that forces krun to have 3 digits
        run_dir_single = path.join(config['run_dir'], str(krun).zfill(3))
        print(run_dir_single)
        try:
            chdir(run_dir_single)
            # TODO: make get_output run with parameters e.g. config['interface']['params'] as *args
            data[krun, :] = interface.get_output()
        except:
            data[krun, :] = nan
        finally:
            chdir(config['base_dir'])
    savetxt('output.txt', data, header=' '.join(config['output']))


class DefaultInterface:

    def __init__(self, name=None):
        self.name = name

    def get_output(self):
        from numpy import loadtxt
        return loadtxt(self.name)
