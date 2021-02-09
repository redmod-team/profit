from os import path, chdir, listdir
from .util import load_txt


def read_input(run_dir):
    """ Loads data from 'input.txt' into a numpy array. """
    data = load_txt(path.join(run_dir, 'input.txt'))
    return data.view((float, len(data.dtype.names))).T


def collect_output(config, default_interface=False):
    """ Collects simulation results from each run directory into a single output file. """

    from numpy import zeros, arange, nan, savetxt
    from importlib.util import spec_from_file_location, module_from_spec
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        def tqdm(x): return x

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

    # Get vector output
    # TODO: make this more stable
    nout = 0
    for k, v in config['output'].items():
        nout += 1
        for d, r in v['range'].items():
            nout -= 1
            try:
                nout += (r[1] - r[0]) / (r[2] if len(r) > 2 else 1)
            except IndexError:
                raise RuntimeError("No range specified for independent variable '{}' "
                                   "which is used for output '{}'.".format(d, k))

    data = zeros((config['ntrain'], max(int(nout), 1)))

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
            chdir(config['run_dir'])

    # TODO: do this in less code?
    # Header for output
    header = []
    for out, values in config['output'].items():
        if not values['range']:
            header.append("{f}".format(f=out))
        else:
            for dependent, entries in values['range'].items():
                if not entries:
                    header.append("{f}({x})".format(f=out, x=dependent))
                else:
                    rng = arange(*entries)
                    for number in rng:
                        header.append("{f}({x}={n})".format(f=out, x=dependent, n=round(number, 2)))
    savetxt('output.txt', data, header=' '.join(header))


class DefaultInterface:
    """ Fallback interface if it is not provided by the user. """

    def __init__(self, name=None):
        self.name = name

    def get_output(self):
        from numpy import loadtxt
        return loadtxt(self.name)
