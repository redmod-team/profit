from os import path, chdir, listdir
from .util import save, load


def read_input(filename):
    """ Loads data from input file into a numpy array. """
    data = load(filename)
    return data.view((float, len(data.dtype.names))).T


def collect_output(config, default_interface=False):
    """ Collects simulation results from each run directory into a single output file. """

    from numpy import zeros, nan
    from importlib.util import spec_from_file_location, module_from_spec
    from tqdm import tqdm

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

    # TODO: do this in less code?
    # Header for output
    header = []
    for out, values in config['output'].items():
        if not values['range']:
            header.append("{f}".format(f=out))
        else:
            for dependent, rng in values['range'].items():
                for number in rng.flatten():
                    header.append("{f}({x}={n})".format(f=out, x=dependent, n=round(number, 2)))

    # Get vector output
    nout = 0
    for v in config['output'].values():
        for rng in v['range'].values():
            nout += rng.size

    dtypes = [(key, float) for key in config['output'].keys()]
    data = zeros((config['ntrain'], max(int(nout), 1)), dtype=dtypes)

    kruns = tqdm(range(config['ntrain']))
    for krun in kruns:
        # .zfill(3) is an option that forces krun to have 3 digits
        run_dir_single = path.join(config['run_dir'], str(krun).zfill(3))
        print(run_dir_single)
        try:
            chdir(run_dir_single)
            # TODO: make get_output run with parameters e.g. config['interface']['params'] as *args
            # Interface should return a tuple or list if more than one output variable.
            alldata = interface.get_output()
            for i, key in enumerate(data.dtype.names):
                data[key][krun, :] = alldata[i] if isinstance(alldata, (tuple, list)) else alldata
        except:
            data[krun, :] = nan
        finally:
            chdir(config['run_dir'])

    save(config['files']['output'], data, ' '.join(header))
