from os import path, getcwd
from re import match, split
import yaml
from collections import OrderedDict
from typing import Mapping

from profit.run import Runner
from profit.sur import Surrogate

VALID_FORMATS = ('.yaml', '.py')

"""
yaml has to be configured to represent OrderedDict 
see https://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
and https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
"""


def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


def try_parse(s):
    funcs = [int, float]
    for f in funcs:
        try:
            return f(s)
        except ValueError:
            pass
    return s


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
yaml.add_representer(OrderedDict, represent_ordereddict)
yaml.add_constructor(_mapping_tag, dict_constructor)

""" now yaml is configured to handle OrderedDict input and output """


def load_config_from_py(filename):
    """ Load the configuration parameters from a python file into dict. """
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('f', filename)
    f = module_from_spec(spec)
    spec.loader.exec_module(f)
    return {name: value for name, value in f.__dict__.items() if not name.startswith('_')}


class Config(OrderedDict):
    """
    Configuration class
    This class provides a dictionary with possible configuration parameters for
    simulation, fitting and uncertainty quantification.

    Possible parameters in .yaml:

    base_dir: .
    run_dir: .
    uq: # TODO: implement
    interface: ./interface.py
    files:
        input: ./input.txt
        output: ./output.txt
    ntrain: 30
    variables:
        input1:
            kind: Normal
            range: (0, 1)
            dtype: float
        ...
        independent1:
            kind: Independent
            range: (0, 10, 1)
            dtype: int
        ...
        output1:
            kind: Output
            range: independent1
            dtype: float
    run:
        cmd: python3 ../simulation.py
        ntask: 4
    fit:
        surrogate: GPy
        kernel: RBF
        sigma_n: None
        sigma_f: 1e-6
        save: ./model.hdf5
        load: ./model.hdf5
        plot: Bool
            xpred: ((0, 1, 0.01), (0, 10, 0.1))
        plot_searching_phase: Bool
    """

    def __init__(self, base_dir=getcwd(), **entries):
        super(Config, self).__init__()
        self['base_dir'] = path.abspath(base_dir)
        self['run_dir'] = self['base_dir']
        self['uq'] = {}
        self['variables'] = {}
        self['fit'] = {'surrogate': 'GPy',
                       'kernel': 'RBF'}
        self['files'] = {'input': path.join(self['base_dir'], 'input.txt'),
                         'output': path.join(self['base_dir'], 'output.txt')}

        # Not to fill directly in file
        self['independent'] = {}
        self['input'] = {}
        self['output'] = {}
        self.update(entries)

    def write_yaml(self, filename='profit.yaml'):
        """ Dump UQ configuration to a yaml file.
        The default filename is profit.yaml
        """
        dumpdict = dict(self)
        self._remove_nones(dumpdict)
        with open(filename,'w') as file:
            yaml.dump(dumpdict,file,default_flow_style=False)

    @classmethod
    def from_file(cls, filename='profit.yaml'):
        """ Load configuration from .yaml or .py file.
        The default filename is profit.yaml """
        from profit.util import variable_kinds, safe_str, get_class_methods

        self = cls(base_dir=path.split(filename)[0])

        if filename.endswith('.yaml'):
            with open(filename) as f:
                entries = yaml.safe_load(f)
        elif filename.endswith('.py'):
            entries = load_config_from_py(filename)
        else:
            raise TypeError("Not supported file extension .{} for config file.\n"
                            "Valid file formats: {}".format(filename.split('.')[-1], VALID_FORMATS))
        self.update(entries)

        if path.isabs(filename):
            self['config_path'] = filename
        else:
            self['config_path'] = path.abspath(path.join(getcwd(), filename))

        """ Variable configuration
        kind: Independent, Uniform, etc.
        range: (start, end, step=1) or {'dependent variable': (start, end, step=1)} for output
        dtype: float64
        """

        halton_dim = []
        for k, v in self['variables'].items():
            if isinstance(v, str):
                # match word(comma seperated list of int or float)
                parsed = split('[()]', v)
                kind = parsed[0]
                args = parsed[1] if len(parsed) >= 2 else ''
                entries = tuple(try_parse(a) for a in args.split(',')) if args != '' else tuple()

                if isinstance(try_parse(v), (int, float)):  # PyYaml has problems parsing scientific notation
                    kind = 'Constant'
                    entries = (try_parse(v),)

                self['variables'][k] = {'kind': kind}

                if safe_str(kind) == 'output':
                    spl = split('[()]', v)
                    if len(spl) >= 3:
                        dependent = [var.strip() for var in split(',', spl[1])]
                    else:
                        dependent = []
                    self['variables'][k]['depend'] = tuple(dependent)
                    self['variables'][k]['range'] = {k: None for k in dependent}
                else:
                    try:
                        func = getattr(variable_kinds, safe_str(kind))
                        if safe_str(kind) == 'halton':
                            halton_dim.append((k, entries))
                        elif safe_str(kind) == 'independent':
                            self['variables'][k]['range'] = func(*entries, size=self['ntrain']) if entries else None
                        elif safe_str(kind) == 'activelearning':
                            self['variables'][k]['range'] = func(size=self['ntrain'])  # to insert nan in input file
                            self['variables'][k]['al_range'] = entries  # range between active learning should happen
                        else:
                            self['variables'][k]['range'] = func(*entries, size=self['ntrain'])
                            self['variables'][k]['dtype'] = str(self['variables'][k]['range'].dtype)
                    except AttributeError:
                        raise RuntimeError(f"Variable kind {kind} not defined.\n"
                                           f"Valid Functions: {get_class_methods(variable_kinds)}")
            elif isinstance(v, (int, float)):
                self['variables'][k] = {'kind': 'constant', 'range': variable_kinds.constant(v, size=self['ntrain'])}
                self['variables'][k]['dtype'] = str(self['variables'][k]['range'].dtype)
            elif isinstance(v, Mapping) and 'kind' in v:  # assume config dictionary is valid
                self['variables'][k] = v
            else:
                raise TypeError(f'cannot interpret variable {k} with value {v}')

            # Process data types
            if 'dtype' not in self['variables'][k].keys():
                self['variables'][k]['dtype'] = 'float64'

            # Add to corresponding variables 'output', 'independent' or 'input'
            kind = self['variables'][k]['kind'].lower()
            kind = kind if kind in ('output', 'independent') else 'input'
            if self['variables'][k].get('range') is not None:
                self[kind][k] = self['variables'][k]

        # Fill halton variables with single dimensions of n-D halton
        if halton_dim:
            halton = variable_kinds.halton(size=(self['ntrain'], len(halton_dim)))
            for d, (k, entries) in enumerate(halton_dim):
                diff = (entries[1] - entries[0]) if entries else 1
                low = entries[0] if entries else 0
                self['variables'][k]['range'] = diff * halton[:, d].reshape(-1, 1) + low
                self['input'][k] = self['variables'][k]

        # Fill range of output vector
        for k, v in self['output'].items():
            if not isinstance(v['range'], dict):
                v['range'] = {d: None for d in v['range']}
            shape = []
            for d in v['range']:
                self['output'][k]['range'][d] = self['variables'][d]['range']
                shape.append(self['variables'][d]['range'].shape[0])
            self['output'][k]['shape'] = tuple(shape)

        # Run configuration
        if 'run' not in self:
            self['run'] = {}
        if isinstance(self['run'], str):
            self['run'] = {'command': self['run']}
        Runner.handle_run_config(self)

        if self.get('fit'):
            Surrogate.handle_config(self['fit'], self)

        # Set missing mandatory dict entries to default
        if not self['files'].get('input'):
            self['files']['input'] = path.join(self['base_dir'], 'input.txt')
        if not self['files'].get('output'):
            self['files']['output'] = path.join(self['base_dir'], 'output.txt')

        # Set absolute paths
        self['files']['input'] = path.join(self['base_dir'], self['files']['input'])
        self['files']['output'] = path.join(self['base_dir'], self['files']['output'])
        if self['fit'].get('load'):
            self['fit']['load'] = path.join(self['base_dir'], self['fit']['load'])
        if self['fit'].get('save'):
            self['fit']['save'] = path.join(self['base_dir'], self['fit']['save'])
        return self

    def _remove_nones(self,config=None):
        if config==None: config=self.__dict__
        for key in list(config):
            if type(config[key]) is dict:
                self._remove_nones(config[key])
            #elif (type(config[key]) is not list) and (config[key] is None):
            else:
                if config[key] is None:
                    del config[key]
