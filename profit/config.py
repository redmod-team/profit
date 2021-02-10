from os import path, getcwd
from re import match
import yaml
from collections import OrderedDict

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
    """

    def __init__(self, base_dir=getcwd(), **entries):
        super(Config, self).__init__()
        self['base_dir'] = path.abspath(base_dir)
        self['template_dir'] = path.join(self['base_dir'], 'template')
        self['run_dir'] = path.join(self['base_dir'], 'run')
        self['command'] = None
        self['runner_backend'] = None
        self['uq'] = {}
        self['interface'] = path.join(self['base_dir'], 'interface.py')
        self['variables'] = {}
        self['save_format'] = 'txt'  # TODO: implement saving files as '.txt' or '.hdf'
        self['fit'] = {'surrogate': 'GPy',
                       'kernel': 'RBF',
                       'save': False,
                       'load': False,
                       'plot_results': False,
                       'plot_searching_phase': False}
        self['files'] = {'input': path.join(self['run_dir'], 'input.' + self['save_format']),
                         'output': path.join(self['run_dir'], 'output.' + self['save_format'])}

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

        """ Variable configuration
        kind: Independent, Uniform, etc.
        range: (start, end, step=1) or {'dependent variable': (start, end, step=1)} for output
        dtype: float64
        """
        for k, v in self['variables'].items():
            if isinstance(v, str):
                # match word(int_or_float, int_or_float, int_or_float)
                mat = match(r'(\w+)\(?(-?\d+(?:\.\d+)?)?,?\s?(-?\d+(?:\.\d+)?)?,?\s?(-?\d+(?:\.\d+)?)?\)?', v)
                kind = mat.group(1)
                entries = tuple(float(entry) for entry in mat.groups()[1:] if entry is not None)

                self['variables'][k] = {'kind': kind}

                if kind == 'Output':
                    # TODO: match arbitrary number of independent variables
                    mat = match(r'.*\((\w+)?[\,,\,\s]?(\w+)?', v)
                    dependent = tuple(d for d in mat.groups() if d is not None) \
                        if mat else ()
                    self['variables'][k]['range'] = {k: None for k in dependent}
                else:
                    self['variables'][k]['range'] = entries

            # Process data types
            if 'dtype' not in self['variables'][k].keys():
                self['variables'][k]['dtype'] = 'float64'

            # Add to corresponding variables 'output', 'independent' or 'input'
            kind = self['variables'][k]['kind']
            kind = kind.lower() if kind in ('Output', 'Independent') else 'input'
            self[kind][k] = self['variables'][k]

        # Fill range of output vector
        for k, v in self['output'].items():
            if not isinstance(v['range'], dict):
                v['range'] = {d: None for d in v['range']}
            for d in v['range']:
                self['output'][k]['range'][d] = self['variables'][d]['range']

        # Run configuration
        try:
            run = self['run']
            # Shorthand to put cmd direcly into run
            if isinstance(run, str):
                self['run'] = {'cmd': run}

            # Default to single-thread
            if 'ntask' not in self['run']:
                self['run']['ntask'] = 1
        except KeyError:
            pass

            # TODO: add options like active_learning and check if e.g. cmd is in run
            #       But don't do it here, but in the run phase.
            #       So the 'run' directory can be filled without the 'run' command in the config file.

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
