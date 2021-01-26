from os import path, getcwd
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
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('f', filename)
    f = module_from_spec(spec)
    spec.loader.exec_module(f)
    return {name: value for name, value in f.__dict__.items() if not name.startswith('_')}


class Config(OrderedDict):
    # TODO: Update docstring
    """
    UQ configuration class
    This class provides a dictionary with possible configuration
    parameters of the RedMod UQ software, including parameters
    of variation, UQ-backend, and the SLURM configuration
    """

    def __init__(self, base_dir=getcwd(), **entries):
        super(Config, self).__init__()
        self['base_dir'] = base_dir
        self['template_dir'] = path.join(self['base_dir'], 'template')
        self['run_dir'] = path.join(self['base_dir'], 'run')
        self['command'] = None
        self['runner_backend'] = None
        self['uq'] = {}
        self['interface'] = path.join(self['base_dir'], 'interface.py')
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
        """ Load configuration from yaml file.
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

        # Variable configuration
        # Shorthand to put kind directly into variables
        for k, v in self['variables'].items():
            if isinstance(v, str):
                self['variables'][k] = {'kind': v}

            if self['variables'][k]['kind'] == 'Output':
                self['output'][k] = self['variables'][k]

        # Run configuration
        run = self['run']
        if run:
            # Shorthand to put cmd direcly into run
            if isinstance(run, str):
                self['run'] = {'cmd': run}

            # Default to single-thread
            if not 'ntask' in self['run']:
                self['run']['ntask'] = 1

            # TODO: add options like active_learning and check if e.g. cmd is in run
        else:
            raise NameError("No 'run' command found in config file.")

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
