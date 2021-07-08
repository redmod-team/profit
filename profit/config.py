from os import path
import yaml
from collections import OrderedDict
from profit import defaults
from abc import ABC

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


class AbstractConfig(ABC):
    """
    Abstract base class with general methods.
    """
    _sub_configs = {}

    def update(self, **entries):
        for name, value in entries.items():
            if name.lower() in (name.lower() for name in self._sub_configs):
                pass
            elif hasattr(self, name):
                self.__setattr__(name, value)
            else:
                raise AttributeError("Config parameter {} is not available.".format(name))

    def as_dict(self):
        return {name: getattr(self, name)
                if name not in map(str.lower, self._sub_configs.keys()) or getattr(self, name) is None
                else getattr(self, name).as_dict() for name in vars(self)}

    def set_defaults(self, default_dict):
        for name, value in default_dict.items():
            setattr(self, name, value)


class BaseConfig(AbstractConfig):
    """
    This class and its modular subclasses provide all possible configuration parameters.

    Parts of the Config:
        - base_dir
        - run_dir
        - config_file
        - ntrain
        - variables
        - files
            - input
            - output
        - run
        - fit
            - surrogate
            - save / load
            - fixed_sigma_n
        - active_learning
        - ui

    Base configuration for fundamental parameters.

    Parameters:
        base_dir (str): Base directory.
        run_dir (str): Run directory.
        config_path (str): Path to configuration file.
        files (dict): Paths for input and output files.
        ntrain (int): Number of training samples.
        variables (dict): All variables.
        input (dict): Input variables.
        output (dict): Output variables.
        independent (dict): Independent variables, if the result of the simulation is a vector.
    """

    def __init__(self, base_dir=defaults.base_dir, **entries):
        self.base_dir = path.abspath(base_dir)
        self.run_dir = self.base_dir
        self.config_path = path.join(self.base_dir, defaults.config_file)
        self.ntrain = defaults.ntrain
        self.variables = defaults.variables
        self.input = {}
        self.output = {}
        self.independent = {}
        self.files = defaults.files

        # Create sub configs
        entries_lower = {key.lower(): entry for key, entry in entries.items()}
        for name, sub_config in self._sub_configs.items():
            sub_config_label = name.lower()
            if name.lower() in entries_lower:
                sub = sub_config(**entries_lower[name.lower()])
                self.__setattr__(sub_config_label, sub)
            else:
                self.__setattr__(sub_config_label, sub_config())

        self.update(**entries)  # Update the attributes with given entries.
        self.process_entries()  # Postprocess the attributes to standardize different user entries.

    def process_entries(self):
        from profit.util.variable_kinds import Variable, VariableGroup

        # Set absolute paths
        self.files['input'] = path.join(self.base_dir, self.files.get('input', defaults.files['input']))
        self.files['output'] = path.join(self.base_dir, self.files.get('output', defaults.files['output']))

        # Variable configuration as dict
        variables = VariableGroup(self.ntrain)
        vars = []
        for k, v in self.variables.items():
            if type(v) in (str, int, float):
                if isinstance(try_parse(v), (int, float)):
                    v = 'Constant({})'.format(try_parse(v))
                vars.append(Variable.create_from_str(k, (self.ntrain, 1), v))
            else:
                vars.append(Variable.create(name=k, size=(self.ntrain,1), **v))
        variables.add(vars)

        self.variables = variables.as_dict
        self.input = {k: v for k, v in self.variables.items()
                         if not any(k in v['kind'].lower() for k in ('output', 'independent'))}
        self.output = {k: v for k, v in self.variables.items()
                       if 'output' in v['kind'].lower()}
        self.independent = {k: v for k, v in self.variables.items()
                            if 'independent' in v['kind'].lower() and v['size'] != (1, 1)}

        # Process sub configurations
        for name in self._sub_configs:
            sub = getattr(self, name.lower())
            if sub:
                sub.process_entries(self)

    @classmethod
    def from_file(cls, filename=defaults.config_file):

        if filename.endswith('.yaml'):
            with open(filename) as f:
                entries = yaml.safe_load(f)
        elif filename.endswith('.py'):
            entries = load_config_from_py(filename)
        else:
            raise TypeError("Not supported file extension .{} for config file.\n"
                            "Valid file formats: {}".format(filename.split('.')[-1], VALID_FORMATS))
        self = cls(base_dir=path.split(filename)[0], **entries)
        self.config_path = path.join(self.base_dir, filename)
        return self

    @classmethod
    def register(cls, label):
        def decorator(config):
            if label in cls._sub_configs:
                raise KeyError(f'registering duplicate label {label} for Interface')
            cls._sub_configs[label] = config
            return config
        return decorator


@BaseConfig.register("Run")
class RunConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(defaults.run)
        self.update(**entries)

    def process_entries(self, base_config):
        from profit.run import Runner
        sub = Runner.handle_run_config(base_config.as_dict(), self.as_dict())
        self.update(**sub)


@BaseConfig.register("Fit")
class FitConfig(AbstractConfig):

    def __init__(self, **entries):
        from profit.sur import Surrogate
        from profit.sur.gaussian_process import GaussianProcess
        self.set_defaults(defaults.fit)

        if issubclass(Surrogate._surrogates[self.surrogate], GaussianProcess):
            self.set_defaults(defaults.fit_gaussian_process)

        self.update(**entries)

    def process_entries(self, base_config):
        from profit.sur import Surrogate
        if self.load:
            self.load = path.join(base_config.base_dir, self.load)
        if self.save:
            self.save = path.join(base_config.base_dir, self.save)
        sub = Surrogate.handle_config(self.as_dict(), base_config.as_dict())
        self.update(**sub)


@BaseConfig.register("Active_Learning")
class ALConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(defaults.active_learning)
        self.update(**entries)

    def process_entries(self, base_config):
        from profit.fit import ActiveLearning
        sub = ActiveLearning.handle_config(self.as_dict(), base_config.as_dict)
        self.update(**sub)


@BaseConfig.register("UI")
class UIConfig(AbstractConfig):

    def __init__(self, **entries):
        self.plot = defaults.ui['plot']
        self.update(**entries)

    def process_entries(self, base_config):
        pass
