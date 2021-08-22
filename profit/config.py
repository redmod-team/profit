from os import path
import yaml
from collections import OrderedDict
from profit import defaults
from abc import ABC
import warnings

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
    funcs = [float]
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
    """General class with methods which are useful for all Config classes."""
    _sub_configs = {}

    def update(self, **entries):
        """Updates the attributes with user inputs. A warning is issued if the attribute set by the user is unknown.

        Parameters:
            entries (dict): User input of the config parameters.
        """
        for name, value in entries.items():
            if hasattr(self, name) or name in map(str.lower, self._sub_configs):
                attr = getattr(self, name, None)
                if isinstance(attr, dict):
                    attr.update(value)
                    setattr(self, name, attr)
                else:
                    setattr(self, name, value)
            else:
                message = "Config parameter '{}' for {} configuration may be unused.".format(name, self.__class__.__name__)
                warnings.warn(message)
                setattr(self, name, value)

    def process_entries(self, base_config):
        """After the attributes are set, they are formatted and edited to standardize the user inputs.

        Parameters:
            base_config (BaseConfig): In sub configs, the data from the base config is needed.
        """
        pass

    def set_defaults(self, default_dict):
        """Default values are set from a default dictionary, which is usually located
        in the global profit.defaults file.
        """
        for name, value in default_dict.items():
            setattr(self, name, value)

    def create_subconfigs(self, **entries):
        """Instances of sub configs are created from a string or a dictionary.

        Parameters:
            entries (dict): User input parameters.
        """
        entries_lower = {key.lower(): entry for key, entry in entries.items()}
        for name, sub_config in self._sub_configs.items():
            sub_config_label = name.lower()
            if name.lower() in entries_lower:
                entry = entries_lower[name.lower()]
                if isinstance(entry, str):
                    entry = {'class': entry}
                sub = sub_config(**entry)
                setattr(self, sub_config_label, sub)
            else:
                setattr(self, sub_config_label, sub_config())

    def __getitem__(self, item):
        """Implements the dictionary like get method with brackets.

        Parameters:
            item (str): Label of the attribute to return.

        Returns:
            Attribute or if the attribute is a sub config, a dictionary of the sub config items.
        """
        attr = getattr(self, item)
        if item in self._sub_configs.keys():
            return {key: attr[key] for key, _ in attr.items()}
        return getattr(self, item)

    def items(self):
        """Implements the dictionary like self.items() method.

        Returns:
            list: List of (key, value) tuples of the class attributes.
        """
        return [(key, self[key]) for key in vars(self)]

    def get(self, item, default=None):
        """Implements the dictionary like get method with a default value.

        Parameters:
            item (str): Label of the attribute to return.
            default: Default value, if the attribute is not found.

        Returns:
            Attribute or the default value.
        """
        try:
            return self[item]
        except AttributeError:
            return default

    @classmethod
    def register(cls, label):
        """Registeres sub configs with a specific label."""
        def decorator(config):
            if label in cls._sub_configs:
                raise KeyError(f'registering duplicate label {label} for Interface')
            cls._sub_configs[label] = config
            return config
        return decorator


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
            - runner
            - interface
            - pre
            - post
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
        self.variables = defaults.variables.copy()
        self.input = {}
        self.output = {}
        self.independent = {}
        self.files = defaults.files.copy()

        self.update(**entries)  # Update the attributes with given entries.
        self.create_subconfigs(**entries)
        self.process_entries()  # Postprocess the attributes to standardize different user entries.

    def process_entries(self):
        """Sets absolute paths, creates variables and delegates to the sub configs."""
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
        """Creates a configuration class from a .yaml or .py file."""

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


@BaseConfig.register("run")
class RunConfig(AbstractConfig):
    """Run configuration with the following sub classes:
        - runner
            - local
            - slurm
        - interface
            - memmap
            - zeromq
        - pre
            - template
        - post
            - json
            - numpytxt
            - hdf5

    A default sub class which just updates the entries from a user input is also implemented and used if the
    class from the user input is not found.

    Custom config classes can also be registered, e.g. as a custom runner:

    .. code-block:: python

        @RunnerConfig.register("custom")
        class CustomRunner(LocalRunnerConfig):
            def process_entries(self, base_config):
                # do something else than the usual LocalRunnerConfig
                pass

    Default values from the global profit.defaults.py file are loaded.
    """
    _sub_configs = {}

    def __init__(self, **entries):
        self.set_defaults(defaults.run)
        self.update(**entries)

        for key, sub_config in self._sub_configs.items():
            attr = getattr(self, key.lower())
            if isinstance(attr, str):
                attr = {'class': attr}
            try:
                setattr(self, key.lower(), sub_config._sub_configs[attr['class']](**attr))
            except KeyError:
                setattr(self, key.lower(), sub_config._sub_configs['default'](**attr))

    def process_entries(self, base_config):
        """Set 'include' and paths and process entries of sub configs."""
        from profit.util import load_includes

        if isinstance(self.include, str):
            self.include = [self.include]

        for p, include_path in enumerate(self.include):
            if not path.isabs(include_path):
                self.include[p] = path.abspath(path.join(base_config.base_dir, include_path))
        load_includes(self.include)

        if not path.isabs(self.log_path):
            self.log_path = path.abspath(path.join(base_config.base_dir, self.log_path))

        for key in self._sub_configs:
            getattr(self, key.lower()).process_entries(base_config)


@RunConfig.register("runner")
class RunnerConfig(AbstractConfig):
    """Base Runner config."""
    _sub_configs = {}

    def __init__(self, **entries):
        for key, value in self._sub_configs.items():
            if value.__name__ == self.__class__.__name__:
                self.set_defaults(getattr(defaults, f"run_runner_{key}"))
                self.update(**entries)
                break


@RunnerConfig.register("local")
class LocalRunnerConfig(RunnerConfig):
    """
    Example:
        .. code-block:: yaml

            class: local
            parallel: all   # maximum number of simultaneous runs (for spawn array)
            sleep: 0        # number of seconds to sleep while polling
            fork: true      # whether to spawn the worker via forking instead of a subprocess (via a shell)
    """
    pass


@RunnerConfig.register("slurm")
class SlurmRunnerConfig(RunnerConfig):
    """
    Example:
        .. code-block:: yaml

               class: slurm
               parallel: null      # maximum number of simultaneous runs (for spawn array)
               sleep: 0            # number of seconds to sleep while (internally) polling
               poll: 60            # number of seconds between external polls (to catch failed runs), use with care!
               path: slurm.bash    # the path to the generated batch script (relative to the base directory)
               custom: false       # whether a custom batch script is already provided at 'path'
               prefix: srun        # prefix for the command
               OpenMP: false       # whether to set OMP_NUM_THREADS and OMP_PLACES
               cpus: 1             # number of cpus (including hardware threads) to use (may specify 'all')
               options:            # (long) options to be passed to slurm: e.g. time, mem-per-cpu, account, constraint
                   job-name: profit
    """

    def process_entries(self, base_config):
        """Converts paths to absolute and check type of 'cpus'"""
        # Convert path to absolute path
        if not path.isabs(self.path):
            self.path = path.abspath(path.join(base_config.base_dir, self.path))
        # Check type of 'cpus'
        if (type(self.cpus) is not int or self.cpus < 1) and self.cpus != 'all':
            raise ValueError(f'config option "cpus" may only be a positive integer or "all" and not {self.cpus}')


@RunConfig.register("interface")
class InterfaceConfig(AbstractConfig):
    """Base runner interface config."""
    _sub_configs = {}

    def __init__(self, **entries):
        for key, value in self._sub_configs.items():
            if value.__name__ == self.__class__.__name__:
                self.set_defaults(getattr(defaults, f"run_interface_{key}"))
                self.update(**entries)
                break


@InterfaceConfig.register("memmap")
class MemmapInterfaceConfig(InterfaceConfig):
    """
    Example:
        .. code-block:: yaml

            class: memmap
            path: interface.npy     # path to memory mapped interface file, relative to base directory
    """

    def process_entries(self, base_config):
        """Converts 'path' to absolute."""
        if not path.isabs(self.path):
            self.path = path.abspath(path.join(base_config.base_dir, self.path))


@InterfaceConfig.register("zeromq")
class ZeroMQInterfaceConfig(InterfaceConfig):
    """
    Example:
        .. code-block:: yaml

            class: zeromq
            transport: tcp      # transport system used by zeromq
            port: 9000          # port for the interface
            address: null       # override bind address used by zeromq
            connect: null       # override connect address used by zeromq
            timeout: 2500       # zeromq polling timeout, in ms
            retries: 3          # number of zeromq connection retries
            retry-sleep: 1      # sleep between retries, in s
    """
    pass


@RunConfig.register("pre")
class PreConfig(AbstractConfig):
    """Base config for preprocessors."""
    _sub_configs = {}

    def __init__(self, **entries):
        for key, value in self._sub_configs.items():
            if value.__name__ == self.__class__.__name__:
                self.set_defaults(getattr(defaults, f"run_pre_{key}"))
                self.update(**entries)
                break


@PreConfig.register("template")
class TemplatePreConfig(PreConfig):
    """
    Example:
        .. code-block:: yaml

            class: template
            path: template      # directory to copy from, relative to base directory
            param_files: null   # files in template which contain placeholders for variables, null means all files
                                # can be a filename or a list of filenames
    """

    def process_entries(self, base_config):
        """Convert 'path' to absolute and set 'param_files'."""
        if not path.isabs(self.path):
            self.path = path.abspath(path.join(base_config.base_dir, self.path))

        if isinstance(self.param_files, str):
            self.param_files = [self.param_files]


@RunConfig.register("post")
class PostConfig(AbstractConfig):
    """Base class for postprocessor configs."""
    _sub_configs = {}

    def __init__(self, **entries):
        for key, value in self._sub_configs.items():
            if value.__name__ == self.__class__.__name__:
                self.set_defaults(getattr(defaults, f"run_post_{key}"))
                self.update(**entries)
                break


@PostConfig.register("json")
class JsonPostConfig(PostConfig):
    """
    Example:
        .. code-block:: yaml

            class: json
            path: stdout    # file to read from, relative to the run directory
    """
    pass


@PostConfig.register("numpytxt")
class NumpytxtPostConfig(PostConfig):
    """
    Example:
        .. code-block:: yaml

            class: numpytxt
            path: stdout    # file to read from, relative to the run directory
            names: "f g"    # list or string of output variables in order, default read from config/variables
            options:        # options which are passed on to numpy.genfromtxt() (fname & dtype are used internally)
                deletechars: ""
    """

    def process_entries(self, base_config):
        """Sets the included names of variables. The Keyword 'all' includes all variables."""
        if isinstance(self.names, str):
            self.names = list(base_config.output.keys()) if self.names == 'all' else self.names.split()


@PostConfig.register("hdf5")
class HDF5PostConfig(PostConfig):
    """
    Example:
        .. code-block:: yaml

            class: hdf5
            path: output.hdf5   # file to read from, relative to the run directory
    """
    pass


@RunnerConfig.register("default")
@InterfaceConfig.register("default")
@PreConfig.register("default")
@PostConfig.register("default")
class DefaultConfig(AbstractConfig):
    """Default config for all run sub configs which just updates the attributes with user entries."""

    def __init__(self, **entries):
        self.update(**entries)


@BaseConfig.register("fit")
class FitConfig(AbstractConfig):
    """Configuration for the surrogate and encoder. Currently, the only sub config is for the GaussianProcess classes."""

    def __init__(self, **entries):
        from profit.sur import Surrogate
        from profit.sur.gaussian_process import GaussianProcess
        self.set_defaults(defaults.fit)

        if issubclass(Surrogate._surrogates[self.surrogate], GaussianProcess):
            self.set_defaults(defaults.fit_gaussian_process)

        self.update(**entries)

    def process_entries(self, base_config):
        """Set 'load' and 'save' as well as the encoder."""
        for mode_str in ('save', 'load'):
            filepath = getattr(self, mode_str)
            if filepath:
                if self.surrogate not in filepath:
                    filepath = filepath.rsplit('.', 1)
                    filepath = ''.join(filepath[:-1]) + f'_{self.surrogate}.' + filepath[-1]
                setattr(self, mode_str, path.abspath(path.join(base_config.base_dir, filepath)))

        if self.load:
            self.save = False

        for enc in getattr(self, 'encoder'):
            cols = enc[1]
            out = enc[2]
            if isinstance(cols, str):
                variables = getattr(base_config, 'output' if out else 'input')
                if cols.lower() == 'all':
                    enc[1] = list(range(len(variables)))
                elif cols.lower() in [v['kind'] for v in variables.values()]:
                    enc[1] = [idx for idx, v in enumerate(variables.values()) if v['kind'].lower() == cols.lower()]
                else:
                    enc[1] = []


@BaseConfig.register("active_learning")
class ALConfig(AbstractConfig):
    """Active learning configuration."""

    def __init__(self, **entries):
        self.set_defaults(defaults.active_learning)
        self.update(**entries)


@BaseConfig.register("ui")
class UIConfig(AbstractConfig):
    """Configuration for the Graphical User Interface."""

    def __init__(self, **entries):
        self.plot = defaults.ui['plot']
        self.update(**entries)
