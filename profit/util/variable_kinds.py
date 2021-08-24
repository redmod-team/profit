from .halton import halton as _halton_base
from .util import check_ndim
import numpy as np

# TODO: Sample all variables from halton.
EXCLUDE_FROM_HALTON = ('output', 'constant', 'uniform', 'loguniform', 'normal', 'linear', 'independent')


def halton(size=(1, 1)):
    if isinstance(size, (tuple, list, np.ndarray)):
        return _halton_base(*size)
    else:
        return check_ndim(_halton_base(size, 1))


def uniform(start=0, end=1, size=None):
    return check_ndim(start + np.random.random(size) * (end - start))


def loguniform(start=1e-6, end=1, size=None):
    return check_ndim(start * np.exp((np.log(end) - np.log(start)) * np.random.random(size)))


def normal(mu=0, std=1, size=None):
    return check_ndim(np.random.normal(mu, std, size))


def linear(start=0, end=1, step=1, size=None):
    return check_ndim(np.arange(start, end, step))


def independent(start=0, end=1, step=1, size=None):
    return linear(start, end, step)


def activelearning(start=0, end=1, size=None):
    return check_ndim(np.full(size, np.nan))


def constant(value=0, size=None):
    return check_ndim(np.full(size, value))


class VariableGroup:
    """Table of input, output and independent variables.

    Parameters:
        samples (int): Samples of the training data.

    Attributes:
        samples (int): Samples of the training data.
        list (list): List of all variables in the order of the user entry.
    """

    def __init__(self, samples):
        self.samples = samples
        self.list = []

    @property
    def all(self):
        """
        Returns:
            View on all variables.

        Not working yet for vector output.
        """
        values = np.hstack([v.value for v in self.list])
        dtypes = [(v.name, v.dtype) for v in self.list]
        return values.view(dtype=dtypes)

    @property
    def as_dict(self):
        """
        Returns:
            View of all variables as a dictionary.
        """
        input_dict = {k: v.as_dict() for k, v in self.input_dict.items()}
        independent_dict = {v.name: v.as_dict() for v in self.list if v.kind.lower() == 'independent'}
        output_dict = {k: v.as_dict() for k, v in self.output_dict.items()}
        return {**input_dict, **independent_dict, **output_dict}

    @property
    def input(self):
        """
        Returns:
            View of the input variables only. Also excluded are independent variables.
        """
        return np.hstack([v.value for v in self.list if not any(s in v.kind.lower() for s in ('output', 'independent'))])

    @property
    def named_input(self):
        """
        Returns:
            Ndarray with dtype of the input variables.
        """
        dtypes = [(v.name, v.dtype) for v in self.list if not any(s in v.kind.lower() for s in ('output', 'independent'))]
        return self.input.view(dtype=dtypes)

    @property
    def input_dict(self):
        """
        Returns:
            Dictionary of the input variables.
        """
        return {v.name: v for v in self.list if not any(s in v.kind.lower() for s in ('output', 'independent'))}

    @property
    def input_list(self):
        """
        Returns:
            List of input variables without independent variables.
        """
        return [v for v in self.list if v.__class__ is InputVariable]

    @property
    def output(self):
        """
        Returns:
            View on the output variables only.
        """
        return np.hstack([v.value for v in self.list if 'output' in v.kind.lower()])

    @property
    def named_output(self):
        """
        Returns:
            Ndarray with dtype of the output variables.
        """
        dtypes = [(v.name, v.dtype) for v in self.list if 'output' in v.kind.lower()]
        return self.output.view(dtype=dtypes)

    @property
    def output_dict(self):
        """
        Returns:
            Dictionary of the output variables.
        """
        return {v.name: v for v in self.list if 'output' in v.kind.lower()}

    def __getitem__(self, item):
        """Implements dict like behavior to get a variable by its identifier or index.

        Parameters:
            item (int/str): Index or label of variable.
        Returns:
            Variable.
        """
        if isinstance(item, str):
            item = [i for i, v in enumerate(self.list) if item == v.name]
            if len(item) > 0:
                item = item[0]
        return self.list[item]

    def add(self, variables):
        """Adds a single or a list of variables to the table.
        If a list is added, a common n-D halton sequence is generated and the variables are transformed
        according to their distribution.

        Parameters:
            variables (Variable/list): Variable(s) to add.
        """
        if not isinstance(variables, (list, tuple)):
            variables = [variables]

        for v in variables:
            if isinstance(v, Variable):
                self.list.append(v)
            elif isinstance(v, dict):
                self.list.append(Variable.create(**v))
            else:
                raise NotImplementedError

        self.generate_from_halton()
        for v in self.list:
            if any(e in v.kind.lower() for e in EXCLUDE_FROM_HALTON) and 'output' not in v.kind.lower():
                v.generate_values()
        self.resolve_dependent()

    def delete_variable(self, columns):
        """Deletes one or more variables from the table.

        Parameters:
            columns (int/list): Columns of the table to remove.
        """
        if not isinstance(columns, (list, tuple)):
            columns = [columns]
        for col in columns:
            if isinstance(col, str):
                col = [i for i, v in enumerate(self.list) if v.name == col][0]
            self.list.pop(col)

    def delete_sample(self, rows):
        """Deletes one or more rows of the table.

        Parameters:
            rows (int/list): Rows to delete.
        """
        if not isinstance(rows, (list, tuple)):
            rows = [rows]
        for v in self.list:
            v.value = np.delete(v.value, rows, axis=0)

    def generate_from_halton(self):
        """Generates a common halton sequence for all variables where this is possible and transforms them
        according to their distribution."""
        halton_variables = [v for v in self.list if v.kind.lower() not in EXCLUDE_FROM_HALTON]
        if halton_variables:
            nd_halton_seq = halton((self.samples, len(halton_variables)))
            for idx, v in enumerate(halton_variables):
                v.generate_values(nd_halton_seq[:, idx])

    def resolve_dependent(self):
        """Create a Variable instance for the independent variables of vector outputs."""
        output = [v for v in self.list if 'output' in v.kind.lower() and v.dependent]
        for o in output:
            dvars = []
            for d in o.dependent:
                dv = [v for v in self.list if v.name == d][0]
                o.size = (o.size[0], dv.value.shape[0])
                o.value = np.full(o.size, np.nan)
                dvars.append(dv)
            o.dependent = dvars


class Variable:
    """Base class for a single variable.
    To create input, independent and output variables, use the cls.create() or cls.create_from_str() methods.

    Attributes:
        name (str): Name of the variable.
        kind (str): Distribution for input variables, 'Output' or 'Independent'.
        size (tuple): Size as (nsamples, ndim).
        value (ndarray): Data.
        dtype (dtype): Datatype.
    """

    def __init__(self, name, kind, size, value=None, dtype=np.float64):
        self.name = name
        self.kind = kind
        self.size = size
        self.value = value if value is not None else np.empty(size)
        assert self.value.shape == size
        self.dtype = dtype

    @property
    def named_value(self):
        """
        Returns:
            Ndarray with dtype.
        """
        return np.array(self.value, dtype=[(self.name, self.dtype)])

    @classmethod
    def create_from_str(cls, name, size, v_str):
        """Creates a Variable instance from a string. E.g. 'Uniform(3.4, 7.8)'

        Parameters:
            name (str): Name of the variable.
            size (tuple): Size as (nsamples, ndim).
            v_str (str): String from which the variable is constructed.

        Returns:
            Variable.
        """
        from re import split

        def try_parse(s):
            funcs = [int, float]
            for f in funcs:
                try:
                    return f(s)
                except ValueError:
                    pass
            return s

        parsed = split('[()]', v_str)
        kind = parsed[0]
        args = parsed[1] if len(parsed) >= 2 else ''
        entries = tuple(try_parse(a) for a in args.split(',')) if args != '' else tuple()

        if isinstance(try_parse(v_str), (int, float)):  # PyYaml has problems parsing scientific notation
            kind = 'Constant'
            entries = (try_parse(v_str),)

        v_dict = {'name': name, 'kind': kind, 'size': size, 'entries': entries}
        return cls.create(**v_dict)

    @classmethod
    def create(cls, name, kind, size, entries=(0, 1), value=None, dtype=np.float64):
        """Directly creates a variable from keyword entries.

        Parameters:
            name (str): Name of the variable.
            kind (str): Distribution of input variables, 'Output' or 'Independent'.
            size (tuple): Size as (nsamples, ndim).
            entries (tuple/str): Tuple for input variables used to create them from their distribution,
                or str for output variables for their dependent variable.
            value (ndarray): Data.
            dtype (dtype): Datatype.

        Returns:
            Variable.
        """
        if 'output' in kind.lower():
            dependent = entries if len(entries) > 0 and isinstance(entries[0], (str, Variable)) else ()
            return OutputVariable(name, kind, size, dependent, value, dtype)
        else:
            constraints = entries if entries else (0, 1)
            if 'independent' in kind.lower():
                return IndependentVariable(name, kind, size, constraints, value, dtype)
            return InputVariable(name, kind, size, constraints, value, dtype)

    def as_dict(self):
        """Dictionary of the variable attributes."""
        return {k: v for k, v in vars(self).items()}


class InputVariable(Variable):
    """Sub class for input variables."""

    def __init__(self, name, kind, size, entries=(0, 1), value=None, dtype=np.float64):
        super().__init__(name, kind, size, value, dtype)
        self.constraints = entries

    def generate_values(self, halton_seq=None):
        if halton_seq is None or self.kind.lower() == 'activelearning':
            self.value = globals().get(self.kind.lower())(*self.constraints, size=self.size)
        else:
            self.value = check_ndim((self.constraints[1] - self.constraints[0]) * halton_seq + self.constraints[0])

    def as_dict(self):
        return {k if k != 'constraints' else 'entries': v for k, v in vars(self).items()}


class IndependentVariable(InputVariable):
    """Sub class for independent variables."""

    def __init__(self, name, kind, size, entries=(0, 1), value=None, dtype=np.float64):
        super().__init__(name, kind, size, entries, value, dtype)
        if value is None:
            self.generate_values()
            self.size = self.value.shape


class OutputVariable(Variable):
    """Sub class for output variables."""

    def __init__(self, name, kind, size, entries=(), value=None, dtype=np.float64):
        super().__init__(name, kind, size, value, dtype)
        self.dependent = entries
        self.value = value if value is not None else np.full(self.size, np.nan)

    def as_dict(self):
        return {k if k != 'dependent' else 'entries': v if k != 'dependent' else [vi.as_dict() for vi in v] for k, v in vars(self).items()}
