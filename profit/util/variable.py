from profit.util.halton import halton as _halton_base
from profit.util import check_ndim
import numpy as np
from profit.util.base_class import CustomABC


# TODO: Sample all variables from halton.
EXCLUDE_FROM_HALTON = (
    "output",
    "constant",
    "uniform",
    "loguniform",
    "normal",
    "linear",
    "independent",
)


def halton(size=(1, 1)):
    if isinstance(size, (tuple, list, np.ndarray)):
        return _halton_base(*size)
    else:
        return check_ndim(_halton_base(size, 1))


def uniform(start=0, end=1, size=None):
    return check_ndim(start + np.random.random(size) * (end - start))


def loguniform(start=1e-6, end=1, size=None):
    return check_ndim(
        start * np.exp((np.log(end) - np.log(start)) * np.random.random(size))
    )


def normal(mu=0, std=1, size=None):
    return check_ndim(np.random.normal(mu, std, size))


def linear(start=0, end=1, size=1):
    return check_ndim(np.linspace(start, end, size))


def independent(start=0, end=1, size=1):
    return linear(start, end, size)


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
        independent_dict = {
            v.name: v.as_dict() for v in self.list if v.__class__ == IndependentVariable
        }
        output_dict = {k: v.as_dict() for k, v in self.output_dict.items()}
        return {**input_dict, **independent_dict, **output_dict}

    @property
    def input(self):
        """
        Returns:
            View of the input variables only. Also excluded are independent variables.
        """
        return np.hstack(
            [
                v.value
                for v in self.list
                if v.__class__ in (InputVariable, ActiveLearningVariable)
            ]
        )

    @property
    def named_input(self):
        """
        Returns:
            Ndarray with dtype of the input variables.
        """
        dtypes = [
            (v.name, v.dtype)
            for v in self.list
            if v.__class__ in (InputVariable, ActiveLearningVariable)
        ]
        return np.rec.fromarrays(
            [
                v.value
                for v in self.list
                if v.__class__ in (InputVariable, ActiveLearningVariable)
            ],
            dtype=dtypes,
        )

    @property
    def input_dict(self):
        """
        Returns:
            Dictionary of the input variables.
        """
        return {
            v.name: v
            for v in self.list
            if v.__class__ in (InputVariable, ActiveLearningVariable)
        }

    @property
    def input_list(self):
        """
        Returns:
            List of input variables without independent variables.
        """
        return [
            v
            for v in self.list
            if v.__class__ in (InputVariable, ActiveLearningVariable)
        ]

    @property
    def kind_dict(self):
        kinds = {}
        for v in self.input_list:
            if v["kind"] in kinds:
                kinds[v["kind"]].append(v)
            else:
                kinds[v["kind"]] = [v]
        return kinds

    @property
    def output(self):
        """
        Returns:
            View on the output variables only.
        """
        return np.hstack([v.value for v in self.list if v.__class__ == OutputVariable])

    @property
    def named_output(self):
        """
        Returns:
            Ndarray with dtype of the output variables.
        """
        dtypes = [(v.name, v.dtype) for v in self.list if v.__class__ == OutputVariable]
        return self.output.view(dtype=dtypes)

    @property
    def formatted_output(self):
        dtype = []
        columns = {}
        # prepare dtype
        for key in self.output_dict:
            spec = self[key]
            if spec.size[-1] == 1:
                dtype.append((key, spec.dtype))
                columns[key] = [key]
            else:
                ranges = []
                columns[key] = []
                for dep in spec.dependent:
                    ranges.append(dep.value)
                meshes = [m.flatten() for m in np.meshgrid(*ranges)]
                for i in range(meshes[0].size):
                    name = key + "(" + ", ".join([f"{m[i]}" for m in meshes]) + ")"
                    dtype.append((name, spec.dtype))
                    columns[key].append(name)
        # fill data
        output = np.zeros(len(self.named_output), dtype=dtype)
        for key, spec in self.output_dict.items():
            if spec.size[-1] == 1:
                output[key] = self.named_output[key].flatten()
            else:
                for i, values in enumerate(self.named_output):
                    output[columns[key]][i] = tuple(values[key])
        output = output.reshape(-1, 1)
        return output

    @property
    def output_dict(self):
        """
        Returns:
            Dictionary of the output variables.
        """
        return {v.name: v for v in self.list if v.__class__ == OutputVariable}

    @property
    def output_list(self):
        """
        Returns:
            List of output variables.
        """
        return [v for v in self.list if v.__class__ == OutputVariable]

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
            if (
                any(e in v.kind.lower() for e in EXCLUDE_FROM_HALTON)
                and v.__class__ != OutputVariable
            ):
                v.generate_values()

        for v in self.list:
            if v.__class__ == OutputVariable:
                dep = [v if isinstance(v, str) else v["name"] for v in v.dependent]
                ind = [
                    i
                    for i in self.list
                    if i.__class__ == IndependentVariable and i.name in dep
                ]
                if ind:
                    v.resolve_dependent(ind)

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
        halton_variables = [
            v for v in self.list if v.kind.lower() not in EXCLUDE_FROM_HALTON
        ]
        if halton_variables:
            nd_halton_seq = halton((self.samples, len(halton_variables)))
            for idx, v in enumerate(halton_variables):
                v.generate_values(nd_halton_seq[:, idx])


class Variable(CustomABC):
    """Base class for a single variable.
    To create input, independent and output variables, use the cls.create() or cls.create_from_str() methods.

    Attributes:
        name (str): Name of the variable.
        kind (str): Distribution for input variables, 'Output' or 'Independent'.
        size (tuple): Size as (nsamples, ndim).
        value (ndarray): Data.
        dtype (dtype): Datatype.
    """

    labels = {}

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

        if isinstance(try_parse(v_str), (int, float)):
            v_str = "Constant({})".format(try_parse(v_str))

        parsed = split("[()]", v_str)
        kind = parsed[0]
        args = parsed[1] if len(parsed) >= 2 else ""
        entries = (
            tuple(try_parse(a) for a in args.split(",")) if args != "" else tuple()
        )
        entries_dict = (
            cls.labels.get(kind.lower(), cls.labels["input"]).parse_entries(entries)
            if len(entries) > 0
            else {}
        )

        dtype = type(entries[0]) if kind.lower() == "constant" else np.float64

        v_dict = {"name": name, "kind": kind, "size": size, "dtype": dtype}
        v_dict = {**v_dict, **entries_dict}
        return cls.create(**v_dict)

    @classmethod
    def create(cls, name, kind, size, value=None, dtype=np.float64, **kwargs):
        """Directly creates a variable from keyword entries.

        Parameters:
            name (str): Name of the variable.
            kind (str): Distribution of input variables, 'Output' or 'Independent'.
            size (tuple): Size as (nsamples, ndim).
            kwargs (tuple/str): Keyword arguments depending on the sub variables.
                E.g. constraints for input variables, a search distribution for active learning variables or
                dependent variables of outputs.
            value (ndarray): Data.
            dtype (dtype): Datatype.

        Returns:
            Variable.
        """
        if isinstance(dtype, str) and kind.lower() != "constant":
            dtype = np.dtype(dtype).type
        return cls.labels.get(kind.lower(), cls.labels["input"])(
            name=name, kind=kind, size=size, value=value, dtype=dtype, **kwargs
        )

    def as_dict(self):
        """Dictionary of the variable attributes."""
        return {k: v for k, v in vars(self).items()}

    def __getitem__(self, item):
        """Implement dict like behavior to get an attribute by using square brackets.

        Paramters:
            item (str)

        Returns:
            Attribute
        """
        return getattr(self, item)


@Variable.register("input")
class InputVariable(Variable):
    """Sub class for input variables."""

    def __init__(
        self, name, kind, size, constraints=(0, 1), value=None, dtype=np.float64
    ):
        super().__init__(name, kind, size, value, dtype)
        self.constraints = constraints

    def generate_values(self, halton_seq=None):
        if halton_seq is None:
            if len(self.constraints) == 3:
                self.value = (
                    globals()
                    .get(self.kind.lower())(*self.constraints)
                    .astype(self.dtype)
                )
            else:
                self.value = (
                    globals()
                    .get(self.kind.lower())(*self.constraints, size=self.size)
                    .astype(self.dtype)
                )
        else:
            self.value = check_ndim(
                (self.constraints[1] - self.constraints[0]) * halton_seq
                + self.constraints[0]
            ).astype(self.dtype)

    @classmethod
    def parse_entries(cls, entries):
        return {"constraints": entries}

    def create_Xpred(self, size, spacing=None):
        """creates an array of suitably spaced X-values for prediction

        spacing (shape: size) can be used to override the default linear spacing"""
        assert spacing is None or spacing.shape == size
        if not isinstance(size, tuple):
            size = (size, 1)
        if spacing is None:
            spacing = np.linspace(0, 1, size[0]).reshape(size)
        if "log" in self.kind.lower():
            return self.constraints[0] * np.exp(
                (np.log(self.constraints[1]) - np.log(self.constraints[0])) * spacing
            )
        elif "constant" in self.kind.lower():
            return np.full(size, self.value[0])
        else:
            return (
                self.constraints[0]
                + (self.constraints[1] - self.constraints[0]) * spacing
            )


@Variable.register("independent")
class IndependentVariable(InputVariable):
    """Sub class for independent variables."""

    def __init__(
        self, name, kind, size, constraints=(0, 1), value=None, dtype=np.float64
    ):
        super().__init__(name, kind, size, constraints, value, dtype)
        if value is None:
            self.generate_values()
            self.size = self.value.shape


@Variable.register("activelearning")
class ActiveLearningVariable(InputVariable):
    """Sub class for active learning variables."""

    def __init__(
        self,
        name,
        kind,
        size,
        distr="uniform",
        constraints=(0, 1),
        value=None,
        dtype=np.float64,
    ):
        if value is None:
            value = np.full(size, np.nan)
        super().__init__(name, kind, size, constraints, value, dtype)
        self.distr = distr

    @classmethod
    def parse_entries(cls, entries):
        return {
            "constraints": entries[:2],
            "distr": entries[2].strip() if len(entries) == 3 else "uniform",
        }

    def generate_values(self, halton_seq=None):
        return check_ndim(np.full(self.size, np.nan))

    def create_Xpred(self, size, spacing=None):
        """creates an array of suitably spaced X-values for prediction

        spacing (shape: size) can be used to override the default linear spacing"""
        assert spacing is None or spacing.shape == size
        if not isinstance(size, tuple):
            size = (size, 1)
        if spacing is None:
            spacing = np.linspace(0, 1, size[0]).reshape(size)
        if self.distr.lower() == "log":
            return self.constraints[0] * np.exp(
                (np.log(self.constraints[1]) - np.log(self.constraints[0])) * spacing
            )
        else:
            return (
                self.constraints[0]
                + (self.constraints[1] - self.constraints[0]) * spacing
            )


@Variable.register("output")
class OutputVariable(Variable):
    """Sub class for output variables."""

    def __init__(self, name, kind, size, dependent=(), value=None, dtype=np.float64):
        super().__init__(name, kind, size, value, dtype)
        if not isinstance(dependent, (list, tuple)):
            dependent = [dependent]
        self.dependent = dependent
        self.value = value if value is not None else np.full(self.size, np.nan)

    @classmethod
    def parse_entries(cls, entries):
        return {"dependent": entries}

    def resolve_dependent(self, ind):
        """Create a :class:`.Variable` instance for the independent variables of vector outputs
        and set :attr:`dependent`.

        Parameters:
            ind (profit.util.variable.IndependentVariable or list[profit.util.variable.IndependentVariable]):
                Independent variables.
        """
        if not isinstance(ind, (list, tuple)):
            ind = [ind]

        dvars = []
        for d in self.dependent:
            if isinstance(d, str):
                d = {"name": d}
            dv = [v for v in ind if v.name == d["name"]][0]
            self.size = (self.size[0], dv.value.shape[0])
            self.value = np.full(self.size, np.nan)
            dvars.append(dv)
        self.dependent = dvars

    def as_dict(self):
        return {
            k: v if k != "dependent" else [vi.as_dict() for vi in v]
            for k, v in vars(self).items()
        }
