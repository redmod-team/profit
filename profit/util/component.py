""" Component Base Class

abstract base class to register subclasses
"""

from abc import ABC, abstractmethod
import logging


class Component(ABC):
    _components = {}
    component = "Component"

    @abstractmethod
    def __init__(self):
        pass

    def __init_subclass__(cls, label=None):
        """This method is called when a class is subclassed.

        Register a new (sub-)component"""
        if cls.component == "Component":
            if label is None:
                label = cls.__name__
            # register with parent
            cls.register(label)(cls)
            # set up new registry for subcomponents
            cls._components = {}
            cls.component = label
            cls.labels = cls._components.keys()
        elif label is not None:
            cls.register(label)(cls)

    @classmethod
    def register(cls, label):
        """Decorator to register new subcomponents.

        This will allow access to the subcomponent via ``Component[label]``.
        Internally the subcomponents are stored in ``_components``.
        Warning: this is totally unrelated to ABC.register
        Not necessary when a Component is subclassed, __init_subclass__ is used instead
        """

        def decorator(subcls):
            if label in cls._components:
                logging.warning(
                    f"replacing {cls._components[label]} with {cls} for label '{label}' ({cls.component})."
                )
            cls._components[label] = subcls
            subcls.label = label
            return subcls

        return decorator

    def __class_getitem__(cls, label):
        """Returns the subcomponent."""
        return cls._components[label]
