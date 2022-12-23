from abc import ABC


class CustomABC(ABC):

    labels = {}

    @classmethod
    def get_label(cls):
        """Returns the string label of a class object."""
        for label, item in cls.labels.items():
            if item == cls:
                return label
        raise NotImplementedError(f"Class {cls} is not implemented.")

    @classmethod
    def register(cls, label):
        """Decorator to register new classes."""

        def decorator(obj):
            if label in cls.labels:
                print(f"registering duplicate label '{label}' for {cls.__name__}.")
                # raise KeyError(f"registering duplicate label '{label}' for {cls.__name__}.")
            cls.labels[label] = obj
            return obj

        return decorator

    def __class_getitem__(cls, item):
        """Returns the child."""
        if item is None:
            return cls
        return cls.labels[item]
