from .gaussian_process import GaussianProcess
from .custom_surrogate import GPSurrogate
from .sklearn_surrogate import SklearnGPSurrogate

# GPy is optional - only import if available
try:
    from .gpy_surrogate import GPySurrogate
except ImportError:
    pass  # GPySurrogate will not be available if GPy is not installed
