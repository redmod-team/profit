from .gaussian_process import GaussianProcess
from .custom_surrogate import GPSurrogate
from .sklearn_surrogate import SklearnGPSurrogate

# GPyTorch is recommended - only import if available
try:
    from .gpytorch_surrogate import GPyTorchSurrogate, MultiOutputGPyTorchSurrogate
except ImportError:
    pass  # GPyTorchSurrogate will not be available if gpytorch is not installed
