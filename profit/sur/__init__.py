from .sur import Surrogate

# Import all surrogate implementations to register them
# This ensures Surrogate.labels is populated when users import Surrogate
from .gp import GaussianProcess, GPSurrogate, SklearnGPSurrogate

# GPyTorch is optional - only import if available
try:
    from .gp import GPyTorchSurrogate, MultiOutputGPyTorchSurrogate
except ImportError:
    pass  # GPyTorch surrogates not available if gpytorch is not installed
