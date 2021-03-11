"""This module contains runners for domain codes.

This is used to run domain code via Python functions or
local and distributed systems.
"""
import os
import sys
from importlib.util import spec_from_file_location, module_from_spec
import logging


def load_includes(paths):
    logger = logging.getLogger(__name__)
    for path in paths:
        name = f"profit_include_{os.path.basename(path).split('.')[0]}"
        try:
            spec = spec_from_file_location(name, path)
        except FileNotFoundError:
            logger.error(f'could not find {path} to include')
            continue
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
