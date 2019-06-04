import os
from os import path, mkdir, walk
import yaml
from collections import OrderedDict

# from https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

base_dir = os.getcwd()

eval_points = None
dont_copy = None
param_files = None

class Config(yaml.YAMLObject):
  
  def __init__(self, **entries):
    self.base_dir = os.getcwd()
    self.template_dir = path.join(base_dir, 'template')
    self.run_dir = path.join(base_dir, 'run')
    self.__dict__.update(entries)
    
  @classmethod
  def load(cls, filename):
    with open(filename) as f:
      data = ordered_load(f, yaml.SafeLoader)
    print(yaml.dump(data))
    return cls(**data)