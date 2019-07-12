import os
from os import path, mkdir, walk
import yaml
from collections import OrderedDict

# from https://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

yaml.add_representer(OrderedDict, represent_ordereddict)

# from https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_constructor(loader, node):
    return collections.OrderedDict(loader.construct_pairs(node))

yaml.add_constructor(_mapping_tag, dict_constructor)


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

class NewConfig(dict):

  def __init__(self,**entries):
    self['base_dir'] = os.getcwd()
    self['template_dir'] = path.join(base_dir, 'template')
    self['run_dir'] = path.join(base_dir, 'run')
    self['command'] = None
    self['runner_backend'] = None
    self['uq']={}
    self.update(entries)
  
  def write_yaml(self,filename='suruq.yaml'):
    dumpdict = dict(self)
    self.remove_nones(dumpdict)
    with open(filename,'w') as file:
      yaml.dump(dumpdict,file,default_flow_style=False)

  def load(self,filename='suruq.yaml'):
    with open(filename) as f:
      entries = yaml.safe_load(f)
    self.update(entries)
  
  def remove_nones(self,config=None):
      if config==None: config=self.__dict__
      for key in list(config):
        if type(config[key]) is dict:
          self.remove_nones(config[key])
        #elif (type(config[key]) is not list) and (config[key] is None):
        else:
          if config[key] is None:
            del config[key]


    

class Config(yaml.YAMLObject):
  
  def __init__(self, **entries):
    self.base_dir = os.getcwd()
    self.template_dir = path.join(base_dir, 'template')
    self.run_dir = path.join(base_dir, 'run')
    self.uq = OrderedDict()
    self.__dict__.update(entries)

  def write_yaml(self,filename='suruq.yaml'):
    with open(filename,'w') as file:
      yaml.dump(self,file)

  @classmethod
  def load(cls, filename):
    with open(filename) as f:
      data = ordered_load(f, yaml.SafeLoader)
    #print(yaml.dump(data))
    return cls(**data)
