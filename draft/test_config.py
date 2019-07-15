"""
Created: Tue May 28 15:40:29 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import yaml


with open('../examples/mockup/suruq.yaml', 'r') as document:
    config = yaml.load(document)

print(yaml.dump(config))
