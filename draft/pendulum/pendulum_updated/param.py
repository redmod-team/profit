import numpy as np
import yaml

with open('config.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

Nm = data['Nm'] #mapping time
N = data['N']

sig = data['sig'] # sigma for GP
sig_n = data['sig_n'] # noise for GP

U0 = data['U0']

nm = data['nm'] # how often the map should be applied
dtsymp = data['dtsymp']

qmin = data['qmin']
qmax = data['qmax']
pmin = data['pmin']
pmax = data['pmax']

qminmap = np.array(np.pi - 2.8)
qmaxmap = np.array(np.pi + 1.5)
pminmap = np.array(-2.3)
pmaxmap = np.array(1.8)
Ntest = 3