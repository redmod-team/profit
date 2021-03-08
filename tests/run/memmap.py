#!/bin/env python3

import numpy as np
import pandas as pd
import os

PATH = './interface.npy'

if os.path.exists(PATH):
    data = np.load(PATH, mmap_mode='r+')
    df = pd.DataFrame(data.flatten(), index=np.arange(data.size))
    print(df)

else:
    data = np.zeros((4, 1), dtype=[('u', float), ('v', float), ('DONE', bool), ('f', float), ('TIME', int)])
    data['u'][:, 0] = np.linspace(4.7, 5.3, data.size)
    data['v'][:, 0] = np.linspace(0.55, 0.6, data.size)
    np.save(PATH, data)
