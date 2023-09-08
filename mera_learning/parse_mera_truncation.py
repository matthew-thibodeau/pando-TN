#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:02:06 2023

@author: matthewthibodeau
"""

import numpy as np
import pickle
import os


datadir = 'data/mera_data/'
jobdirs = os.listdir(datadir)




# save the data
svd_bonds = []
for k,t in zip(range(len(emo.tensors)), emo.tensors):
    svd_bonds.append((k,t.shape))

with open(f'data/mera_data/{this_id}_bonds.pkl', 'wb') as f:
    pickle.dump(svd_bonds, f)
np.save(f'data/mera_data/{this_id}_errors_svd.npy', errors)
np.save(f'data/mera_data/{this_id}_errors_largest.npy', errors_largest)