#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:02:06 2023

@author: matthewthibodeau
"""

import numpy as np
import pickle
import os

def uid(fname):
    ...


datadir = 'data/mera_data/'
jobfiles = os.listdir(datadir)

#find random field ising uids
hamtype = 'ising_random_fields'

hamtypefiles = [x for x in jobfiles if 'hamtype' in x]
uids = []
for s in hamtypefiles:
    with open(f'{datadir}{s}', 'rb') as f:
        x = pickle.load(f)
        if x[0][1] == hamtype:
            uids.append(s.split('hamtype')[0][:-1])
            


# with open(f'data/mera_learning_data/jobdir/{this_id}_hamtype.pkl', 'wb') as f:
#     pickle.dump([('HAMTYPE', HAMTYPE), ('disorder_strength', disorder_strength),
#                  ('j0', j0)], f)

all_errors = []
error_comparison = []
k = 0
for uid in uids:
    e_svd = np.load(f'{datadir}{uid}_errors_svd.npy')
    e_largest = np.load(f'{datadir}{uid}_errors_largest.npy')
    all_errors.append([e_svd, e_largest])
    error_comparison.append(e_largest - e_svd) # positive == svd is working better
    print(f'\r{100 * k / len(uids):.2f}% done loading...', end='')
    k+=1
    
max_iterations = max([len(x) for x in error_comparison])
ec_array = np.zeros((len(uids), max_iterations))
for j in range(len(error_comparison)):
    lj = error_comparison[j]
    ec_array[j,:len(lj)] = lj
    
ec_cleaned = ec_array.copy()
ec_cleaned[np.abs(ec_cleaned) > 1] = 0
    
error_comparison_avg = np.mean(ec_cleaned, axis=0)

for k in range(len(error_comparison_avg)):
    print(f'step {k}: error is {error_comparison_avg[k]:.3e}')




# # save the data
# svd_bonds = []
# for k,t in zip(range(len(emo.tensors)), emo.tensors):
#     svd_bonds.append((k,t.shape))

# with open(f'data/mera_data/{this_id}_bonds.pkl', 'wb') as f:
#     pickle.dump(svd_bonds, f)
# np.save(f'data/mera_data/{this_id}_errors_svd.npy', errors)
# np.save(f'data/mera_data/{this_id}_errors_largest.npy', errors_largest)