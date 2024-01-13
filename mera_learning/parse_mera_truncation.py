#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:02:06 2023

@author: matthewthibodeau
"""

import numpy as np
import pickle
import os


def mera_adjacency_matrix(mera_bonds):
    
    # mera_bonds should be the data loaded from one of the '{uid}_bonds.npy' files
    mera_phys_dim = min((min(t) for _,t in mera_bonds))
    mera_phys_size = len([t for _,t in mera_bonds if list(t) == [mera_phys_dim] * 4])
    adj = np.zeros((len(mera_bonds), len(mera_bonds)))
    
    level = 0
    max_level = int(np.log2(mera_phys_size))
    start = 0
    while level < max_level:
        level_size = mera_phys_size // (2 ** level)
        
        # add the unitaries
        this_level_unitaries = np.arange(start, start + 2*level_size, 2)
        unitary_sizes = np.array([mera_bonds[k][1][0] for k in this_level_unitaries])
        this_level_isometries = this_level_unitaries + 1
        
        next_level_size = level_size // 2
        next_level_start = start + 2 * level_size
        next_level_unitaries = np.arange(next_level_start, next_level_start + 2*next_level_size, 2)
        next_level_unitary_sizes = np.array([mera_bonds[k][1][0] for k in next_level_unitaries])
        # connect the unitaries and isometries
        
        adj[this_level_unitaries, this_level_isometries] = unitary_sizes
        adj[this_level_unitaries, np.roll(this_level_isometries, 1)] = unitary_sizes
        adj[this_level_isometries, this_level_unitaries] = unitary_sizes
        adj[np.roll(this_level_isometries, 1),this_level_unitaries] = unitary_sizes
        
        adj[this_level_isometries, np.repeat(next_level_unitaries, 2)] = np.repeat(next_level_unitary_sizes, 2)
        adj[np.repeat(next_level_unitaries, 2), this_level_isometries] = np.repeat(next_level_unitary_sizes, 2)

        level += 1
        start = next_level_start
        
    adj[-2, -1] = sum(mera_bonds[-1][1])
    adj[-1, -2] = sum(mera_bonds[-1][1])
    
    return adj


datadir = 'data/mera_data/'
jobfiles = os.listdir(datadir)

# find random field ising uids
hamtype = 'ising_random_fields'
Lval = 32
Dval = 2

uidfiles = [x for x in jobfiles if 'bonds' in x]
uids = []
for s in uidfiles:
    _, slurmid, procid, runid, lid, did, _= s.split('_')
    if lid != f'L{Lval}' or did != f'D{Dval}':
        continue
    
    hamtypefile = f'MERA_{slurmid}_{procid}_0_{lid}_{did}_hamtype.pkl'
    try:
        
        # check to see if bonds file is valid
        with open(f'{datadir}{s}', 'rb') as f:
            x = pickle.load(f)
        
        with open(f'{datadir}{hamtypefile}', 'rb') as f:
            x = pickle.load(f)
            if x[0][1] == hamtype:
                uids.append(s.split('bonds')[0][:-1])
    except:
        continue
            


# with open(f'data/mera_learning_data/jobdir/{this_id}_hamtype.pkl', 'wb') as f:
#     pickle.dump([('HAMTYPE', HAMTYPE), ('disorder_strength', disorder_strength),
#                  ('j0', j0)], f)

# determine bonds


all_errors = []
error_comparison = []
bonds = []
adjs = []
k = 0
for uid in uids:
    with open(f'{datadir}{uid}_bonds.pkl', 'rb') as f:
        thisbonds = pickle.load(f)
        bonds.append(thisbonds)
    
    adjpath = f'{datadir}{uid}_adjacency.pkl'
    if os.path.isfile(adjpath):
        # adj = np.load(adjpath)
        with open(adjpath, 'rb') as f:
            adj = pickle.load(f)
    else:
        if type(thisbonds) == list:
            adj = [mera_adjacency_matrix(t) for t in thisbonds]
        else:
            adj = mera_adjacency_matrix(thisbonds)
            
        with open(adjpath, 'wb') as f:
            pickle.dump(adj, f)
        # np.save(adjpath, adj)
    adjs.append(adj)
    
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