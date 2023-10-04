#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Program to run the optimization of Tree Tensor Networks using Simulated Annealing

The optimization involves both the connectivity of the tree tensor and
the bond dimension associated with the edges.

Arguments are:
* '-L' = number of spins/qubits in the Hamiltonian
* '-m' = max bond dimension
* '-d' = strength of disorder
* '-s' = seed of the Random Number Generator
* '-r' = number of Hamiltonian instances
"""
import sys
import os
import time
import argparse
import pickle


import numpy as np
from scipy.sparse import coo_matrix
from numpy.random import SeedSequence, PCG64


sys.path.insert(0, '../ttn_demo')

import ttn
import ttn_q_learn

import quimb as qu
import quimb.tensor as qtn

####################################################################
# Default values and utility variables.
####################################################################

# Set double precision.
dtype = 'float32'

# Matrices.
sz = np.array([[ 1.0,  0],
               [ 0, -1.0]], dtype=dtype)
i2 = np.eye(2, dtype=dtype)
Z1I2 = np.kron(sz, i2)
H2 = qu.ham_heis(2).real.astype(dtype)
X2 = np.array([[0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [1, 0, 0, 0]], dtype=dtype)

# Default values
j0 = 1
L = 16
max_bond = 3
disorder_strength = 1.0
site_dim = 2
num_runs = 1 # TODO: consider calling it num_hams
num_opt_rounds = 10
rng_seed = 7777

####################################################################
# Main
####################################################################

if __name__ == "__main__":
    
    data_path = 'data/TTN_SA_data'
    
    # make sure data path exists
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--nqubits', dest='L', default=L)
    parser.add_argument('-m', '--maxbond', dest='max_bond', default=max_bond)
    parser.add_argument('-d', '--disorder', dest='disorder_strength', default=disorder_strength)
    parser.add_argument('-s', '--rngseed', dest='rng_seed', default=rng_seed)
    parser.add_argument('-r', '--runs', dest='num_runs', default=num_runs)
    args = parser.parse_args()

    L = int(args.L)
    max_bond = int(args.max_bond)
    disorder_strength = float(args.disorder_strength)
    rng_seed = int(args.rng_seed)
    num_runs = int(args.num_runs)
    
    
    # Generate randomness.
    ss = SeedSequence(rng_seed)
    rng = np.random.Generator(PCG64(ss))
    
    # If SLURM environment variables are not present, assume this is a local test run.
    slurm_jobid = os.environ.get('SLURM_JOB_ID', int(time.time()))
    slurm_procid = os.environ.get('SLURM_PROCID', int(time.time()))
    
    # TODO: Here a 'run' correponds to a different Hamiltonian, right?
    for run in range(num_runs):
        this_id =  f'TTN_SA_{slurm_jobid}_{slurm_procid}_{run}_L{L}_D{site_dim}'
        
        hamiltonian_vals = rng.normal(j0, disorder_strength, L)
    
        builder = qtn.SpinHam1D(S=1/2, cyclic=False)
        terms = {}
        HAMTYPE = 'ising_random_fields'
        for i in range(0, L-1):
            builder[i, i+1] += -1.0, 'X', 'X'
            builder[i, i+1] += hamiltonian_vals[i], 'Z', 'I'
            terms[(i, (i+1)%L)] =  -1 * X2 + hamiltonian_vals[i] * Z1I2
        builder[L-2, L-1] += hamiltonian_vals[L-1], 'I', 'Z'
        H_mpo = builder.build_mpo(L)
    
        states, energies, all_energies = ttn.optimize_MPO(H_mpo, max_bond, rounds = num_opt_rounds,
                                                          min_coord = 3, max_coord = 3, rng = rng)
        adj_matrices = [coo_matrix(x.get_adjacency_matrix()) for x in states]
    
        # Save the data.
        with open(f'{data_path}/{this_id}_acceptedstates.pkl', 'wb') as f:
            pickle.dump(adj_matrices, f)
        with open(f'{data_path}/{this_id}_acceptedenergies.pkl', 'wb') as f:
            pickle.dump(energies, f)
        with open(f'{data_path}/{this_id}_proposedenergies.pkl', 'wb') as f:
            pickle.dump(all_energies, f)
        with open(f'{data_path}/{this_id}_randomseed.pkl', 'wb') as f:
            pickle.dump(ss.entropy, f)
        with open(f'{data_path}/{this_id}_hamiltonian_vals.pkl', 'wb') as f:
            pickle.dump(hamiltonian_vals, f)
            
        if run == 0:
            # TODO: Are we saving only the result of the first run?
            with open(f'{data_path}/{this_id}_hamtype.pkl', 'wb') as f:
                pickle.dump([('HAMTYPE', HAMTYPE), ('disorder_strength', disorder_strength),
                             ('j0', j0)], f)








    