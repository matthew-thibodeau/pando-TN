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
* '--runstart' = first run
* '--runend' = last run
* '-i' = number of Simulated Annealing iterations
* '-T' = temperature of the SA
* '--today' = date or other label of the SA run
"""
import sys
import os
import time
import datetime
import argparse
import pickle

import numpy as np
from numpy.random import SeedSequence, PCG64
from scipy.sparse import coo_matrix

sys.path.insert(0, '../ttn_demo')

import ttn
import ttn_q_learn
from utils import from_optimization_run_to_dataframe

import quimb as qu
import quimb.tensor as qtn

####################################################################
# Default values and utility variables.
####################################################################

today = str(datetime.date.today())
    
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
bond_size = 10
disorder_strength = 1.0
site_dim = 2

run_start = 0
run_end = 1

num_opt_rounds = 1
rng_seed = 7777
sa_temp = 1e-4

####################################################################
# Main
####################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--nqubits', dest='L', default=L)
    parser.add_argument('-m', '--bond_size', dest='bond_size', default=bond_size)
    parser.add_argument('-d', '--disorder', dest='disorder_strength', default=disorder_strength)
    parser.add_argument('-s', '--rngseed', dest='rng_seed', default=rng_seed)
    parser.add_argument('--runstart', dest='run_start', default=run_start)
    parser.add_argument('--runend', dest='run_end', default=run_end)
    parser.add_argument('-i', '--iterations', dest='num_opt_rounds', default=num_opt_rounds)
    parser.add_argument('-T', '--temperature', dest='sa_temp', default=sa_temp)
    parser.add_argument('--today', dest='today', default=today)
    args = parser.parse_args()
    
    L = int(args.L)
    disorder_strength = float(args.disorder_strength)
    rng_seed = int(args.rng_seed)
    run_start = int(args.run_start)
    run_end = int(args.run_end)
    num_opt_rounds = int(args.num_opt_rounds)
    sa_temp = float(args.sa_temp)
    today = args.today
    bond_size = int(args.bond_size)

    # Print values of the program arguments:
    print('\nOptimization with Simulated Annealing.\n',
          f'L: Hamiltonians have {L} qubits\n',
          f'm: bond dimension is {bond_size}\n',
          f'd: disorder has strength {disorder_strength}\n',
          f's: seed of RNG is {rng_seed}\n',
          f'runstart: first run, i.e. of random values for the field, is {run_start}\n',
          f'runend: last run, i.e. of random values for the field, is {run_end}\n',
          f'i: number of iterations of each SA optimization is {num_opt_rounds}\n',
          f'T: SA optimization temperature is {sa_temp:.3e}\n',
          f'today: label for the datafiles is {today}\n'
          )

    data_path = f'data/{today}_TTN_SA'
    # Make sure data path exists  
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    
    # Generate randomness.
    ss = SeedSequence(rng_seed)
    rng = np.random.Generator(PCG64(ss))
    rng_H = np.random.Generator(PCG64(ss))
    
    # If SLURM environment variables are not present, assume this is a local test run.
    slurm_jobid = os.environ.get('SLURM_JOB_ID', int(time.time()))
    slurm_procid = os.environ.get('SLURM_PROCID', int(time.time()))
    
    for run in range(run_start):
        # catch up the rng_H to the correct run
        _ = rng_H.normal(j0, disorder_strength, L)
    
    for run in range(run_start, run_end):
        print(f'---- run {run} ----')
        
        hamiltonian_vals = rng_H.normal(j0, disorder_strength, L)
    
        builder = qtn.SpinHam1D(S=1/2, cyclic=False)
        terms = {}
        HAMTYPE = 'ising_random_fields'
        for i in range(0, L-1):
            builder[i, i+1] += -1.0, 'X', 'X'
            builder[i, i+1] += hamiltonian_vals[i], 'Z', 'I'
            terms[(i, (i+1)%L)] =  -1 * X2 + hamiltonian_vals[i] * Z1I2
        builder[L-2, L-1] += hamiltonian_vals[L-1], 'I', 'Z'
        H_mpo = builder.build_mpo(L)
        
        this_id =  f'TTN_SA_{slurm_jobid}_{slurm_procid}_{run}_L{L}_D{site_dim}_m{bond_size}'

        states, energies, all_energies = ttn.optimize_MPO(H_mpo, bond_size, rounds = num_opt_rounds,
                                                          min_coord = 3, max_coord = 3, rng = rng, temp = sa_temp)
        adj_matrices = [coo_matrix(x.get_adjacency_matrix()) for x in states]
    
        # From lists to pandas dataframe.
        df = from_optimization_run_to_dataframe(states, energies, all_energies,
                                                ss.entropy, hamiltonian_vals)
        #print(df)

        # Save the data.
        with open(f'{data_path}/{this_id}_summary.pkl', 'wb') as f:
            df.to_pickle(f)
        with open(f'{data_path}/{this_id}_hamiltonian_vals.pkl', 'wb') as f:
            pickle.dump(hamiltonian_vals, f)
        if False:
            with open(f'{data_path}/{this_id}_acceptedstates.pkl', 'wb') as f:
                pickle.dump(adj_matrices, f)
            with open(f'{data_path}/{this_id}_acceptedenergies.pkl', 'wb') as f:
                pickle.dump(energies, f)
            with open(f'{data_path}/{this_id}_proposedenergies.pkl', 'wb') as f:
                pickle.dump(all_energies, f)
            with open(f'{data_path}/{this_id}_randomseed.pkl', 'wb') as f:
                pickle.dump(ss.entropy, f)
            
        if run == 0:
            # The Hamiltonian type is common between all runs, thus we save it only once.
            with open(f'{data_path}/{this_id}_hamtype.pkl', 'wb') as f:
                pickle.dump([('HAMTYPE', HAMTYPE), ('disorder_strength', disorder_strength),
                             ('j0', j0)], f)

    print('---- The end ----')
    