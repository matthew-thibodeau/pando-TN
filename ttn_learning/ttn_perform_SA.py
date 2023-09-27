#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Program to run the optimization of Tree Tensor Networks using Simulated Annealing

The optimization involves both the connectivity of the tree tensor and
the bond dimension associated with the edges.

Arguments are:

"""
import sys
import argparse
import numpy as np

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

####################################################################
# Main
####################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--nqubits', dest='L', default=L)
    parser.add_argument('-m', '--maxbond', dest='max_bond', default=max_bond)
    parser.add_argument('-d', '--disorder', dest='disorder_strength', default=disorder_strength)
    args = parser.parse_args()

    L = int(args.L)
    max_bond = int(args.max_bond)
    disorder_strength = float(args.disorder_strength)


    coupling_vals = np.random.normal(j0, disorder_strength, L)

    builder = qtn.SpinHam1D(S=1/2, cyclic=False)
    terms = {}

    for i in range(0, L):
        builder[i, i+1] += -1.0, 'X', 'X'
        builder[i, i+1] += coupling_vals[i], 'Z', 'I'


        terms[(i, (i+1)%L)] =  -1 * X2 + coupling_vals[i] * Z1I2

    H_mpo = builder.build_mpo(L)

    states, energies, all_energies, init_e =  ttn.optimize_MPO(H_mpo, max_bond, rounds = 10, min_coord = 3, max_coord = 3)

    