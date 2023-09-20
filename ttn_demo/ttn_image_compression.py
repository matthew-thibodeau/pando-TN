# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:23:58 2022

@author: mthibode
"""

import numpy as np
import torch
# import torchvision
import quimb
import quimb.tensor as qtn
from random import random, sample, choice
from ttn import TTN, propose_move, evaluate_move, optimize_MPO, anneal_timestep, sweep_tree, energy, local_ham
from copy import copy
from math import log2
from ttn_q_learn import q_learn
from PIL import Image
from ttn import TTN, propose_move, evaluate_move, optimize_MPO, anneal_timestep, sweep_tree, energy, local_ham
from copy import copy

def img_to_vector(image):
    
    return np.ravel(image), image.shape

def pad_to_twopower(v):
    bits = int(np.log2(len(v)) + 1)
    val = 2 ** bits
    return np.pad(v, (0,val - len(v)))

def singlesite_matrices(basis_state_in, basis_state_out, nsites):
    
    this_projectors = []
    inbits = bin(basis_state_in)[2:]#[::-1]
    outbits = bin(basis_state_out)[2:]#[::-1]
    
    inbits = inbits.zfill(nsites)#[::-1]
    outbits = outbits.zfill(nsites)#[::-1]
    
    for bitidx in range(len(inbits)):
        inbit = int(inbits[bitidx])
        outbit = int(outbits[bitidx])
        this_mat = np.zeros((2,2))
        this_mat[inbit, outbit] = 1
        this_projectors.append(this_mat)

    return this_projectors

def projector_mpo(kernel_state, max_bond = 512):
    
    space_dim = kernel_state.shape[0]
    nsites = int(log2(space_dim))
    
    upper = qtn.MatrixProductState.from_dense(kernel_state, dims = [2] * nsites, site_ind_id = 'u{}', max_bond = max_bond)
    lower = qtn.MatrixProductState.from_dense(kernel_state, dims = [2] * nsites, site_ind_id = 'l{}', max_bond = max_bond)
    
    H = upper & lower
    for i in range(nsites):
        H = H ^ f'I{i}'
    H.fuse_multibonds()
    
    
    HMPO  = qtn.MatrixProductOperator.from_TN(H, upper_ind_id = 'u{}', lower_ind_id = 'l{}', cyclic = False)
    
    return -1 * HMPO
    
img = Image.open('images/tree.png').convert('L')
# img = img.resize((100,100))
imgdata = np.array(img.getdata()).reshape((img.size[1], img.size[0]))
# image = np.load('')
imv = np.ravel(imgdata)
imp = pad_to_twopower(imv)
impn = imp/np.linalg.norm(imp)

space_dim = impn.shape[0]
nsites = int(log2(space_dim))

bd = None

H = projector_mpo(impn, bd)
dmrg = qtn.DMRG(H, int(bd / 2))

do_solve = False
if do_solve:
    dmrg.solve(verbosity = 1)
    print(dmrg.energy)
    
    cv = np.abs(dmrg.state.to_dense()[:len(imv)])
    imcd = (np.reshape(cv, imgdata.shape) / np.max(cv) * 255).astype('uint8')
    imc = Image.fromarray(imcd)
    print(dmrg.state.to_dense().T @ impn)
    
    upper = qtn.MatrixProductState.from_dense(impn, dims = [2] * nsites, site_ind_id = 'u{}', max_bond = bd)
