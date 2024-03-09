"""Utility function to identify the path of a specific simulation round"""

from typing import Tuple
import os
import numpy as np
import pickle
import pandas as pd
import sys
sys.path.insert(0, '../ttn_demo')
import ttn

####################################################################

def get_id_of_files(today: str, L: int, site_dim: int, bond_size: int, run: int = 0) -> Tuple[str]:
    """Get the identifiers of a specific simulation round

    This function is necessary because, for certain runs, there is no way
    to know the slurm_jobid and slurm_procid before actually launching the
    job. It has to be added manually.
    """
    assert site_dim == 2
    slurm_jobid = None
    slurm_procid = None
    if today == '2023-10-05':
        slurm_jobid = '1696495532'
        slurm_procid= '1696495532'
    elif today == '2023-11-27':
        if L == 16 and bond_size == 3:
            if run < 28:
                slurm_jobid = '19496'
            elif run < 56:
                slurm_jobid = '19500'
            slurm_procid= '0'
        elif L == 16 and bond_size == 10:
            if run < 28:
                slurm_jobid = '19498'
            elif run < 56:
                slurm_jobid = '19502'
            slurm_procid= '0'
        elif L == 24 and bond_size == 3:
            if run < 28:
                slurm_jobid = '19497'
            elif run < 56:
                slurm_jobid = '19501'
            slurm_procid= '0'
        elif L == 24 and bond_size == 10:
            print('Simulation round failed.')
    elif today == 'TEST':
        assert L == 16 and bond_size == 3
        slurm_jobid = '19434'
        slurm_procid = '0'
    else:
        print('Unknown simulation round.')
    return slurm_jobid, slurm_procid

def get_all_uids(today: str, L: int, site_dim: int, bond_size: int, data_path: str):
    
    
    if today == '2023-10-05':
        raise NotImplementedError('runs not supported yet')
    elif today == '2023-11-27':
        jobids = [str(19496 + k) for k in range(7)]
        
    
    job_list = os.listdir(data_path)
    accept_jobs = [x.split('_summary.pkl')[0] for x in job_list 
                   if f'D{site_dim}' in x and f'L{L}' in x and f'm{bond_size}' in x
                   and 'summary.pkl' in x]
    
    # check jobid against whitelist
    accept_jobs = [x for x in accept_jobs if x.split('_')[2] in jobids]
    
    return accept_jobs


def associate_pairs(today, L, site_dim, bond_size, data_path, save_path):
    '''
    Get all data given the specifiers; for each, find the most optimized TTN x;
    and save the pairs (ham, x) for learning
    '''
    uids = get_all_uids(today, L, site_dim, bond_size, data_path)
    
    all_hamvals = []
    all_ttns = []
    
    for uid in uids:
        
        # load the data
        with open(f'{data_path}/{uid}_summary.pkl', 'rb') as f:
            df = pd.read_pickle(f)
        with open(f'{data_path}/{uid}_hamiltonian_vals.pkl', 'rb') as f:
            hamvals = pd.read_pickle(f)
            
        all_hamvals.append(hamvals)
        
        valid_idxs = np.where(1 - df['state'].isna())[0]
        min_energy_idx = np.argmin(df['energy'][valid_idxs])
        min_state = df['state'][valid_idxs[min_energy_idx]]
        
        all_ttns.append(min_state)
        
    pairs = list(zip(all_hamvals, all_ttns))
    
    with open(f'{save_path}/pairs_L{L}_D{site_dim}_m{bond_size}.pkl', 'wb') as f:
        pickle.dump(pairs, f)
    
    
    
    
####################################################################

