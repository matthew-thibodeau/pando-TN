"""Utility function to identify the path of a specific simulation round"""

from typing import Tuple

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
        
####################################################################

