"""Utility functions for the I/O of the TTN optimization"""

import pandas as pd
import numpy as np
from typing import List

from ttn import TTN

####################################################################

def from_optimization_run_to_dataframe(
        states: List[TTN], energies: List[float], all_energies: List[float],
        seed: int, ham_vals: np.ndarray) -> pd.DataFrame:
    """Convert the output of a SA optimization run to a dataframe
    
    Arguments:
    - states: List of the states accepted during the optimization, they are objects of class tnn.TNN
    - energies: List of energies of the accepted states
    - all_energies: List of energies obtained during the optimization, including both accepted and rejected moves (or states)

    Output:
    - A pandas.DataFrame with two columns ('state' and 'energy') capturing all moves.
      If the move was not accepted, the corresponding state is 'None'.
      The DataFrame also has the global metadata 'seed'.
    """
    assert len(states) == len(energies)
    assert len(all_energies) >= len(energies)
    id_e = 0
    all_states = []
    for e in all_energies:
        if id_e < len(energies) and energies[id_e] == e:
            all_states.append(states[id_e])
            id_e += 1
        else:
            all_states.append(None)
    # Confirm that all energies could be found, ordered, among the all_energies.
    assert id_e == len(energies)
    df = pd.DataFrame({'state': all_states, 'energy': all_energies})
    df.attrs = {'seed': seed, 'ham': ham_vals}
    return df

