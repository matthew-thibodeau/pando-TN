# Tests for learn_ttn.py
# Usage: pytest -s test_learn_ttn.py


import learn_ttn
import numpy as np
import networkx as nx


def test_matrixweights_to_tree():

    mat = np.array(
    [[1.19178, 3.21255, 4.01415, 1.85112, 4.56274, 0.02254],
     [1.44957, 0.59603, 3.71608, 4.77957, 1.71311, 2.72046],
     [3.68687, 3.52903, 3.67817, 4.29076, 0.18122, 3.12189],
     [4.12143, 2.50373, 1.57742, 5.23519, 2.64804, 5.16431],
     [3.94234, 5.12345, 5.63192, 5.72876, 4.05916, 4.67962],
     [3.41054, 2.46785, 5.34059, 1.17318, 5.61336, 1.45742]]
    )

    G = learn_ttn.matrixweights_to_tree(mat,[0,1,2])
    assert nx.is_tree(G) == True
    # print(G.edges())







