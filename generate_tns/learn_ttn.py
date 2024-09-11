# Pando functions for learning TTNs


import numpy as np
import networkx as nx


def matrixweights_to_tree(adjmat,leafs):
    """Convert matrix weights to tree structure.

    Note: Uses *only* upper triangular part of matrix.
    
    Args:
        mat (np.ndarray): Matrix weights
        leafs (list): List of leaf node IDs

    Returns:
        tree (dict): Tree structure
    """

    # 
    # Perform the following checks:
    # 
    # - Check if there are any cycles
    # - Leafs cannot have more than one edge
    # - Is graph connected?
    # 
    # - Double check it's indeed a tree
    # 


    G = nx.Graph()
    G.add_nodes_from(range(adjmat.shape[0]))

    # Take only upper triangular part       
    mat = np.triu(adjmat)
    # Remove diagonal
    np.fill_diagonal(mat,0)
    # print(mat)

    # Loop until tree is produced
    while not np.all(mat <= 0):

        # Find max index
        max_index = np.unravel_index(np.argmax(mat), mat.shape)
        # Break if max weight is 0
        if mat[max_index] == 0:
            break
        val = mat[max_index]
        w = round(val)
        if w==0: # if w==0 here, means was rounded down from finite number
            w = 1

        # Add edge
        G.add_edge(max_index[0],max_index[1],weight=w)
        # Set to zero
        mat[max_index] = 0

        # Check if there are any cycles
        if len(nx.cycle_basis(G)) > 0:
            G.remove_edge(max_index[0],max_index[1])
            continue

        # Check if leafs have only one edge
        for leaf in leafs:

            if G.degree[leaf] > 1:
                G.remove_edge(max_index[0],max_index[1])
                continue

        # Check if graph is connected
        if nx.is_connected(G):
            # If here, means it should be a tree
            break




    # Check if tree was found
    if not nx.is_tree(G):
        raise ValueError("Tree not found")
    
    # Return
    return G



