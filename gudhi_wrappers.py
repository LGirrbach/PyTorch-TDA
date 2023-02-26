"""
From https://github.com/MathieuCarriere/difftda/blob/master/difftda.py
"""


import numpy as np

from gudhi.rips_complex import RipsComplex
from gudhi.cubical_complex import CubicalComplex


############################
# Vietoris-Rips filtration #
############################

# The parameters of the model are the point coordinates.

def Rips(DX, mel, dim, card):
    # Parameters: DX (distance matrix),
    #             mel (maximum edge length for Rips filtration),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Compute the persistence pairs with Gudhi
    rc = RipsComplex(distance_matrix=DX, max_edge_length=mel)
    st = rc.create_simplex_tree(max_dimension=dim + 1)
    dgm = st.persistence()
    pairs = st.persistence_pairs()

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for s1, s2 in pairs:
        if len(s1) == dim + 1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1, :][:, l1]), [len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2, :][:, l2]), [len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(st.filtration(s2) - st.filtration(s1))

    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())

    # Output indices
    indices = indices[:4 * card] + [-1 for _ in range(0, max(0, 4 * card - len(indices)))]
    return list(np.array(indices, dtype=np.int32))


######################
# Cubical filtration #
######################

# The parameters of the model are the pixel values.

def Cubical(X, dim, card):
    # Parameters: X (image),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Compute the persistence pairs with Gudhi
    cc = CubicalComplex(dimensions=X.shape, top_dimensional_cells=X.flatten())
    cc.persistence()
    try:
        cof = cc.cofaces_of_persistence_pairs()[0][dim]
    except IndexError:
        cof = np.array([])

    Xs = X.shape

    if len(cof) > 0:
        # Sort points with distance-to-diagonal
        pers = [X[np.unravel_index(cof[idx, 1], Xs)] - X[np.unravel_index(cof[idx, 0], Xs)] for idx in range(len(cof))]
        perm = np.argsort(pers)
        cof = cof[perm[::-1]]

    # Retrieve and ouput image indices/pixels corresponding to positive and negative simplices
    D = len(Xs)
    ocof = np.array([-1 for _ in range(D * card * 2)])
    count = 0
    for idx in range(0, min(2 * card, 2 * cof.shape[0]), 2):
        ocof[D * idx:D * (idx + 1)] = np.unravel_index(cof[count, 0], Xs)
        ocof[D * (idx + 1):D * (idx + 2)] = np.unravel_index(cof[count, 1], Xs)
        count += 1
    return list(np.array(ocof, dtype=np.int32))
