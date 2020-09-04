import numpy as np
from scipy import stats


def Qc(matrix, init_permutation=None, pre_process=False, normalized=False):
    ''' Equation 1 of the paper. '''

    C = np.array(matrix)
    assert C.ndim in (2, 3)

    # assume the trivial order if the initial permutation is not informed
    N = C.shape[1]
    if init_permutation is None: init_permutation = list(range(N))

    # correct matrix (index -> id space)
    ids = np.argsort(init_permutation)
    C = C[ids][:, ids]
    if pre_process:
        np.fill_diagonal(C, 0)
        C = C.max() - C
        np.fill_diagonal(C, 1e7)

    F_ij = 0
    for i in range(N - 1):
        # row-wise verification
        row_min = C[i].min()
        Rv = (C[i, i + 1] == row_min) and (np.sum(C[i] == row_min) == 1)

        # column-wise verification
        col_min = C[:, i + 1].min()
        Cv = (C[i, i + 1] == col_min) and (np.sum(C[:, i + 1] == col_min) == 1)

        # Equation 2
        F_ij += int(Rv and Cv)

    if normalized:
        return F_ij / (N - 1)
    return F_ij


def accuracy(solution, init_permutation=None, sizes=None):
    ''' Accuracy by neighbor comparison.
    solution: permutation output by the solver.
    init_permutation: id initial permutation.
    sizes: listes with the number of strips of each reconstruction instance.
    '''

    assert len(solution) > 0

    # assume an unique instance with N = len(solution) strips
    N = len(solution)
    if sizes == None:
        sizes = [N]

    # assume the trivial order if the initial permutation is not informed
    if init_permutation is None:
        init_permutation = list(range(N))

    # correct solution (index -> id space)
    corrected = [init_permutation[idx] for idx in solution]

    # ids of the first/last strip of each document
    first = []
    curr = 0
    for size in sizes:
        first.append(curr)
        curr += size

    # iterate over documents
    num_correct = 0
    neighbors = {}
    curr = 0
    for doc_idx, size in enumerate(sizes):
        # iterate over strips
        for _ in range(size - 1):
            neighbors[curr] = [curr + 1]
            curr += 1
        # add each first strip of the documents as neighbor (except the first strip of the current document)
        neighbors[curr] = first.copy()
        neighbors[curr].remove(first[doc_idx])
        curr += 1
    for idx in range(N - 1):
        if corrected[idx + 1] in neighbors[corrected[idx]]:
            num_correct += 1
    return num_correct / (N - 1)