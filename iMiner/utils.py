import numpy as np


def dist_mat(crd1: np.ndarray, crd2: np.ndarray) -> np.ndarray:
    '''
    Calculate distance matrix between two coordinates sets

    Parameters
    ----------
    crd1: numpy.ndarray
        The first coordinate set with dim N1 x 3
    crd2: numpy.ndarray
        The second coordinate set with dim N2 x 3
        
    Return
    ------
    dist_mat: numpy.ndarray
        Distance matrix with dim N1 x N2
    '''
    
    expand_crd1 = np.expand_dims(crd1, 1).repeat(crd2.shape[0], axis=1)
    expand_crd2 = np.broadcast_to(crd2, (crd1.shape[0], crd2.shape[0], 3))
    dist_mat = np.linalg.norm(expand_crd1 - expand_crd2, ord=2, axis=-1)
    return dist_mat
