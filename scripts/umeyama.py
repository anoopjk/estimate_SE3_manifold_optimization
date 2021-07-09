"""
Copyright: Carlo Nicolini, 2013
Code adapted from the Mark Paskin Matlab version
from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 

modified by Anoop Jakka
"""

import numpy as np


def estimate_SIM3_umeyama(X,Y):
    """[summary]
    estimates rigid transform from X(source) to Y(target)
    Args:
        X ([type]): Nx3 array
        Y ([type]): Nx3 array

    Returns:
        [type]: [description]
    """
    n= X.shape[0]
    m = 3

    mx = X.mean(0, keepdims=True)
    my = Y.mean(0, keepdims=True)
    Xc =  X - mx#np.tile(mx, (n, 1)).T
    Yc =  Y - my#np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, axis=1))
    sy = np.mean(np.sum(Yc*Yc, axis=1))

    Sxy = (Xc.T @ Yc) / n

    U,D,VT = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    r = Sxy.ndim #np.rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1  
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t
    print(U.shape, S.shape, VT.shape)
    R = ((U @ S) @ VT).T
    c = np.trace(np.diag(D) @ S) / sx
    #t = my - c * mx @ R
    t = my.reshape(-1,1) - c *(R @ mx.reshape(-1,1)) 
    print('t: ', t)
    # create a 4x4 transform matrix
    T = np.eye(4)
    # c = 1.0
    T[:3,:3] = c*R 
    print(my.shape, R.shape, mx.shape)
    T[:3,3] = t.reshape(-1,)

    return T