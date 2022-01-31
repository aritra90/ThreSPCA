import numpy as np #for all numerical computations
import pandas as pd #similar to dataframe in R
import scipy as sp #scientific computations (includes stats)
import scipy.io as sio
from sklearn.utils.extmath import randomized_svd
import random
import sys
import time
import math
from scipy.sparse import lil_matrix
from scipy.sparse import identity
from numpy import dot


def ThreSPCA(DatMat, s, l):

    """
    Input:
    DatMat: mxn data matrix.
    s: Scalar denoting the target sparsity.
    l: Thresholding scalar. We set it to one.
    
    Output:
    z:  vector of order nx1.
    nnz: Number of nonzero elements in z.
    """
    
    """
    In order to get the PC2,
    run this function on DatMat_new=DatMat-DatMat*vv^T.
    Here v is the top right singular vector of DatMat.
    Similarly, run for PC2, PC4,... and so on.
    """
    
    
    ## get the top l singular vectors of A
    _, Sig, VT =randomized_svd(A, n_components=l)
    U=VT.T
    
    ## Compute the squared row norms of U
    row_normsq= np.linalg.norm(U,axis=1)**2
    
    ## Indices of top s elements of row_normsq
    S = np.argsort(row_normsq)[-s:]
    
    ## Compute the sampling matrix R of order n x |R|
    R=identity(DatMat.shape[1],dtype='int8', format='csc')[:,S]
    
    ## Extract the columns of UT from R
    UT_sampled=VT[:,S]
    
    ## Compute y=max ||Sig UT R x||_2 such that ||x||_2=1
    _, _, VT_new= randomized_svd(np.matmul(np.diag(Sig),UT_sampled), n_components=1)
    y=VT_new[0]
    
    ## sparse ouput
    z=R.dot(y)
    nnz=np.count_nonzero(z)

    print("ThreSPCA done!")
    return z, nnz


