# Module for basic manipulations of atoms/crds/angles/proteins in numpy and torch 
import numpy as np 
# import torch 

def np_kabsch(A,B):
    """
    Numpy version of kabsch algorithm. Superimposes B onto A

    Parameters:
        (A,B) np.array - shape (N,3) arrays of xyz crds of points


    Returns:
        rms - rmsd between A and B
        R - rotation matrix to superimpose B onto A
        rB - the rotated B coordinates
    """
    A = np.copy(A)
    B = np.copy(B)

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    def rmsd(V,W, eps=1e-6):
        # First sum down atoms, then sum down xyz
        N = V.shape[-2]
        return np.sqrt(np.sum((V-W)*(V-W), axis=(-2,-1)) / N + eps)


    N, ndim = A.shape

    # move to centroid
    A = A - centroid(A)
    B = B - centroid(B)

    # computation of the covariance matrix
    C = np.matmul(A.T, B)

    # compute optimal rotation matrix using SVD
    U,S,Vt = np.linalg.svd(C)


    # ensure right handed coordinate system
    d = np.eye(3)
    d[-1,-1] = np.sign(np.linalg.det(Vt.T@U.T))

    # construct rotation matrix
    R = Vt.T@d@U.T

    # get rotated coords
    rB = B@R

    # calculate rmsd
    rms = rmsd(A,rB)

    return rms, rB, R


def th_kabsch(A,B):
    """
    Torch version of kabsch algorithm. Superimposes B onto A

    Parameters:
        (A,B) torch tensor - shape (N,3) arrays of xyz crds of points


    Returns:
        R - rotation matrix to superimpose B onto A
        rB - the rotated B coordinates
    """

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return torch.mean(X, dim=0, keepdim=True)

    def rmsd(V,W, eps=1e-6):
        # First sum down atoms, then sum down xyz
        N = V.shape[-2]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(-2,-1)) / N + eps)


    N, ndim = A.shape

    # move to centroid
    A = A - centroid(A)
    B = B - centroid(B)

    # computation of the covariance matrix
    C = np.matmul(A.T, B)

    # compute optimal rotation matrix using SVD
    U,S,Vt = torch.svd(C)

    # ensure right handed coordinate system
    d = torch.eye(3)
    d[-1,-1] = torch.sign(torch.det(Vt@U.T))

    # construct rotation matrix
    R = Vt@d@U.T

    # get rotated coords
    rB = B@R

    # calculate rmsd
    rms = rmsd(A,rB)

    return rms, rB, R
