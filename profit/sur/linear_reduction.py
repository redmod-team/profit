"""High-dimensional output

This module concerns the following use-case: we make a parameter study over some
input parameters x and the domain code yields an output vector y contains many
entries. This is typically the case, when y is function-valued, i.e. depends
on an indenpendent variable t, or even "pixel-valued" output is produced,
like on a 2D map (like 1024x1024). Before fitting an input-output relation,
one has to reduce the output to a manageable number of dimensions.
This means extracting the relevant features.

Imagine the following idea: instead of viewing the output as many 1D outputs,
we look at it as a single vector in a high-dimensional space. Linear
combinations between vectors give new vectors. We can choose y from our
training data to span a basis in this vector space. Reducing this basis to an
appropriate set of orthonormal entries is done via the Karhunen-Loeve expansion.

"""

import numpy as np
from numpy.linalg import eigh


class KarhunenLoeve:
    r"""Linear dimension reduction by the Karhunen-Loeve expansion.
    This is efficient if the number of training samples ntrain is
    smaller than the number of support points N in independent variables.

    We want to write the i-th output function $y_i$
    as a linear combination of the other $y_j$ with, e.g. $j=1,2,3$
    $$
    y_i = a_1 y_1 + a_2 y_2 + a_3 y_3
    $$

    This can be done by projection with inner products:

    $$
    \begin{align}
    y_1 \cdot y_i &= a_1 y_1 \cdot y_1+a_2 y_1 \cdot y_2+a_3 y_1 \cdot y_3 \\
    y_2 \cdot y_i &= a_1 y_2 \cdot y_1+a_2 y_2 \cdot y_2+a_3 y_2 \cdot y_3 \\
    y_3 \cdot y_i &= a_1 y_3 \cdot y_1+a_2 y_3 \cdot y_2+a_3 y_3 \cdot y_3
    \end{align}
    $$

    We see that we have to solve a linear system with the
    collocation matrix $M_{ij} = y_i \cdot y_j$. To find the most relevant
    features and reduce dimensionality, we use only the highest eigenvalues.
    We center the data around the mean, i.e. subtract `ymean` before.

    Parameters:
        ytrain: ntrain sample vectors of length N.
        tol: Absolute cutoff tolerance of eigenvalues.
    """
    def __init__(self, ytrain, tol=1e-2):
        self.tol = tol
        self.ymean = np.mean(ytrain, 0)
        self.dy = ytrain - self.ymean
        w, Q = eigh(self.dy @ self.dy.T)
        condi = w>tol
        self.w = w[condi]
        self.Q = Q[:, condi]

    def project(self, y):
        """
        Parameters:
            y: ntest sample vectors of length N.

        Returns:
            Expansion coefficients of y in eigenbasis.
        """
        ntrain = self.dy.shape[0]
        ntest = y.shape[0]
        b = np.empty((ntrain, ntest))
        for i in range(ntrain):
            b[i,:] = (y - self.ymean) @ self.dy[i]
        return np.diag(1.0/self.w) @ self.Q.T @ b

    def lift(self, z):
        """
        Parameters:
            z: Expansion coefficients of y in eigenbasis.

        Returns:
            Reconstructed ntest sample vectors of length N.
        """
        return self.ymean + (self.dy.T @ (self.Q @ z)).T

    def features(self):
        """
        Returns:
            neig feature vectors of length N.
        """
        return self.dy.T @ self.Q


class PCA:
    """Linear dimension reduction by principle component analysis (PCA).
    This is efficient if the number of training samples ntrain is
    larger than the number of support points in independent variables.

    Parameters:
        ytrain: ntrain sample vectors of length N.
        tol: Absolute cutoff tolerance of eigenvalues.
    """
    def __init__(self, ytrain, tol=1e-2):
        self.tol = tol
        self.ymean = np.mean(ytrain, 0)
        self.dy = ytrain - self.ymean
        w, Q = eigh(self.dy.T @ self.dy)
        condi = w>tol
        self.w = w[condi]
        self.Q = Q[:, condi]

    def project(self, y):
        """
        Parameters:
            y: ntest sample vectors of length N.

        Returns:
            Expansion coefficients of y in eigenbasis.
        """
        return (y - self.ymean) @ self.Q

    def lift(self, z):
        """
        Parameters:
            z: Expansion coefficients of y in eigenbasis.

        Returns:
            Reconstructed ntest sample vectors of length N.
        """
        return self.ymean + (self.Q @ z.T).T

    def features(self):
        """
        Returns:
            neig feature vectors of length N.
        """
        return self.Q
