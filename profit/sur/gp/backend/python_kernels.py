"""
Module which includes kernels for the Custom surrogate.
"""

import numpy as np


def RBF(X, Y, length_scale=1, sigma_f=1, sigma_n=0, eval_gradient=False):
    r"""Squared exponential kernel, also call Radial-Basis-Function kernel (RBF).

    The RBF kernel is one of the most common kernels used and is especially suited for smooth functions.

    $$
    \begin{equation}
    K(X, Y) = \sigma_f^2 \exp(-\frac{1}{2} \frac{\lvert X-Y \rvert}{l^2}) + \sigma_n^2~\mathbf{1}
    \end{equation}
    $$

    Parameters:
        X (ndarray): Input points.
        Y (ndarray): Other input points.
        length_scale (float/ndarray): Length scale $l$ of the kernel function.
        sigma_f (float): Scale $\sigma_f$.
        sigma_n (float): Additive noise $\sigma_n$.
        eval_gradient (bool): Indicates if the gradient with respect to the hyperparameters $l$, $\sigma_f$ and
            $\sigma_n$ should be returned.

    Returns:
        tuple/ndarray: A tuple containing:
            - K (ndarray): Kernel matrix of size (X.shape[0], Y.shape[0]).
            - dK (ndarray): Derivative of the kernel w.r.t. the hyperparameters $l$, $\sigma_f$ and $\sigma_n$.
            If eval_gradient is False, only K is returned.
    """

    x1 = X / length_scale
    x2 = Y / length_scale
    dx = (
        x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
    )  # Distance between points in all dimensions.
    dx2 = np.linalg.norm(dx, axis=-1) ** 2
    K = sigma_f**2 * np.exp(-0.5 * dx2) + sigma_n**2 * np.eye(*dx2.shape)
    if eval_gradient:
        len_length_scale = (
            len(length_scale) if hasattr(length_scale, "__iter__") else 1
        )  # Check for isotropic kernel.
        dK = np.empty((K.shape[0], K.shape[1], len_length_scale + 2))
        K_pure = K - sigma_n**2 * np.eye(*K.shape)  # Kernel without noise sigma_n
        if len_length_scale == 1:
            dK[..., 0] = dx2 / length_scale * K_pure  # dK/dl
        else:
            dK[..., :-2] = (
                dx**2 / length_scale * K_pure[..., np.newaxis]
            )  # dK/dl for n-D length_scale
        dK[..., -2] = 2 / sigma_f * K_pure  # dK/dsigma_f
        dK[..., -1] = 2 * sigma_n * np.eye(*K_pure.shape)  # dK/dsigma_n
        return K, dK
    return K


def LinearEmbedding(X, Y, R, sigma_f=1e-6, sigma_n=0, eval_gradient=False):
    r"""The linear embedding kernel according to Garnett (2014) is a generalisation of the RBF kernel.

     The RBF kernel with different length scales in each dimension (ARD kernel) is a special case where only
     the diagonal elements of the matrix $R$ are non-zero and represent inverse length scales.
     The ARD kernel is suited to detect relevant dimensions of the input data. In contrast,
     the linear embedding kernel can also detect relevant linear combinations of dimensions, e.g. if a function only
     varies in the direction $x1 + x2$. Thus, the linear embedding kernel can be used to find a lower dimensional
     representation of the data in aribitrary directions.

     $$
     \begin{equation}
     K(X, Y) = \sigma_f^2 \exp(-\frac{1}{2} (X-Y)R^T R(X-Y)^T) + \sigma_n^2~\mathbf{1}
     \end{equation}
     $$

     Here, we use the convention that the data $X$ and $Y$ are of shape (n, D) and therefore R has to be of shape
     (d, D) where D > d to find a lower dimensional representation.

     Parameters:
         X (ndarray): Input points of shape (n, D)
         Y (ndarray): Other input ponits.
         R (ndarray): Matrix or flattened array of hyperparameters. It is automatically reshaped to fit the input data.
             Every matrix element represents a hyperparameter.
         sigma_f (float): Scale $\sigma_f$.
         sigma_n (float): Additive noise $\sigma_n$.
         eval_gradient (bool): Indicates if the gradient with respect to the hyperparameters $R_{ij}$, $\sigma_f$ and
             $\sigma_n$ should be returned.

    Returns:
         tuple/ndarray: A tuple containing:
             - K (ndarray): Kernel matrix of size (X.shape[0], Y.shape[0]).
             - dK (ndarray): Derivative of the kernel w.r.t. the hyperparameters $R_{ij}$, $\sigma_f$ and $\sigma_n$.
             If eval_gradient is False, only K is returned.
    """

    X = np.atleast_2d(X)
    # X @ R.T is of shape (n x D) @ (D x d)  -> second dimension of R must be D, so first must be R.size / D
    R = R.reshape(R.size // X.shape[-1], X.shape[-1])
    X1 = X @ R.T
    X2 = X @ R.T if Y is None else Y @ R.T
    dX = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    dX2 = np.linalg.norm(dX, axis=-1) ** 2
    K = sigma_f**2 * np.exp(-0.5 * dX2) + sigma_n**2 * np.eye(*dX2.shape)
    if eval_gradient:
        # TODO: Check gradient dK/dR. For a 2x2 R it should be (4,) dimensional.
        dK = np.empty((K.shape[0], K.shape[1], R.size + 2))
        K_pure = K - sigma_n**2 * np.eye(*K.shape)  # Kernel without noise sigma_n
        dK[..., :-2] = (
            np.einsum("ijk,kl", dX**2, R) * K_pure[..., np.newaxis]
        )  # dK/dR_ij
        dK[..., -2] = 2 / sigma_f * K_pure  # dK/dsigma_f
        dK[..., -1] = 2 * sigma_n * np.eye(*K_pure.shape)  # dK/dsigma_n
        return K, dK
    return K
