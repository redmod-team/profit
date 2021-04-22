"""
Module which includes kernels for the Custom surrogate.
"""

import numpy as np


def RBF(X, Y, length_scale=1, sigma_f=1, sigma_n=0, eval_gradient=False):
    x1 = X / length_scale
    x2 = Y / length_scale
    dx = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
    dx2 = np.linalg.norm(dx, axis=-1) ** 2
    K = sigma_f ** 2 * np.exp(-0.5 * dx2) + sigma_n ** 2 * np.eye(*dx2.shape)
    if eval_gradient:
        len_length_scale = len(length_scale) if hasattr(length_scale, '__iter__') else 1
        dK = np.empty((K.shape[0], K.shape[1], len_length_scale + 2))
        K_pure = K - sigma_n ** 2 * np.eye(*K.shape)
        if len_length_scale == 1:
            dK[..., 0] = dx2 / length_scale * K_pure
        else:
            dK[..., :-2] = dx ** 2 / length_scale * K_pure[..., np.newaxis]
        dK[..., -2] = 2 / sigma_f * K_pure
        dK[..., -1] = 2 * sigma_n * np.eye(*K_pure.shape)
        return K, dK
    return K


def LinearEmbedding(X, Y, R, sigma_f=1e-6, sigma_n=0, eval_gradient=False):
    X = np.atleast_2d(X)
    # We want X @ R.T  -> (n x D) @ (D x d)  -> second dim of R must be D, so first must be R.size / D
    R = R.reshape(R.size // X.shape[-1], X.shape[-1])
    X1 = X @ R.T
    X2 = X @ R.T if Y is None else Y @ R.T
    dX = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    dX2 = np.linalg.norm(dX, axis=-1) ** 2
    K = sigma_f ** 2 * np.exp(-0.5 * dX2) + sigma_n ** 2 * np.eye(*dX2.shape)
    if eval_gradient:
        # TODO: Check gradient dK/dR. For a 2x2 R it should be (4,) dimensional.
        dK = np.empty((K.shape[0], K.shape[1], R.size + 2))
        K_pure = K - sigma_n ** 2 * np.eye(*K.shape)
        dK[..., :-2] = np.einsum('ijk,kl', dX ** 2, R) * K_pure[..., np.newaxis]
        dK[..., -2] = 2 / sigma_f * K_pure
        dK[..., -1] = 2 * sigma_n * np.eye(*K_pure.shape)
        return K, dK
    return K
