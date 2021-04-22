"""
Module with a collection of GP functions for the Custom surrogate.
"""

import numpy as np


def optimize(xtrain, ytrain, a0, kernel, fixed_sigma_n=False, eval_gradient=False, return_hess_inv=False):
    """Find optimal hyperparameters from initial array a0, sorted as [length_scale, scale, noise].
    Loss function is the negative log likelihood.
    Add opt_kwargs to tweak the settings in the scipy.minimize optimizer.
    Optionally return inverse of hessian matrix. This is important for the marginal likelihood calcualtion in active learning.
    """
    # TODO: add kwargs for negative_log_likelihood
    from scipy.optimize import minimize

    if fixed_sigma_n:
        sigma_n = a0[-1]
        a0 = a0[:-1]
    else:
        sigma_n = None

    a0 = np.log10(a0)

    # Avoid values too close to zero
    dy = ytrain.max() - ytrain.min()
    bounds = [(1e-6, None)] * len(a0[:-1 if fixed_sigma_n else -2]) + \
             [(1e-6 * dy, None)] + \
             [(1e-6 * dy, None)] * (1 - fixed_sigma_n)

    bounds = [(None, None)] * len(a0)

    args = [xtrain, ytrain, kernel, eval_gradient, True]
    if sigma_n is not None:
        args.append(sigma_n)

    opt_result = minimize(negative_log_likelihood, a0, args=tuple(args), bounds=bounds, jac=eval_gradient)
    if return_hess_inv:
        return opt_result.x, opt_result.hess_inv
    return 10**opt_result.x


def solve_cholesky(L, b):
    """Matrix-vector product with L being a lower triangular matrix from the Cholesky decomposition."""
    from scipy.linalg import solve_triangular
    alpha = solve_triangular(L.T, solve_triangular(L, b, lower=True, check_finite=False),
                             lower=False, check_finite=False)
    return alpha


def negative_log_likelihood_cholesky(hyp, X, y, kernel, eval_gradient=False, log_scale_hyp=False, fixed_sigma_n=False):
    """Compute the negative log-likelihood using the Cholesky decomposition of the covariance matrix
     according to Rasmussen&Williams 2006, p. 19, 113-114.

    hyp: Hyperparameter array (length_scale, sigma_f, sigma_n)
    x: Training points
    y: Function values at training points
    kernel: Function to build covariance matrix
    """

    Ky = kernel(X, X, hyp[:-2], *hyp[-2:], eval_gradient=eval_gradient)
    if eval_gradient:
        dKy = Ky[1]
        Ky = Ky[0]

    L = np.linalg.cholesky(Ky)
    alpha = solve_cholesky(L, y)
    nll = 0.5 * y.T @ alpha + np.sum(np.log(L.diagonal())) + len(X) * 0.5 * np.log(2.0 * np.pi)
    if not eval_gradient:
        return nll.item()
    KyinvaaT = invert_cholesky(L)
    KyinvaaT -= np.outer(alpha, alpha)
    dnll = 0.5 * np.trace(KyinvaaT @ dKy)
    if fixed_sigma_n:
        dnll = dnll[:-1]
        hyp = hyp[:-1]
    if log_scale_hyp:
        dnll *= hyp * np.log(10)
    return nll.item(), dnll


def negative_log_likelihood(hyp, X, y, kernel, eval_gradient=False, log_scale_hyp=False,
                            fixed_sigma_n_value=None, neig=0, max_iter=1000):
    """Compute the negative log likelihood of GP either by Cholesky decomposition or
    by finding the first neig eigenvalues. Solving for the eigenvalues is tried at maximum max_iter times.

    hyp: Hyperparameter array (length_scale, sigma_f, sigma_n)
    X: Training points
    y: Function values at training points
    kernel: Function to build covariance matrix
    """
    from scipy.sparse.linalg import eigsh

    if log_scale_hyp:
        hyp = 10**hyp
    fixed_sigma_n = fixed_sigma_n_value is not None
    if fixed_sigma_n:
        hyp = np.append(hyp, fixed_sigma_n_value)

    clip_eig = max(1e-3 * min(abs(hyp[:-2])), 1e-10)
    Ky = kernel(X, X, hyp[:-2], *hyp[-2:], eval_gradient=eval_gradient)  # Construct covariance matrix

    if eval_gradient:
        dKy = Ky[1]
        Ky = Ky[0]

    converged = False
    iteration = 0
    neig = max(neig, 1)
    while not converged:
        if not neig or neig > 0.05 * len(X):  # First try with Cholesky decomposition if neig is big
            try:
                return negative_log_likelihood_cholesky(hyp, X, y, kernel, eval_gradient=eval_gradient,
                                                        log_scale_hyp=log_scale_hyp, fixed_sigma_n=fixed_sigma_n)
            except np.linalg.LinAlgError:
                print("Warning! Fallback to eig solver!")
        w, Q = eigsh(Ky, neig, tol=clip_eig)  # Otherwise, calculate the first neig eigenvalues and eigenvectors
        if iteration >= max_iter:
            print("Reached max. iterations!")
            break
        neig *= 2  # Calculate more eigenvalues
        converged = w[0] <= clip_eig or neig >= len(X)

    # Calculate the NLL with these eigenvalues and eigenvectors
    w = np.maximum(w, 1e-10)
    alpha = Q @ (np.diag(1.0 / w) @ (Q.T @ y))
    nll = 0.5 * (y.T @ alpha + np.sum(np.log(w)) + min(neig, len(X)) * np.log(2 * np.pi))
    if not eval_gradient:
        return nll.item()

    # This is according to Rasmussen&Williams 2006, p. 114, Eq. (5.9).
    KyinvaaT = invert(Ky)
    KyinvaaT -= np.outer(alpha, alpha)

    dnll = 0.5 * np.trace(KyinvaaT @ dKy)
    if fixed_sigma_n:
        dnll = dnll[:-1]
        hyp = hyp[:-1]
    if log_scale_hyp:
        dnll *= hyp * np.log(10)
    return nll.item(), dnll


def invert_cholesky(L):
    from scipy.linalg import solve_triangular
    """Inverts a positive-definite matrix A based on a given Cholesky decomposition
       A = L^T*L."""
    return solve_triangular(L.T, solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False),
                            lower=False, check_finite=False)


def invert(K, neig=0, tol=1e-10, max_iter=1000):
    from scipy.sparse.linalg import eigsh
    """Inverts a positive-definite matrix A using either an eigendecomposition or
       a Cholesky decomposition, depending on the rapidness of decay of eigenvalues.
       Solving for the eigenvalues is tried at maximum max_iter times."""
    """
    L = np.linalg.cholesky(K)  # Cholesky decomposition of the covariance matrix
    if neig <= 0 or neig > 0.05 * len(K):  # First try with Cholesky decomposition
        try:
            return cls.invert_cholesky(L)
        except np.linalg.LinAlgError:
            print('Warning! Fallback to eig solver!')

    # Otherwise, calculate the first neig eigenvalues and eigenvectors
    w, Q = eigsh(K, neig, tol=tol)
    for iteration in range(max_iter):  # Iterate until convergence or max_iter
        if neig > 0.05 * len(K):
            try:
                return cls.invert_cholesky(L)
            except np.linalg.LinAlgError:
                print("Warning! Fallback to eig solver!")
        neig = 2 * neig  # Calculate more eigenvalues
        w, Q = eigsh(K, neig, tol=tol)

        # Convergence criterion
        if np.abs(w[0] - tol) <= tol or neig >= len(K):
            break
    if iteration == max_iter:
        print("Tried maximum number of times.")
    """
    from scipy.linalg import eigh, inv
    # w, Q = eigh(K)
    # negative_eig = (-1e-2 < w) & (w < 0)
    # w[negative_eig] = 1e-2
    # K_inv = Q @ (np.diag(1 / w) @ Q.T)
    K_inv = inv(K, check_finite=True)
    return K_inv


def marginal_variance_BBQ(Xtrain, ytrain, Xpred, kernel, hyperparameters, hess_inv, fixed_sigma_n,
                              alpha=None, predictive_variance=0):
    r"""Calculate the marginal variance to infer the next point in active learning.
    The calculation follows Osborne (2012).
    Currently, only an isotropic RBF kernel is supported.

    Derivation of the marginal variance:

        $\tilde{V}$ ... Marginal covariance matrix
        $\hat{V}$ ... Predictive variance
        $\frac{dm}{d\theta}$ ... Derivative of the predictive mean w.r.t. the hyperparameters
        $H$ ... Hessian matrix

        $$
        \begin{equation}
        \tilde{V} = \left( \frac{dm}{d\theta} \right) H^{-1} \left( \frac{dm}{d\theta} \right)^T
        \end{equation}
        $$

    Parameters:
        Xpred (ndarray): Possible prediction input points.
    Returns:
        ndarray: Sum of the actual marginal variance and the predictive variance.
    """
    # TODO: Add full derivatives as in Osborne (2021) and Garnett (2014)
    ordered_hyperparameters = [hyperparameters[key] for key in ('length_scale', 'sigma_f', 'sigma_n')]

    # If no Hessian is available, use only the predictive variance.
    if hess_inv is None:
        return predictive_variance.reshape(-1, 1)
    if fixed_sigma_n is not False:
        padding = np.zeros((len(ordered_hyperparameters), 1))
        hess_inv = np.hstack([np.vstack([hess_inv, padding[:-1].T]), padding])

    # Kernels and their derivatives
    Ky, dKy = kernel(Xtrain, Xtrain, *ordered_hyperparameters, eval_gradient=True)
    Kstar, dKstar = kernel(Xpred, Xtrain, *ordered_hyperparameters[:-1], eval_gradient=True)
    Kyinv = invert(Ky)
    if alpha is None:
        alpha = Kyinv @ ytrain
    dalpha_dl = -Kyinv @ (dKy[..., 0] @ alpha)
    dalpha_ds = -Kyinv @ (np.eye(Xtrain.shape[0]) @ alpha)

    # TODO: check derivatives
    dm = np.empty((Xpred.shape[0], len(ordered_hyperparameters), 1))
    dm[:, 0, :] = dKstar[..., 0] @ alpha - Kstar @ dalpha_dl
    dm[:, 1, :] = Kstar @ dalpha_ds
    dm[:, 2, :] = Kstar @ dalpha_ds
    dm = dm.squeeze()

    marginal_variance = dm @ (hess_inv @ dm.T)
    marginal_variance = np.diag(marginal_variance).reshape(-1, 1)

    return marginal_variance + predictive_variance


def predict_f(hyp, x, y, xtest, kernel, return_full_cov=False, neig=0):
    if len(hyp) < 3:
        hyp[2] = 0  # sigma_n
    Ky = kernel(x, x, hyp[:-2], *hyp[-2:])
    Kstar = kernel(x, xtest, hyp[:-2], hyp[-2])
    Kstarstar = kernel(xtest, xtest, *hyp)
    Kyinv = invert(Ky, neig, 1e-6*hyp[-1])
    fstar = Kstar.T @ (Kyinv @ y)
    vstar = Kstarstar - Kstar.T @ (invert(Ky) @ Kstar)
    vstar = np.maximum(vstar, 1e-10)  # Assure a positive variance
    if not return_full_cov:
        vstar = np.diag(vstar)
    return fstar, vstar
