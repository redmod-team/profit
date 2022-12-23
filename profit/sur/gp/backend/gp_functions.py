"""
Collection of functions for the Custom GPSurrogate.
"""

import numpy as np


def optimize(
    xtrain,
    ytrain,
    a0,
    kernel,
    fixed_sigma_n=False,
    eval_gradient=False,
    return_hess_inv=False,
):
    r"""Finds optimal hyperparameters from initial array a0, sorted as [length_scale, scale, noise].

    The objective function, which is minimized, is the negative log likelihood.
    The hyperparameters are transformed logarithmically before the optimization to enhance its flexibility. They are
    transformed back afterwards.

    Parameters:
        xtrain (ndarray):
        ytrain (ndarray):
        a0 (ndarray): Flat array of initial hyperparameters.
        kernel (function): Function of the kernel, not the actual matrix.
        fixed_sigma_n (bool): If the noise $\sigma_n$ should be fixed during optimization.
        eval_gradient (bool): Whether the analytic derivatives of the kernel and likelihood are used in the
            optimization.
        return_hess_inv (bool): If True, returns the inverse Hessian matrix from the optimization result.
            This is important for advanced active learning.

    Returns:
        tuple: A tuple containing:
            - opt_hyperparameters (ndarray): Flat array of optimized hyperparameters.
            - hess_inv (scipy.optimize.LbfgsInvHessProduct): Inverse Hessian matrix in the form of a scipy linear
            operator.
            If return_hess_inv is False, only the optimized hyperparameters are returned.
    """

    from scipy.optimize import minimize

    if fixed_sigma_n:
        sigma_n = a0[-1]
        a0 = a0[:-1]
    else:
        sigma_n = None

    a0 = np.log10(a0)

    # Additional arguments for the negative_log_likelihood function.
    args = [xtrain, ytrain, kernel, eval_gradient, True]

    # If sigma_n should be kept fixed during optimization.
    if sigma_n is not None:
        args.append(sigma_n)

    opt_result = minimize(
        negative_log_likelihood,
        a0,
        method="bfgs",
        args=tuple(args),
        bounds=None,
        jac=eval_gradient,
    )
    if return_hess_inv:
        return 10**opt_result.x, opt_result.hess_inv
    return 10**opt_result.x


def solve_cholesky(L, b):
    r"""Solves a linear equation with a lower triangular matrix L from the Cholesky decomposition.

    In the context of GP's the $L$ is the Cholesky decomposition of the training kernel matrix and $b$ is the training
    output data.

    $$
    \begin{equation}
    \alpha = L^T L b
    \end{equation}
    $$

    Parameters:
        L (ndarray): Lower triangular matrix.
        b (ndarray): A vector.

    Returns:
        ndarray
    """

    from scipy.linalg import solve_triangular

    alpha = solve_triangular(
        L.T,
        solve_triangular(L, b, lower=True, check_finite=False),
        lower=False,
        check_finite=False,
    )
    return alpha


def negative_log_likelihood_cholesky(
    hyp, X, y, kernel, eval_gradient=False, log_scale_hyp=False, fixed_sigma_n=False
):
    r"""Computes the negative log likelihood using the Cholesky decomposition of the covariance matrix.

    The calculation follows Rasmussen&Williams 2006, p. 19, 113-114.

    $$
    \begin{align}
    NL &= \frac{1}{2} y^T \alpha + tr(\log(L)) + \frac{n}{2} \log(2 \pi) \\
    \frac{dNL}{d\theta} &= \frac{1}{2} tr\left( (K_y^{-1} - \alpha \alpha^T) \frac{\partial K}{\partial \theta} \right) \\
    \alpha &= K_y^{-1} y
    \end{align}
    $$

    Parameters:
        hyp (ndarray): Flat hyperparameter array [length_scale, sigma_f, sigma_n]. They can also be already
            log transformed.
        X (ndarray): Training input points.
        y (ndarray): Observed training output.
        kernel (function): Function to build the covariance matrix.
        eval_gradient (bool): If the analytic gradient of the negative log likelihood w.r.t. the hyperparameters should
            be returned.
        log_scale_hyp (bool): Whether the hyperparameters are log transformed. This is important for the gradient
            calculation.
        fixed_sigma_n (bool): If the noise $\sigma_n$ is kept fixed. In this case, there is no gradient with respect to
            sigma_n.

    Returns:
        tuple: A tuple containing:
            - nll (float): The negative log likelihood.
            - dnll (ndarray): The derivative of the negative log likelihood w.r.t. to the hyperparameters.
            If eval_gradient is False, only nll is returned.
    """

    Ky = kernel(X, X, hyp[:-2], *hyp[-2:], eval_gradient=eval_gradient)
    if eval_gradient:
        dKy = Ky[1]
        Ky = Ky[0]

    L = np.linalg.cholesky(Ky)
    alpha = solve_cholesky(L, y)
    nll = (
        0.5 * y.T @ alpha
        + np.sum(np.log(L.diagonal()))
        + len(X) * 0.5 * np.log(2.0 * np.pi)
    )
    if not eval_gradient:
        return nll.item()
    KyinvaaT = invert_cholesky(L)
    KyinvaaT -= np.outer(alpha, alpha)
    dnll = 0.5 * np.trace(KyinvaaT @ dKy)  # Rasmussen&Williams p. 114, eq. 5.9
    if fixed_sigma_n:
        dnll = dnll[:-1]
        hyp = hyp[:-1]
    if log_scale_hyp:
        # Chain rule
        dnll *= hyp * np.log(10)
    return nll.item(), dnll


def negative_log_likelihood(
    hyp,
    X,
    y,
    kernel,
    eval_gradient=False,
    log_scale_hyp=False,
    fixed_sigma_n_value=None,
    neig=0,
    max_iter=1000,
):
    r"""Computes the negative log likelihood either by a Cholesky- or an Eigendecomposition.

    First, the Cholesky decomposition is tried. If this results in a LinAlgError, the biggest eigenvalues are
    calculated until convergence or until the maximum iterations are reached.
    The eigenvalues are cut off at $1e-10$ due ensure numerical stability.

    $$
    \begin{align}
    NL &= \frac{1}{2} \left( y^T \alpha + tr(\log(\lambda)) + n_{eig} \log(2 \pi) \right) \\
    \frac{dNL}{d\theta} &= \frac{1}{2} tr\left( (K_y^{-1} - \alpha \alpha^T) \frac{\partial K}{\partial \theta} \right) \\
    \alpha &= v (\lambda^{-1} (v^T y))
    \end{align}
    $$

    Parameters:
        hyp (ndarray): Flat hyperparameter array [length_scale, sigma_f, sigma_n].
        X (ndarray): Training input points.
        y (ndarray): Observed training output.
        kernel (function): Function to build the covariance matrix.
        eval_gradient (bool): If the analytic gradient of the negative log likelihood w.r.t. the hyperparameters should
            be returned.
        log_scale_hyp (bool): Whether the hyperparameters are log transformed. This is important for the gradient
            calculation.
        fixed_sigma_n_value (float): The value of the fixed noise $\sigma_n$. If it should be optimizied as well, this
            should be None.
        neig (int): Initial number of eigenvalues to calculate if the Cholesky decomposition is not successful.
            This is doubled during every iteration.
        max_iter (int): Maximum number of iterations of the eigenvalue solver until convergence must be reached.

    Returns:
        tuple: A tuple containing:
            - nll (float): The negative log likelihood.
            - dnll (ndarray): The derivative of the negative log likelihood w.r.t. to the hyperparameters.
            If eval_gradient is False, only nll is returned.
    """

    from scipy.sparse.linalg import eigsh

    if log_scale_hyp:
        hyp = 10**hyp
    fixed_sigma_n = fixed_sigma_n_value is not None
    if fixed_sigma_n:
        hyp = np.append(hyp, fixed_sigma_n_value)

    clip_eig = max(1e-3 * min(abs(hyp[:-2])), 1e-10)
    Ky = kernel(
        X, X, hyp[:-2], *hyp[-2:], eval_gradient=eval_gradient
    )  # Construct covariance matrix

    if eval_gradient:
        dKy = Ky[1]
        Ky = Ky[0]

    converged = False
    iteration = 0
    neig = max(neig, 1)
    while not converged:
        if not neig or neig > 0.05 * len(
            X
        ):  # First try with Cholesky decomposition if neig is big
            try:
                return negative_log_likelihood_cholesky(
                    hyp,
                    X,
                    y,
                    kernel,
                    eval_gradient=eval_gradient,
                    log_scale_hyp=log_scale_hyp,
                    fixed_sigma_n=fixed_sigma_n,
                )
            except np.linalg.LinAlgError:
                print("Warning! Fallback to eig solver!")
        w, Q = eigsh(
            Ky, neig, tol=clip_eig
        )  # Otherwise, calculate the first neig eigenvalues and eigenvectors
        if iteration >= max_iter:
            print("Reached max. iterations!")
            break
        neig *= 2  # Calculate more eigenvalues
        converged = w[0] <= clip_eig or neig >= len(X)

    # Calculate the NLL with these eigenvalues and eigenvectors
    w = np.maximum(w, 1e-10)
    alpha = Q @ (np.diag(1.0 / w) @ (Q.T @ y))
    nll = 0.5 * (
        y.T @ alpha + np.sum(np.log(w)) + min(neig, len(X)) * np.log(2 * np.pi)
    )
    if not eval_gradient:
        return nll.item()

    KyinvaaT = invert(Ky)
    KyinvaaT -= np.outer(alpha, alpha)

    dnll = 0.5 * np.trace(KyinvaaT @ dKy)  # Rasmussen&Williams p. 114, eq. 5.9
    if fixed_sigma_n:
        dnll = dnll[:-1]
        hyp = hyp[:-1]
    if log_scale_hyp:
        # Chain rule
        dnll *= hyp * np.log(10)
    return nll.item(), dnll


def invert_cholesky(L):
    r"""Inverts a positive-definite matrix based on a Cholesky decomposition.

    This is used to invert the covariance matrix.

    Parameters:
        L (ndarray): Lower triangular matrix from a Cholesky decomposition.

    Returns:
        ndarray: Inverse of the matrix $L^T L$
    """
    from scipy.linalg import solve_triangular

    return solve_triangular(
        L.T,
        solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False),
        lower=False,
        check_finite=False,
    )


def invert(K, neig=0, tol=1e-10, eps=1e-6, max_iter=1000):
    """Inverts a positive-definite matrix using either a Cholesky- or an Eigendecomposition.

    The solution method depends on the rapidness of decay of the eigenvalues.

    Parameters:
        K (np.ndarray): Kernel matrix.
        neig (int): Initial number of eigenvalues to calculate if the Cholesky decomposition is not successful.
            This is doubled during every iteration.
        tol (float): Convergence criterion for the eigenvalues.
        eps (float): Small number to be added to diagonal of kernel matrix to ensure positive definiteness.
        max_iter (int): Maximum number of iterations of the eigenvalue solver until convergence must be reached.

    Returns:
        np.ndarray: Inverse covariance matrix.
    """
    from scipy.sparse.linalg import eigsh

    try:
        L = np.linalg.cholesky(K)  # Cholesky decomposition of the covariance matrix
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(K + eps * np.eye(K.shape[0], K.shape[1]))
    if neig <= 0 or neig > 0.05 * len(K):  # First try with Cholesky decomposition
        try:
            return invert_cholesky(L)
        except np.linalg.LinAlgError:
            print("Warning! Fallback to eig solver!")

    # Otherwise, calculate the first neig eigenvalues and eigenvectors
    w, Q = eigsh(K, neig, tol=tol)
    for iteration in range(max_iter):  # Iterate until convergence or max_iter
        if neig > 0.05 * len(K):
            try:
                return invert_cholesky(L)
            except np.linalg.LinAlgError:
                print("Warning! Fallback to eig solver!")
        neig = 2 * neig  # Calculate more eigenvalues
        w, Q = eigsh(K, neig, tol=tol)

        # Convergence criterion
        if np.abs(w[0] - tol) <= tol or neig >= len(K):
            break
    if iteration == max_iter:
        print("Tried maximum number of times.")

    negative_eig = (-1e-2 < w) & (w < 0)
    w[negative_eig] = 1e-2
    K_inv = Q @ (np.diag(1 / w) @ Q.T)
    return K_inv


def marginal_variance_BBQ(
    Xtrain,
    ytrain,
    Xpred,
    kernel,
    hyperparameters,
    hess_inv,
    fixed_sigma_n=False,
    alpha=None,
    predictive_variance=0,
):
    r"""Calculates the marginal variance to infer the next point in active learning.

    The calculation follows Osborne (2012).

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
        Xtrain (ndarray): Input training points.
        ytrain (ndarray): Observed output daat.
        Xpred (ndarray): Possible prediction input points.
        kernel (function): Function to build the covariance matrix.
        hyperparameters (dict): Dictionary of the hyperparameters.
        hess_inv (ndarray): Inverse Hessian matrix.
        fixed_sigma_n (bool): If the noise $\sigma_n$ was fixed during optimization. Then, the Hessian has to be
            padded with zeros.
        alpha (ndarray): If available, the result of $K_y^{-1} y$, else None.
        predictive_variance (ndarray): Predictive variance only. This is added to the marginal variance.

    Returns:
        ndarray: Sum of the actual marginal variance and the predictive variance.
    """

    # TODO: Add full derivatives as in Osborne (2021) and Garnett (2014)
    ordered_hyperparameters = [
        hyperparameters[key] for key in ("length_scale", "sigma_f", "sigma_n")
    ]

    # If no Hessian is available, use only the predictive variance.
    if hess_inv is None:
        return predictive_variance.reshape(-1, 1)

    if fixed_sigma_n:
        padding = np.zeros((len(ordered_hyperparameters), 1))
        hess_inv = np.hstack([np.vstack([hess_inv, padding[:-1].T]), padding])

    # Kernels and their derivatives
    Ky, dKy = kernel(Xtrain, Xtrain, *ordered_hyperparameters, eval_gradient=True)
    Kstar, dKstar = kernel(
        Xpred, Xtrain, *ordered_hyperparameters[:-1], eval_gradient=True
    )
    Kyinv = invert(Ky)
    if alpha is None:
        alpha = Kyinv @ ytrain
    dalpha_dl = -Kyinv @ (dKy[..., 0] @ alpha)
    dalpha_dsigma_f = -Kyinv @ (dKy[..., 1] @ alpha)
    dalpha_dsigma_n = -Kyinv @ (
        2 * hyperparameters["sigma_n"] * np.eye(Xtrain.shape[0]) @ alpha
    )

    dm = np.empty((Xpred.shape[0], len(ordered_hyperparameters), 1))
    dm[:, 0, :] = dKstar[..., 0] @ alpha + Kstar @ dalpha_dl
    dm[:, 1, :] = dKstar[..., 1] @ alpha + Kstar @ dalpha_dsigma_f
    dm[:, 2, :] = Kstar @ dalpha_dsigma_n
    dm = dm.squeeze()

    marginal_variance = dm @ (hess_inv @ dm.T)
    marginal_variance = np.diag(marginal_variance).reshape(-1, 1)

    return marginal_variance + predictive_variance


def predict_f(hyp, x, y, xtest, kernel, return_full_cov=False, neig=0):
    """Predicts values given only a set of hyperparameters and a kernel.

    This function is independent of the surrogates and used to quickly predict a function output with specific
    hyperparameters.

    The calculation follows Rasmussen&Williams, p. 16, eq. 23-24.

    Parameters:
        hyp (ndarray): Flat array of hyperparameters, sorted as [length_scale, sigma_f, sigma_n].
        x (ndarray): Input training points.
        y (ndarray): Observed output data.
        xtest (ndarray): Prediction points.
        kernel (function): Function to build the covariance matrix.
        return_full_cov (bool): If True, returns the full covariance matrix, otherwise only its diagonal.
        neig (int): Initial number of eigenvalues to be computed during the inversion of the covariance matrix.

    Returns:
        tuple: A tuple containing:
            - fstar (ndarray): Posterior mean.
            - vstar (ndarray): Posterior covariance matrix or its diagonal.
    """

    if len(hyp) < 3:
        hyp[2] = 0  # sigma_n
    Ky = kernel(x, x, hyp[:-2], *hyp[-2:])
    Kstar = kernel(x, xtest, hyp[:-2], hyp[-2])
    Kstarstar = kernel(xtest, xtest, *hyp)
    Kyinv = invert(Ky, neig, 1e-6 * hyp[-1])
    fstar = Kstar.T @ (Kyinv @ y)
    vstar = Kstarstar - Kstar.T @ (invert(Ky) @ Kstar)
    vstar = np.maximum(vstar, 1e-10)  # Assure a positive variance
    if not return_full_cov:
        vstar = np.diag(vstar)
    return fstar, vstar
