import numpy as np
from .SuSiE_class import ResultSuSiE
from .Funmap_class import ResultFunmap


def SUSIE(z, R, n, L=10, max_iter=100, tol=5e-5):
    """
    Runs the SuSiE (Sum of Single Effects) algorithm for sparse regression.

    Args:
        z (numpy.ndarray): A 1D array of z-scores.
        R (numpy.ndarray): A 2D array representing the correlation matrix.
        n (int): The sample size.
        L (int, optional): The maximum number of causal variables. Default is 10.
        max_iter (int, optional): The maximum number of iterations. Default is 100.
        tol (float, optional): The convergence tolerance. Default is 5e-5.

    Returns:
        ResultSuSiE: An instance of the ResultSuSiE class containing the SuSiE results.
    """
    p = z.shape[0]

    adj = (n - 1) / (z ** 2 + n - 2)
    z = np.sqrt(adj.astype(float)) * z.astype(float)

    yty = n - 1
    XtX = (n - 1) * R
    Xty = np.sqrt(n - 1) * z

    XtX_d = np.diag(XtX)
    X_csd = np.sqrt(XtX_d / (n - 1))
    X_csd[X_csd == 0] = 1
    XtX = ((1 / X_csd) * XtX).T / X_csd
    Xty = Xty / X_csd
    XtX_d = np.diag(XtX)

    s = ResultSuSiE(L, p)
    s.fit(L, n, R, XtX, Xty, yty, XtX_d, max_iter, tol)

    return s


def FUNMAP(z, R, A, n, L=10, max_iter=100, tol=5e-5):
    """
    Runs the FUNMAP (Functional Mapping of Annotations) algorithm for sparse regression with functional annotations.

    Args:
        z (numpy.ndarray): A 1D array of z-scores.
        R (numpy.ndarray): A 2D array representing the correlation matrix.
        A (numpy.ndarray): A 2D array representing the functional annotations.
        n (int): The sample size.
        L (int, optional): The maximum number of causal variables. Default is 10.
        max_iter (int, optional): The maximum number of iterations. Default is 100.
        tol (float, optional): The convergence tolerance. Default is 5e-5.

    Returns:
        ResultFunmap: An instance of the ResultFunmap class containing the FUNMAP results.
    """
    p = A.shape[0]
    m = A.shape[1]

    adj = (n - 1) / (z ** 2 + n - 2)
    z = np.sqrt(adj.astype(float)) * z.astype(float)

    yty = n - 1
    XtX = (n - 1) * R
    Xty = np.sqrt(n - 1) * z

    XtX_d = np.diag(XtX)
    X_csd = np.sqrt(XtX_d / (n - 1))
    X_csd[X_csd == 0] = 1
    XtX = ((1 / X_csd) * XtX).T / X_csd
    Xty = Xty / X_csd
    XtX_d = np.diag(XtX)

    # Stage 1
    s_init1 = ResultSuSiE(L, p)
    s_init1.fit(L, n, R, XtX, Xty, yty, XtX_d, max_iter, tol)

    # Stage 2
    s_init2 = ResultFunmap(L, p, m)
    s_init2.init_susie(p, s_init1.alpha, s_init1.XtXr, s_init1.mu, s_init1.mu2, s_init1.sigma2, s_init1.V, s_init1.sets)
    s_init2.fit(L, p, m, n, R, A, XtX, Xty, yty, XtX_d, max_iter, tol)

    # Stage 3
    s = ResultFunmap(L, p, m)
    s.init_funmap(s_init2.prior_weights, s_init2.alpha, s_init2.XtXr, s_init2.mu, s_init2.mu2, s_init2.mu_w,
                  s_init2.xi2, s_init2.rho, s_init2.V, s_init2.Sigma_w, s_init2.sigma_w2, s_init2.sigma2)
    s.fit(L, p, m, n, R, A, XtX, Xty, yty, XtX_d, max_iter, tol)

    return s
