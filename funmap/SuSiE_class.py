"""
This module contains a class ResultSuSiE and several utility functions for performing
statistical analysis and computations related to the SuSiE algorithm.
"""


import numpy as np
import pandas as pd
from numpy import matmul
from scipy.stats import norm
from scipy.optimize import minimize_scalar

class ResultSuSiE:
    """
    A class to store and manage the results of the SuSiE (Sum of Single Effects) model.

    Attributes:
        prior_weights (numpy.ndarray): Prior weights for the variables.
        alpha (numpy.ndarray): The posterior expectation of gamma.
        mu (numpy.ndarray): The posterior expectation of b.
        mu2 (numpy.ndarray): The posterior expectation of b^2.
        sigma2 (float): Residual variance.
        V (numpy.ndarray): Prior variances for each component.
        sets (dict): Dictionary containing information about credible sets.
        pip (numpy.ndarray): Posterior inclusion probabilities for each variable.
        converged (bool): Flag indicating if the algorithm has converged.
    """

    def __init__(self, L, p):
        """
        Initialize a ResultSuSiE object.
        """
        self.prior_weights = np.ones(p) / p
        self.alpha = np.ones((L, p)) / p
        self.mu = np.zeros((L, p))
        self.mu2 = np.zeros((L, p))
        self.XtXr = np.ones(p) / p
        self.KL = np.full(L, np.nan)
        self.lbf = np.full(L, np.nan)
        self.lbf_variable = np.full((L, p), np.nan)
        self.sigma2 = 1
        self.V = np.ones(L) * 0.2
        self.sets = None
        self.null_index = 0
        self.pip = np.ones(p) / p
        self.converged = False

    def fit(self, L, n, R, XtX, Xty, yty, XtX_d, max_iter, tol):
        """
        Fit the SuSiE model to the data.
        """
        elbo = np.full(max_iter + 1, np.nan)
        elbo[0] = -np.inf
        niter = 0

        for i in range(1, max_iter + 1):

            self.update_each_component(L, XtX, Xty, XtX_d)
            self.update_residual_variance(n, XtX, Xty, yty, XtX_d)
            elbo[i] = self.get_elbo_SuSiE(n, XtX, Xty, yty, XtX_d)

            print("objective:", elbo[i])

            if (elbo[i] - elbo[i-1]) < tol * np.abs(elbo[i]):
                self.converged = True
                break
            niter = i

        print("Stage1: iterations={}".format(niter))

        self.get_cs(L, R)
        self.get_pip()

    def update_each_component(self, L, XtX, Xty, XtX_d):
        """
        Update the parameters for each component.
        """
        for l in range(L):
            self.XtXr = self.XtXr - matmul(XtX, (self.alpha[l] * self.mu[l]))
            XtR = Xty - self.XtXr
            res = fit_SER_SuSiE(XtX_d, XtR, self.sigma2, self.prior_weights)
            self.mu[l] = res['mu']
            self.alpha[l] = res['alpha']
            self.mu2[l] = res['mu2']
            self.V[l] = res['V']
            self.lbf[l] = res['lbf_model']
            self.lbf_variable[l] = res['lbf']
            self.KL[l] = -res['lbf_model'] + SER_posterior_e_loglik_ss(XtX_d, XtR, self.sigma2,
                                                                       res['alpha'] * res['mu'],
                                                                       res['alpha'] * res['mu2'])
            self.XtXr = self.XtXr + matmul(XtX, self.alpha[l] * self.mu[l])

    def update_residual_variance(self, n, XtX, Xty, yty, XtX_d):
        """
        Optimize the residual variance for the SuSiE model.
        """
        estimate_sigma2 = (1 / n) * self.get_ER2_ss(XtX, Xty, yty, XtX_d)
        self.sigma2 = np.maximum(1e-4, estimate_sigma2)

    def get_elbo_SuSiE(self, n, XtX, Xty, yty, XtX_d):
        """
        Calculate the evidence lower bound of the SuSiE model.
        """
        return self.Eloglik(n, XtX, Xty, yty, XtX_d) - np.sum(self.KL)

    def Eloglik(self, n, XtX, Xty, yty, XtX_d):
        """
        Calculate the expectation of log-likelihood for the SuSiE model.
        """
        return -(n / 2)*np.log(2 * np.pi * self.sigma2) - (1 / (2 * self.sigma2))*self.get_ER2_ss(XtX, Xty, yty, XtX_d)

    def get_ER2_ss(self, XtX, Xty, yty, XtX_d):
        """
        Calculate the expectation of residual sum of squares.
        """
        B = self.alpha * self.mu
        XB2 = np.sum(matmul(B, XtX) * B)
        betabar = np.sum(B, axis=0)
        postb2 = self.alpha * self.mu2
        return yty - 2 * np.sum(betabar * Xty) + np.sum(betabar * matmul(XtX, betabar)) - XB2 + np.sum(XtX_d * postb2)

    def get_cs(self, L, R, coverage=0.95, min_abs_corr=0.5, n_purity=100):
        """
        Calculate the credible set of a set of variables.
        """
        null_index = self.null_index
        include_idx = self.V > 1e-9
        status = in_CS(self.alpha, coverage)
        cs = np.array([np.where(status[i0, ] != 0)[0] for i0 in range(L)], dtype=object)
        claimed_coverage = np.array([np.sum(self.alpha[l0, :][(cs[l0]).astype(int)]) for l0 in range(L)])
        include_idx = include_idx & np.array([(cs[i0]).shape[0] > 0 for i0 in range(len(cs))])
        cs_dataframe = pd.DataFrame({i0: pd.Series(row) for i0, row in enumerate(cs.tolist())}).T
        include_idx = include_idx & ~cs_dataframe.duplicated(keep='first').values
        if not include_idx.any():
            self.sets = {'cs': None, 'coverage': None, 'requested_coverage': coverage}
        cs = cs[include_idx]
        claimed_coverage = claimed_coverage[include_idx]

        purity = []
        for i0 in range(cs.shape[0]):
            if null_index > 0 and null_index in cs[i0]:
                purity.append([-9, -9, -9])
            else:
                purity.append(get_purity(R, cs[i0], n0=n_purity))

        purity = pd.DataFrame(purity, columns=['min_abs_corr', 'mean_abs_corr', 'median_abs_corr'])
        threshold = min_abs_corr
        is_pure = np.where(purity.iloc[:, 0] >= threshold)[0]
        if len(is_pure) > 0:
            cs = cs[is_pure]
            purity = purity.iloc[is_pure, ]
            row_names = ["L" + str(i0) for i0 in np.where(include_idx)[0][is_pure]]
            cs_dict = dict(zip(row_names, cs))
            purity.index = row_names
            ordering = np.argsort(purity.iloc[:, 0])[::-1]
            self.sets = {'cs': cs_dict, 'purity': purity.iloc[ordering, ],
                         'cs_index': np.where(include_idx)[0][is_pure][ordering],
                         'coverage': pd.Series(claimed_coverage)[ordering],
                         'requested_coverage': coverage}
        else:
            self.sets = {'cs': None, 'coverage': None, 'requested_coverage': coverage}

    def get_pip(self, prune_by_cs=False, prior_tol=1e-9):
        """
        Calculate the PIP of a set of variables.
        """
        include_idx = np.where(np.array(self.V) > prior_tol)[0]

        # Only consider variables in reported CS.
        if prune_by_cs:
            include_idx = np.intersect1d(include_idx, self.sets['cs_index'])

        # now extract relevant rows from alpha matrix
        if len(include_idx) > 0:
            res = self.alpha[include_idx, :]
        else:
            res = np.zeros((1, self.alpha.shape[1]))

        self.pip = 1 - np.prod(1 - res, axis=0)


def fit_SER_SuSiE(XtX_d, y0, residual_variance, prior_weights):
    """
    Fit a single component for the SuSiE model.
    """
    betahat = (1 / XtX_d) * y0
    shat2 = residual_variance / XtX_d

    V0 = optimize_prior_variance(betahat, shat2, prior_weights)
    lbf = norm.logpdf(betahat, 0, np.sqrt(V0 + shat2)) - norm.logpdf(betahat, 0, np.sqrt(shat2))
    lbf[np.isinf(shat2)] = 0
    maxlbf = np.max(lbf)
    w = np.exp(lbf - maxlbf)
    w_weighted = w * prior_weights
    weighted_sum_w = np.sum(w_weighted)
    alpha = w_weighted / weighted_sum_w
    post_var = 1 / (1 / V0 + XtX_d / residual_variance) if V0 else 0
    post_mean = (1 / residual_variance) * post_var * y0
    post_mean2 = post_var + post_mean ** 2
    lbf_model = maxlbf + np.log(weighted_sum_w)

    return dict(alpha=alpha, mu=post_mean, mu2=post_mean2, lbf=lbf, lbf_model=lbf_model, V=V0)


def optimize_prior_variance(betahat, shat2, prior_weights):
    """
    Optimize the prior variance for the SuSiE model.
    """
    logV_opt = minimize_scalar(neg_loglik_logscale, bounds=(-30, 15), args=(betahat, shat2, prior_weights),
                               method='bounded').x
    V = np.exp(logV_opt)
    return V


def loglik(V, betahat, shat2, prior_weights):
    """
    Calculate the log-likelihood for the SuSiE model.
    """
    lbf = norm.logpdf(betahat, 0, np.sqrt(V + shat2)) - norm.logpdf(betahat, 0, np.sqrt(shat2))
    lbf[np.isinf(shat2)] = 0
    maxlbf = np.max(lbf)
    w = np.exp(lbf - maxlbf)
    w_weighted = w * prior_weights
    return np.log(np.sum(w_weighted)) + maxlbf


def neg_loglik_logscale(V, betahat, shat2, prior_weights):
    """
    Calculate the negative log-likelihood for the SuSiE model (log-scale).
    """
    return -loglik(np.exp(V), betahat, shat2, prior_weights)


def SER_posterior_e_loglik_ss(XtX_d, y0, s2, Eb, Eb2):
    """
    Calculate the expected log-likelihood for a single-effect regression (SER) model.
    """
    return -0.5 / s2 * (-2 * np.sum(Eb * y0) + np.sum(XtX_d * Eb2))


def n_in_CS_x(x, coverage=0.95):
    """
    Calculate the number of variables in the credible set for a given coverage level.
    """
    return np.searchsorted(np.cumsum(np.sort(x)[::-1]), coverage) + 1


def in_CS_x(x, coverage=0.95):
    """
    Determine which variables are in the credible set for a given coverage level.
    """
    n0 = n_in_CS_x(x, coverage)
    index = np.argsort(x)[::-1]
    result = np.zeros_like(x)
    result[index[:n0]] = 1
    return result


def in_CS(alpha, coverage=0.95):
    """
    Determine which variables are in the credible set for each component.
    """
    return np.apply_along_axis(in_CS_x, 1, alpha, coverage=coverage)


def get_purity(R, pos, n0=100):
    """
    Calculate the purity of a set of variables based on their correlation.
    """
    if len(pos) == 1:
        return [1, 1, 1]
    else:
        if len(pos) > n0:
            pos = np.random.choice(pos, n0)

        pos = pos.astype(int)
        value = np.abs(R[pos, :][:, pos][np.triu_indices(pos.shape[0], k=1)])

        return [np.min(value), np.mean(value), np.median(value)]
