import numpy as np
import pandas as pd
from numpy import matmul
from scipy.special import softmax

class ResultFunmap:

    def __init__(self, L, p, m):
        self.prior_weights = np.ones((L, p)) / p
        self.alpha = np.ones((L, p)) / p
        self.mu = np.zeros((L, p))
        self.mu2 = np.zeros((L, p))
        self.mu_w = np.zeros((L, m))
        self.Sigma_w = np.array([np.eye(m) for _ in range(L)])
        self.xi2 = np.ones((L, p))
        self.rho = np.zeros(L)
        self.XtXr = np.ones(p) / p
        self.sigma2 = 1
        self.V = np.ones(L) * 0.2
        self.sigma_w2 = np.ones(L) * 1e-5
        self.sets = None
        self.null_index = 0
        self.pip = np.ones(p) / p
        self.init = False
        self.converged = False

    def init_susie(self, p, alpha, XtXr, mu, mu2, sigma2, V, sets):
        self.alpha = alpha
        self.XtXr = XtXr
        self.mu = mu
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.V = V
        if sets['cs'] is not None:
            self.sigma_w2[np.arange(len(sets['cs']))] = 0.1
            self.xi2[np.arange(len(sets['cs']))] = (((p/2)-1)/2) ** 2
        self.init = True

    def init_funmap(self, prior_weights, alpha, XtXr, mu, mu2, mu_w, xi2, rho, V, Sigma_w, sigma_w2, sigma2):
        self.prior_weights = prior_weights
        self.alpha = alpha
        self.XtXr = XtXr
        self.mu = mu
        self.mu2 = mu2
        self.mu_w = mu_w
        self.xi2 = xi2
        self.rho = rho
        self.V = V
        self.Sigma_w = Sigma_w
        self.sigma_w2 = sigma_w2
        self.sigma2 = sigma2
        self.init = False

    def fit(self, L, p, m, n, R, A, XtX, Xty, yty, XtX_d, max_iter, tol):

        elbo = np.full(max_iter+1, np.nan)
        elbo[0] = -np.inf
        niter = 0

        for i in range(1, max_iter + 1):

            self.update_each_component(L, p, m, A, XtX, Xty, XtX_d)
            self.update_residual_variance(n, XtX, Xty, yty, XtX_d)
            elbo[i] = self.get_elbo_Funmap(L, p, m, n, A, XtX, Xty, yty, XtX_d)

            print("objective:", elbo[i])

            if (elbo[i] - elbo[i-1]) < tol * np.abs(elbo[i]):
                self.converged = True
                break
            niter = i

        if self.init:
            print("Stage2: iterations={}".format(niter))
        if not self.init:
            print("Stage3: iterations={}".format(niter))

        self.get_cs(L, R)
        self.get_pip()

    def update_each_component(self, L, p, m, A, XtX, Xty, XtX_d):

        for l in range(L):
            self.XtXr = self.XtXr - matmul(XtX, (self.alpha[l] * self.mu[l]))
            XtR = Xty - self.XtXr
            self.prior_weights[l] = softmax(matmul(A, self.mu_w[l]))
            res = fit_SER_Funmap(p, m, A, XtX_d, XtR, self.V[l], self.sigma_w2[l], self.xi2[l], self.rho[l],
                                 self.sigma2, alpha=self.alpha[l], fix_alpha=self.init)
            self.mu[l] = res['mu']
            self.alpha[l] = res['alpha']
            self.mu2[l] = res['mu2']
            self.V[l] = res['V']
            self.Sigma_w[l] = res['sigmahat']
            self.mu_w[l] = res['what']
            self.xi2[l] = res['xi2']
            self.rho[l] = res['rho']
            self.sigma_w2[l] = (self.Sigma_w[l] + np.outer(self.mu_w[l], self.mu_w[l])).trace() / m
            self.XtXr = self.XtXr + matmul(XtX, self.alpha[l] * self.mu[l])

    def update_residual_variance(self, n, XtX, Xty, yty, XtX_d):

        estimate_sigma2 = (1 / n) * self.get_ER2_ss(XtX, Xty, yty, XtX_d)
        self.sigma2 = np.maximum(1e-4, estimate_sigma2)

    def get_elbo_Funmap(self, L, p, m, n, A, XtX, Xty, yty, XtX_d):

        return self.Eloglik(n, XtX, Xty, yty, XtX_d) + self.Eloglik_b(L) + self.Eloglik_gamma(L, p, A) \
               + self.Eloglik_w(L, m) - self.Eloglik_q(L, m)

    def Eloglik(self, n, XtX, Xty, yty, XtX_d):

        return -(n / 2)*np.log(2 * np.pi * self.sigma2) - (1 / (2 * self.sigma2))*self.get_ER2_ss(XtX, Xty, yty, XtX_d)

    def Eloglik_b(self, L):

        return np.sum(np.array([0.5*self.alpha[l0]*(1+np.log((self.mu2[l0]-self.mu[l0]**2)/self.V[l0])
                                - (self.mu[l0]**2+self.V[l0])/self.V[l0]) for l0 in range(L) if self.V[l0]]))

    def Eloglik_gamma(self, L, p, A):

        if (np.sqrt(self.xi2) < 700).all():
            tmp = np.log(1 + np.exp(np.sqrt(self.xi2)))
        else:
            tmp = np.array([[np.log(1 + np.exp(np.sqrt(self.xi2[l0, j0]))) if np.sqrt(self.xi2[l0, j0]) < 700
                             else np.sqrt(self.xi2[l0, j0]) for j0 in range(p)] for l0 in range(L)])
        return np.sum(self.alpha * matmul(self.mu_w, A.T)) - np.sum(self.rho) - \
            np.sum(lmd_xi(np.sqrt(self.xi2)) *
                   np.array([np.diagonal(matmul(matmul(A, self.Sigma_w[l0]), A.T)) for l0 in range(L)])) - \
            np.sum(0.5 * ((matmul(A, self.mu_w.T) - self.rho).T - np.sqrt(self.xi2)) +
                   lmd_xi(np.sqrt(self.xi2)) * ((matmul(A, self.mu_w.T) - self.rho).T ** 2 - self.xi2) + tmp)

    def Eloglik_w(self, L, m):

        return -0.5 * m * np.sum(np.log(2 * np.pi * self.sigma_w2)) - \
            np.sum([0.5 * (np.outer(self.mu_w[l0], self.mu_w[l0]) + self.Sigma_w[l0]).trace()
                    / self.sigma_w2[l0] for l0 in range(L)])

    def Eloglik_q(self, L, m):

        return np.sum(
            [-0.5 * (np.log((2 * np.pi * np.e) ** m) + np.sum(np.log(np.linalg.eig(self.Sigma_w[l0])[0]))) for l0 in
             range(L)]) + np.nansum(self.alpha * np.log(self.alpha))

    def get_ER2_ss(self, XtX, Xty, yty, XtX_d):

        B = self.alpha * self.mu
        XB2 = np.sum(matmul(B, XtX) * B)
        betabar = np.sum(B, axis=0)
        postb2 = self.alpha * self.mu2
        return yty - 2 * np.sum(betabar * Xty) + np.sum(betabar * matmul(XtX, betabar)) - XB2 + np.sum(XtX_d * postb2)

    def get_cs(self, L, R, coverage=0.95, min_abs_corr=0.5, n_purity=100):

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


def fit_SER_Funmap(p, m, A, XtX_d, y0, V0, sigma_w2, xi2, rho, residual_variance, alpha=None, fix_alpha=False):

    sigmahat = np.linalg.inv(np.eye(m)/sigma_w2 +
                             2*np.array([lmd_xi(np.sqrt(xi2[j])) * np.outer(A[j], A[j]) for j in range(p)]).sum(axis=0))
    what = matmul(sigmahat, np.array([(alpha[j]-0.5+2*rho*lmd_xi(np.sqrt(xi2[j])))*A[j] for j in range(p)]).sum(axis=0))

    rho = (p/4-0.5+np.sum([lmd_xi(np.sqrt(xi2[j]))*matmul(A[j], what) for j in range(p)]))/lmd_xi(np.sqrt(xi2)).sum()

    shat2 = residual_variance / XtX_d
    post_var = 1 / (1 / V0 + 1 / shat2) if V0 else 0
    post_mean = (1 / residual_variance) * post_var * y0
    post_mean2 = post_var + post_mean ** 2

    if not fix_alpha:
        tmp = np.log(post_var)/2 + post_mean**2/post_var/2 if V0 else 0
        alpha = softmax(matmul(A, what) + tmp)

    xi2 = np.array([matmul(matmul(A[j], sigmahat + np.outer(what, what)), A[j]) +
                    rho**2 - 2*rho*matmul(A[j], what) for j in range(p)])

    V0 = np.sum(post_mean2 * alpha)

    return dict(alpha=alpha, mu=post_mean, mu2=post_mean2, V=V0, sigmahat=sigmahat, what=what, xi2=xi2, rho=rho)


def lmd_xi(xi):

    return 0.5 * (1 / (1 + np.exp(-xi)) - 0.5) / xi


def n_in_CS_x(x, coverage=0.9):

    return np.searchsorted(np.cumsum(np.sort(x)[::-1]), coverage) + 1


def in_CS_x(x, coverage=0.9):

    n0 = n_in_CS_x(x, coverage)
    index = np.argsort(x)[::-1]
    result = np.zeros_like(x)
    result[index[:n0]] = 1
    return result


def in_CS(alpha, coverage=0.9):

    return np.apply_along_axis(in_CS_x, 1, alpha, coverage=coverage)


def get_purity(R, pos, n0=100):

    if len(pos) == 1:
        return [1, 1, 1]
    else:
        if len(pos) > n0:
            pos = np.random.choice(pos, n0)

        pos = pos.astype(int)
        value = np.abs(R[pos, :][:, pos][np.triu_indices(pos.shape[0], k=1)])

        return [np.min(value), np.mean(value), np.median(value)]
