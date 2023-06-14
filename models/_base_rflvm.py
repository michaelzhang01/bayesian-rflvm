"""============================================================================
Base class for RFLVMs.

In-comment citations:

    (Oliva 2016)  Bayesian nonparametric kernel-learning
============================================================================"""

from   autograd import jacobian
import autograd.numpy as np
from   autograd.scipy.stats import multivariate_normal as ag_mvn
from   ess.ess import ESS
from   ppca.ppca_ma import PPCA
#from   dpmm_alg8.marg_gaussian_dpmm import MarginalizedGaussianDPMM as dpmm
from   multivariate_t import multivariate_t
from   numpy import ma
from   pykalman import KalmanFilter
from   scipy.linalg.lapack import dpotrs
#from   scipy.linalg import solve_triangular
from   scipy.optimize import minimize
from   scipy.special import logsumexp
from   scipy.stats import (invwishart,
                           multivariate_normal as mvn)
from   sklearn.decomposition import PCA
#import pdb


class _BaseRFLVM:

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize base RFLVM.
        """
        # RNG stream.
        # -----------
        self.rng = rng

        # Model hyperparameters.
        # ----------------------
        self.Y         = data
        self.missing   = ma.isMaskedArray(self.Y)
        self.N, self.J = data.shape
        if n_rffs % 2 != 0:
            # This is because we use the following definition of phi(X):
            #
            #     phi(X) = [cos(XW.T), sin(XW.T)] * normalizer
            #
            # Thus, W must be `M/2 x D` in order for phi(X) to be `N x M`.
            raise ValueError(f'`n_rffs` ({n_rffs}) must be even.')

        if self.missing:
            self.optimize=False            
        else:
            if kwargs.get('optimize') is None:
                self.optimize = False
            else:
                self.optimize = bool(kwargs.get('optimize'))
        
        # TODO: set default values for params
        self.M         = n_rffs
        self.M_div_2   = int(n_rffs / 2)
        self.n_burn    = n_burn
        self.t         = 0
        self.n_iters   = n_iters
        self.n_samples = self.n_iters - self.n_burn
        self.mh_accept = 0  # Number of MH accepts of `W`.
        # Default used by GPy is 1000; however, we run this optimization in
        # every Gibbs sampling step.
        self.max_iters = 5
        
        if kwargs.get('marginalize') is None:
            self.marginalize = True
        else:
            self.marginalize = bool(kwargs['marginalize'])

        if kwargs.get('x_init') is None:
            self.x_init = 'pca'
        else:
            self.x_init = kwargs.get('x_init')
        
        assert(self.x_init == 'pca' or self.x_init == 'kalman' \
                   or self.x_init == 'random')


        # Latent variable `X`.
        # --------------------
        self.D     = latent_dim
        self.mu_x  = np.zeros(self.D)
        self.cov_x = np.eye(self.D)

        # DP-GMM parameters.
        # ------------------
        self.K = n_clusters
        if dp_prior_obs > 0:
            self.prior_obs = dp_prior_obs
        else:
            self.prior_obs = self.D

        if dp_df > 0:
            self.nu0 = dp_df
        else:
            self.nu0 = self.D + 1

        self.alpha_a0 = 3.
        self.alpha_b0 = 1/3.
        self.alpha    = 1

        self.Psi0 = np.eye(self.D)
        self.iw0  = invwishart(df=self.prior_obs, scale=self.Psi0)
        self.mu0  = np.zeros(self.D)

        # Initialize parameters that depend upon above.
        # ---------------------------------------------
        self._init_common_params()
        self._init_specific_params()



# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------

    def fit(self):
        """Fit model to infer latent variable `X` given `Y`.
        """
        for t in range(self.n_iters):
            self.step()

    def predict(self, X, return_params=False):
        """Predict data `Y` given latent variable `X`.
        """
        raise NotImplementedError()

    def phi(self, X, W, add_bias=False):
        """Compute random Fourier features.
        """
        XW    = X @ W.T
        norm  = 1 / np.sqrt(self.M_div_2)
        phi_X = norm * np.hstack([np.cos(XW), np.sin(XW)])
        if add_bias:
            ones  = np.ones(XW.shape[0])[:, np.newaxis]
            phi_X = np.hstack((phi_X, ones))
        return phi_X


    def step(self):
        """Perform a single sampling step.
        """
        self._sample_z()
        self._sample_mu_sigma()
        self._sample_w()
#        self._ess_sample_w()
        self._sample_alpha()
        self._sample_likelihood_params()
        self._sample_x()

        if self.t >= self.n_burn:
            self.X_samples[self.t - self.n_burn] = self.X

        self.t += 1

    def get_params(self):
        """Return model parameters.
        """
        raise NotImplementedError()

    def calc_dpgmm_ll(self):
        """Calculate log likelihood of `W` given the cluster assignments.
        """
        LL = np.zeros(self.M_div_2)
        for m in range(self.M_div_2):
            log_prob = np.log(self.Z_count+self.alpha)
            log_prob += [self._posterior_mvn_t(self.W[self.Z == k], self.W[m])
                         for k in range(log_prob.size)]
            LL[m] = logsumexp(log_prob)
        return LL.sum()

# -----------------------------------------------------------------------------
# Likelihood sampling functions.
# -----------------------------------------------------------------------------

    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        raise NotImplementedError()

# -----------------------------------------------------------------------------
# Sampling `Z`, `mu`, `Sigma`.
# -----------------------------------------------------------------------------

    def _sample_z(self):
        """Draws posterior updates for every latent indicator `z_m`.
        """
        for m in self.rng.permutation(self.M_div_2):
            self.Z_count[self.Z[m]] -= 1
            log_prob  = np.copy(self.Z_count)
            # If Z_count[k] = 0 we will not enter that cluster.
            log_prob  = np.log(log_prob)
            log_prob += [self._posterior_mvn_t(self.W[self._k_idx(m,k)],
                                                  self.W[m]) \
                        for k in range(self.K)]

            new_clust = np.log(self.alpha) + \
                        self._posterior_mvn_t(np.zeros((0, self.D)), self.W[m])
            log_prob  = np.append(log_prob, new_clust)
            log_prob -= logsumexp(log_prob)
            log_prob  = np.exp(log_prob)
            new_k     = self.rng.multinomial(1, log_prob).argmax()
            self.Z[m] = new_k

            if self.Z[m] < self.K:  # If we join previous cluster:
                self.Z_count[self.Z[m]] += 1

            elif self.Z[m] == self.K:  # If we create a new cluster:
                self.K += 1
                self.Z_count = np.append(self.Z_count, 1)

                # Instantiate new features:
                self.Sigma = np.concatenate((
                    self.Sigma,
                    np.eye(self.D).reshape(-1, self.D, self.D)
                ), axis=0)
                self.mu = np.vstack((self.mu, np.zeros(self.D)))
                self._sample_mu_k_Sigma_k(self.Z[m])
            else:
                raise RuntimeError('Impossible cluster configuration.')
            if np.any(self.Z_count == 0):
                self._regularize_label()

    def _posterior_mvn_t(self, W_k, W_star_m):
        """Calculates the multivariate-t likelihood for joining new cluster.
        """
        if W_k.shape[0] > 0:
            W_bar = W_k.mean(axis=0)
            diff = W_k - W_bar
            SSE = np.dot(diff.T, diff)
            N_k = W_k.shape[0]
            prior_diff = W_bar - self.mu0
            SSE_prior = np.outer(prior_diff.T, prior_diff)
        else:
            W_bar = 0.
            SSE = 0.
            N_k = 0.
            SSE_prior = 0.

        mu_posterior = (self.prior_obs * self.mu0) + (N_k * W_bar)
        mu_posterior /= (N_k + self.prior_obs)
        nu_posterior = self.nu0 + N_k
        lambda_posterior = self.prior_obs + N_k
        psi_posterior = self.Psi0 + SSE
        psi_posterior += ((self.prior_obs * N_k) / (
                    self.prior_obs + N_k)) * SSE_prior
        psi_posterior *= (lambda_posterior + 1.) / (
                    lambda_posterior * (nu_posterior - self.D + 1.))
        df_posterior = (nu_posterior - self.D + 1.)
        return multivariate_t.logpdf(W_star_m, mu_posterior, psi_posterior,
                                     df_posterior)

    def _sample_mu_sigma(self):
        """Draw posterior updates for `mu`s and `Sigma`s. Section 3.1 from
        (Oliva 2016).
        """
        # Draw posterior samples for each `mu_k` and `Sigma_k`.
        for k in range(self.K):
            self._sample_mu_k_Sigma_k(k)

    def _sample_mu_k_Sigma_k(self, k):
        """Draw posterior updates for `mu_k` and `Sigma_k`.
        """
        W_k = self.W[self.Z == k]
        N_k = W_k.shape[0]

        if N_k > 0:
            W_k_bar     = W_k.mean(axis=0)
            diff        = W_k - W_k_bar
            mu_post     = (self.prior_obs * self.mu0) + (N_k * W_k_bar)
            mu_post    /= (N_k + self.prior_obs)
            SSE         = np.dot(diff.T, diff)
            prior_diff  = W_k_bar - self.mu0
            SSE_prior   = np.outer(prior_diff.T, prior_diff)
            nu_post     = self.nu0 + N_k
            lambda_post = self.prior_obs + N_k
            Psi_post    = self.Psi0 + SSE
            Psi_post   += ((self.prior_obs * N_k) / lambda_post) * SSE_prior

            self.Sigma[k] = invwishart.rvs(nu_post, Psi_post)
            cov           = self.Sigma[k] / lambda_post
            self.mu[k]    = self.rng.multivariate_normal(mu_post, cov)
        else:
            self.Sigma[k] = invwishart.rvs(self.nu0, self.Psi0)
            cov           = self.Sigma[k] / self.prior_obs
            self.mu[k]    = self.rng.multivariate_normal(self.mu0, cov)

# -----------------------------------------------------------------------------
# Sampling `W`.
# -----------------------------------------------------------------------------

    def opt_w(self):
        # below does constrained optimization of W
#        if self.optimize:
        def obj_fun(W_m,m):
            k = self.Z[m]
            LL = ag_mvn.logpdf(W_m, self.mu[k], self.Sigma[k])            
            low_filter = np.arange(0,m)
            hi_filter = np.arange(m+1,self.M_div_2)
            proposal_W = np.vstack((self.W[low_filter],
                                    W_m,
                                    self.W[hi_filter]))
            LL = LL + self._evaluate_proposal(proposal_W)            
            return -LL

        W_grad = jacobian(obj_fun)
#            LC = LinearConstraint(A=self.X, lb=-np.pi, ub=np.pi)
        for m in self.rng.permutation(self.M_div_2):
#                opt = minimize(obj_fun, self.W[m], jac=W_grad,
#                               constraints=LC, args=(m,))
            opt = minimize(obj_fun, self.W[m], jac=W_grad,
                           args=(m,))
            self.W[m] = opt.x

    def _ess_sample_w(self):
        def ess_LL(param, kwargs):
            k = kwargs['k']
            W_prop = np.copy(self.W)
            k_idx = self.Z == k
            W_prop[k_idx] = param
            return self._evaluate_proposal(W_prop)

        for k in range(self.K):            
            k_idx = self.Z == k
            w_ess = ESS(rng=self.rng, 
                        init_param=self.W[k_idx].reshape(-1,self.D), 
                        prior_mean=self.mu[k],  
                        prior_var=self.Sigma[k])
            
            w_ess._log_likelihood = ess_LL
            self.W[k_idx] = w_ess.step(k=k)

    def _sample_w(self):
        """Section 3.3 and 3.4 from (Oliva 2016). Sample `W` using a
        Metropolis-Hastings-based sampler.
        """
        # TODO: ESS for W sampling?
        W_curr = np.copy(self.W)
        W_prop = np.copy(W_curr)
        for m in self.rng.permutation(self.M_div_2):
            W_m_prop  = self._propose_w_m(m)
            W_m_curr  = W_curr[m]
            W_prop[m] = W_m_prop
            y_prop    = self._evaluate_proposal(W_prop)
            y_curr    = self._evaluate_proposal(W_curr)
            # log(U(0, 1)) < log(alpha)
            if np.log(self.rng.uniform(0, 1)) < y_prop - y_curr:
                self.mh_accept += 1
                W_curr[m] = W_m_prop
            else:
                # Ensure `W_prop` is in sync with `W_curr`.
                W_prop[m] = W_m_curr
        self.W = np.copy(W_curr)

    def _propose_w_m(self, m):
        """Propose sample `W_prop | W_curr ~ N(mu_k, Sigma_k)` for
        Metropolis-Hastings.
        """
        k = self.Z[m]
        return self.rng.multivariate_normal(self.mu[k], self.Sigma[k])

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W_prop`.
        """
        raise NotImplementedError()

# -----------------------------------------------------------------------------
# Sampling `X`.
# -----------------------------------------------------------------------------

    def _sample_x(self):
        """Sample `X` using user-specified method.
        """ 
        if self.optimize:
            self._sample_x_map()
        else:
            for i in range(self.N):
                self.X[i] = self._sample_x_i(i)

    def _sample_x_map(self):
        """Compute the maximum a posteriori estimation of `X`.
        """
        def _neg_log_posterior_x(X_flat):
            X = X_flat.reshape(self.N, self.D)
            return -1 * self._log_posterior_x(X)

        resp = minimize(_neg_log_posterior_x,
                        x0=np.copy(self.X).flatten(),
                        jac=jacobian(_neg_log_posterior_x),
                        method='L-BFGS-B',
                        options=dict(
                            maxiter=self.max_iters
                        ))
        X_map = resp.x
        self.X = X_map.reshape(self.N, self.D)

    def log_likelihood(self, X):
        """Compute likelihood of `X`.
        """
        raise NotImplementedError()

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        raise NotImplementedError()

    def _log_likelihood_i(self, X, i):
        """Compute likelihood of `X_i`.
        """
        raise NotImplementedError()

    def _log_prior_x(self, X):
        """Return the log prior for `X`.
        """
        return ag_mvn.logpdf(X, self.mu_x, self.cov_x).sum()
        
    def _sample_x_block_ess(self, block):
        N_block = len(block)
        nu = self.rng.multivariate_normal(self.mu_x, 
                                          self.cov_x, 
                                          size=N_block)
        u = self.rng.uniform(0,1)
        
        X = np.copy(self.X)
        current_LL = self.log_likelihood(X=X, W=self.W) + np.log(u)

        theta = self.rng.uniform(0,2.*np.pi)
        theta_min = theta - (2. * np.pi)
        theta_max = float(theta)
        X[block]  = ((self.X[block] - self.mu_x) * np.cos(theta) \
                     + nu * np.sin(theta)) + self.mu_x
        new_LL = self.log_likelihood(X=X)
        
        while new_LL < current_LL:
            if theta < 0:
                theta_min = float(theta)
            else:
                theta_max = float(theta)
            theta = self.rng.uniform(theta_min,theta_max)
            X[block] = ((self.X[block] - self.mu_x) * np.cos(theta) \
                         + nu * np.sin(theta)) + self.mu_x
            new_LL = self.log_likelihood(X=X)

        return X[block]

    def _sample_x_i(self, i):
        nu = self.rng.multivariate_normal(self.mu_x, 
                                          self.cov_x)
        u = self.rng.uniform(0,1)
        
        X = np.copy(self.X)
        current_LL = self._log_likelihood_i(X, i) + np.log(u)

        theta = self.rng.uniform(0,2.*np.pi)
        theta_min = theta - (2. * np.pi)
        theta_max = float(theta)
        X[i]      = ((self.X[i] - self.mu_x) * np.cos(theta) \
                     + nu * np.sin(theta)) + self.mu_x
        new_LL = self._log_likelihood_i(X,i)
        
        while new_LL < current_LL:
            if theta < 0:
                theta_min = float(theta)
            else:
                theta_max = float(theta)
            theta  = self.rng.uniform(theta_min,theta_max)
            X[i]   = ((self.X[i] - self.mu_x) * np.cos(theta) \
                     + nu * np.sin(theta)) + self.mu_x
            new_LL = self._log_likelihood_i(X,i)
        return X[i]


# -----------------------------------------------------------------------------
# Sampling `alpha`.
# -----------------------------------------------------------------------------

    def _sample_alpha(self):
        """See Section 6 in (Escobar 1995).
        """
        eta = self.rng.beta(self.alpha + 1, self.M)
        ak1 = self.alpha_a0 + self.K - 1
        pi  = ak1 / (ak1 + self.M * (self.alpha_b0 - np.log(eta)))
        a   = self.alpha_a0 + self.K
        b   = self.alpha_b0 - np.log(eta)
        gamma1 = self.rng.gamma(a, 1. / b)
        gamma2 = self.rng.gamma(a - 1, 1. / b)
        self.alpha = pi * gamma1 + (1 - pi) * gamma2

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_common_params(self):
        """Initialize parameters common to RFLVMs.
        """
        # Initialize and then stabilize the estimated latent variable `X`.
        if self.x_init == "pca":
            if self.missing:
                pca = PPCA(rng=self.rng, data=self.Y, latent_dim=self.D)
                self.X = pca.fit_transform()
            else:
                pca = PCA(n_components=self.D, random_state=self.rng)
                self.X = pca.fit_transform(self.Y)
        elif self.x_init == "kalman":
            kf = KalmanFilter(n_dim_obs=self.J, n_dim_state=self.D,
                              random_state=self.rng.get_state()[1][0])
            kf.em(self.Y)        
            self.X = kf.smooth(self.Y)[0]
        elif self.x_init == "random":
            self.X = self.rng.normal(size=(self.N,self.D))        
            
        self._stabilize_x()

        # Initialize K cluster mean vectors and covariance matrices.
        self.mu    = np.empty((self.K, self.D))
        self.Sigma = np.empty((self.K, self.D, self.D))
        for k in range(self.K):
            self.Sigma[k] = self.iw0.rvs()
            var0          = 1./self.prior_obs * self.Sigma[k]
            self.mu[k]    = self.rng.multivariate_normal(self.mu0, var0)

        # Initialize cluster assignments and counts.
        self.Z       = self.rng.choice(self.K, size=self.M_div_2)
        self.Z_count = np.bincount(self.Z, minlength=self.K)

        # Initialize `W` to approximate RBF kernel.
        self.W = self.rng.normal(0, 1, size=(self.M_div_2, self.D))

        # Gibb samples for analysis and visualization after burn-in.
        self.X_samples = np.empty((self.n_samples, self.N, self.D))


    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        raise NotImplementedError()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

    def _k_idx(self, m, k):
        """Filters Z==k without obs m.
        """
        k_idx    = self.Z==k
        k_idx[m] = False
        return k_idx

    def _sample_gaussian(self, J, h):
        """Copied from Linderman's `PyPolyaGamma`, who copied `pybasicbayes`.
        We actually want to compute

            V = inv(J)
            m = V @ h
            s ~ Normal(m, V)

        This function handles that computation more efficiently. See:

            https://stats.stackexchange.com/questions/32169/
        """
        L = np.linalg.cholesky(J)
        x = self.rng.randn(h.shape[0])
#        A = solve_triangular(L, x, lower=True, trans='T')        
        A = np.linalg.solve(L.T, x) #changed for compatability with ma array
        B = dpotrs(L, h, lower=True)[0]
        return A + B

    def _stabilize_x(self):
        """Fix the rotation according to the SVD.
        """
        U, _, _ = np.linalg.svd(self.X, full_matrices=False)
        L       = np.linalg.cholesky(np.cov(U.T) + 1e-6 * np.eye(self.D)).T
        self.X  = np.linalg.solve(L, U.T).T
        self.X /= np.std(self.X, axis=0)

    def _regularize_label(self):
        """Deletes empty clusters and re-labels cluster indicators while
        maintaining original ordering.
        """
        Z_plus       = self.Z_count.nonzero()[0]
        self.K       = Z_plus.size
        Z_dict       = {k: idx for idx, k in enumerate(Z_plus)}
        self.Z       = np.array([Z_dict[z_i] for z_i in self.Z])
        self.Z_count = self.Z_count[Z_plus]
        self.mu    = self.mu[Z_plus]
        self.Sigma = self.Sigma[Z_plus]
