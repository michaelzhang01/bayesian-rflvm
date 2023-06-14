"""============================================================================
RFLVM with Gaussian observations.
============================================================================"""

#from   autograd import jacobian
import autograd.numpy as np
from   autograd.scipy.special import gammaln
from   autograd.scipy.stats import (norm as ag_norm,
                                    multivariate_normal as ag_mvn)
from   models._base_rflvm import _BaseRFLVM
from   numpy import ma
from   scipy.linalg import solve_triangular
from   scipy.linalg.lapack import dpotrs
#from   scipy.optimize import minimize

# -----------------------------------------------------------------------------

class GaussianRFLVM(_BaseRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize Gaussian RFLVM.
        """

        super().__init__(
            rng=rng,
            data=data,
            n_burn=n_burn,
            n_iters=n_iters,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            n_rffs=n_rffs,
            dp_prior_obs=dp_prior_obs,
            dp_df=dp_df,
            x_init=kwargs.get('x_init'),
            optimize=kwargs.get('optimize'),
            marginalize=kwargs.get('marginalize')
        )
        
# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------

    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W)

        if self.marginalize:
            # This is the mean of the posterior predictive in Bayesian linear
            # regression, a multivariate t-distribution.
            Lambda_n = phi_X.T @ phi_X + np.eye(self.M)
            if self.missing:
                mu_n     = ma.dot(np.linalg.solve(Lambda_n, phi_X.T), self.Y)
                Y = F    = ma.dot(phi_X, mu_n)
            else:
                mu_n     = np.linalg.solve(Lambda_n, phi_X.T) @ self.Y
                Y = F    = phi_X @ mu_n
        else:
            phi_X = self.phi(X, self.W, add_bias=True)
            Y = F = phi_X @ self.beta.T

        if return_latent:
            K = phi_X @ phi_X.T
            return Y, F, K
        return Y

    def _log_likelihood_i(self, X, i):
        if self.marginalize:
            return self.log_marginal_likelihood(X, self.W)
        else:
            phi_X_i = self.phi(X[i].reshape(1,self.D), 
                               self.W, add_bias=True)
            F = phi_X_i @ self.beta.T            
            if self.missing:
                LL  = -.5*((self.Y[i]-F)/np.sqrt(self.sigma_y))**2
                LL  = LL - np.log(np.sqrt(self.sigma_y*2.*np.pi))
                LL  = ma.sum(LL)
            else:
                LL = ag_norm.logpdf(self.Y[i], F, np.sqrt(self.sigma_y)).sum()
            return LL


    def log_likelihood(self, **kwargs):
        """Differentiable log likelihood.
        """
        X = kwargs.get('X', self.X)
        W = kwargs.get('W', self.W)

        if self.marginalize:
            return self.log_marginal_likelihood(X, W)
        else:
            beta = kwargs.get('beta', self.beta)
            phi_X = self.phi(X, W, add_bias=True)
            F = phi_X @ beta.T
            if self.missing:
                LL  = -.5*((self.Y-F)/np.sqrt(self.sigma_y))**2
                LL  = LL -np.log(np.sqrt(self.sigma_y*2.*np.pi))
                LL  = ma.sum(LL)
            else:
                LL  = ag_norm.logpdf(self.Y.flatten(), 
                                    F.flatten(), 
                                    np.tile(np.sqrt(self.sigma_y),
                                            self.N)).sum()
            return LL

    def log_marginal_likelihood(self, X, W):
        """Log marginal likelihood after integrating out `beta`. We assume
        the prior mean of `beta` is zero and that `S_0 = identity(M)`.
        """
        phi_X = self.phi(X, W)
        S_n   = phi_X.T @ phi_X + np.eye(self.M)
        a_n   = self.a0 + self.N / 2

        # Compute Lambda term.
        sign, logdet = np.linalg.slogdet(S_n)
        lambda_term  = -0.5 * sign * logdet

        # Compute a_n term.
        gamma_term = gammaln(a_n) - gammaln(self.a0)
        
        if self.missing:
            mu_n   = ma.dot(np.linalg.solve(S_n, phi_X.T), self.Y)
            A      = np.diag(ma.dot(self.Y.T, self.Y))
            C      = np.diag(ma.dot(ma.dot(mu_n.T, S_n), mu_n))
            b_n    = self.b0 + 0.5 * (A - C)
            b_term = self.a0 * np.log(self.b0) - a_n * ma.log(b_n)
            return ma.sum(gamma_term + b_term + lambda_term)

        else:
            mu_n   = np.linalg.solve(S_n, phi_X.T) @ self.Y
            A      = np.diag(self.Y.T @ self.Y)
            C      = np.diag(mu_n.T @ S_n @ mu_n)
            # Compute b_n term.
            b_n    = self.b0 + 0.5 * (A - C)
            b_term = self.a0 * np.log(self.b0) - a_n * np.log(b_n)
            # Compute sum over all y_n.
            return np.sum(gamma_term + b_term + lambda_term)

    def get_params(self):
        """Return model parameters.
        """
        X = self.X_samples if self.t >= self.n_burn else self.X
        return dict(
            X=X,
            W=self.W
        )

# -----------------------------------------------------------------------------
# Sampling.
# -----------------------------------------------------------------------------

    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        if self.marginalize:
            # We integrated out `beta` a la Bayesian linear regression.
            pass
        else:
            self._sample_beta_and_sigma_Y()


    def _sample_beta_and_sigma_Y(self):
        """Gibbs sample `beta` and noise parameter `sigma_Y`.
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)
        cov_j = (self.B0 + phi_X.T @ phi_X)
        if self.missing:            
            mu_j  = np.tile((self.B0 @ self.beta0), (self.J,1)).T + \
                ma.dot(phi_X.T, self.Y)
            b_post =  self.b0 + .5 * np.diag(ma.dot(self.Y.T, self.Y) + \
                           (self.beta0 @ self.B0 @ self.beta0.T) +\
                           (ma.dot(mu_j.T, np.linalg.solve(cov_j, mu_j))))            

        else:
            mu_j  = np.tile((self.B0 @ self.beta0), (self.J,1)).T + \
                (phi_X.T @ self.Y)
            b_post =  self.b0 + .5 * np.diag((self.Y.T @ self.Y) + \
                                       (self.beta0 @ self.B0 @ self.beta0.T) +\
                                       (mu_j.T @ np.linalg.solve(cov_j, mu_j)))            


        # multi-output generalization of mvn sample code
        L  = np.linalg.cholesky(cov_j)
        Z  = self.rng.normal(size=self.beta.shape).T
        LZ = solve_triangular(L, Z, lower=True, trans='T')            
        L_mu = dpotrs(L, mu_j, lower=True)[0]
        self.beta[:] = (LZ + L_mu).T

        # sample from inverse gamma                 
        a_post =  self.a0 + .5 * self.N

        self.sigma_y = 1. / self.rng.gamma(a_post,1./b_post)

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W` using the log evidence.
        """
        if self.marginalize:
            return self.log_marginal_likelihood(self.X, W_prop)
        else:
            return self.log_likelihood(W=W_prop)

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        if self.marginalize:
            LL = self.log_marginal_likelihood(X, self.W)
        else:
            LL = self.log_likelihood(X=X)
        LP = self._log_prior_x(X)
        return LL + LP

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        # Equivalent to the inverse-gamma hyperparameters in Bayesian
        # linear regression.
        self.a0 = 1
        self.b0 = 1
        
        if not self.marginalize:
            # Linear coefficients β in `Poisson(exp(phi(X)*β))`.
            self.beta0 = np.zeros(self.M + 1)
            self.B0 = np.eye(self.M + 1)
            self.beta = self.rng.multivariate_normal(self.beta0, self.B0,
                                                     size=self.J)
            # likelihood noise term
            self.sigma_y = 1. / self.rng.gamma(self.a0, 1./self.b0, 
                                               size=self.J)