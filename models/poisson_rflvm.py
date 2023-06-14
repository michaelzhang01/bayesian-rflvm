"""============================================================================
RFLVM with Poisson observations.
============================================================================"""

from   autograd import jacobian
import autograd.numpy as np
from   autograd.scipy.stats import (multivariate_normal as ag_mvn,
                                    poisson as ag_poisson)
from   models._base_rflvm import _BaseRFLVM
from   numpy import ma
from   scipy.optimize import minimize
from   scipy.special import gammaln
from   ess.ess import ESS


# -----------------------------------------------------------------------------

class PoissonRFLVM(_BaseRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize Poisson RFLVM.
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
            time_idx=kwargs.get('time_idx'),
            kernel=kwargs.get('kernel'),
            noiseless=kwargs.get('noiseless'),
            cv=kwargs.get('cv')
            
        )

# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------

    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W, add_bias=True)
        F     = phi_X @ self.beta.T
        theta = np.exp(F)
        if return_latent:
            K = phi_X @ phi_X.T
            return theta, F, K
        return theta

    def log_likelihood(self, **kwargs):
        """Differentiable log likelihood.
        """
        X = kwargs.get('X', self.X)
        W = kwargs.get('W', self.W)
        phi_X = self.phi(X, W, add_bias=True)
        F     = phi_X @ self.beta.T
        theta = np.clip(np.exp(F),0,np.finfo('d').max)
        
        if self.missing:
            LL    = (self.Y * F ) - theta
            LL    = LL - gammaln(self.Y + 1) #apprixmation for log(n!)
            LL    = np.sum(LL)
            return LL        
        else:
            LL    = ag_poisson.logpmf(self.Y, theta).sum()
            return LL

    def _log_likelihood_i(self, X, i):
        """Compute likelihood of `X_i`.
        """
        phi_X = self.phi(X[i].reshape(1,-1), self.W, add_bias=True)
        F     = phi_X @ self.beta.T
        theta = np.clip(np.exp(F),0,np.finfo('d').max)
        if self.missing:
            LL    = (self.Y[i] * F ) - theta
            LL    = LL - gammaln(self.Y[i] + 1) #approxmation for log(n!)
            LL    = LL.sum()
        else:
            LL    = ag_poisson.logpmf(self.Y[i], 
                                      theta).sum()
        return LL
    
    def get_params(self):
        """Return model parameters.
        """
        X = self.X_samples if self.t >= self.n_burn else self.X
        return dict(
            X=X,
            beta=self.beta,
            W=self.W
        )

# -----------------------------------------------------------------------------
# Sampling.
# -----------------------------------------------------------------------------

    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        self._sample_beta()

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W_prop`.
        """
        return self.log_likelihood(W=W_prop)

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        LL = self.log_likelihood(X=X)
        LP = self._log_prior_x(X)
        return LL + LP

    def _sample_beta(self):
        """Compute the maximum a posteriori estimation of `beta`.
        """
        if self.optimize:
            def _neg_log_posterior(beta_flat):
                beta = beta_flat.reshape(self.J, self.M+1)
                phi_X = self.phi(self.X, self.W, add_bias=True)
                F     = phi_X @ beta.T
                theta = np.clip(np.exp(F),0,np.finfo('d').max)
                LL    = ag_poisson.logpmf(self.Y, theta).sum()
                LP    = ag_mvn.logpdf(beta, self.beta0, self.B0).sum()
                return -(LL + LP)
    
            resp = minimize(_neg_log_posterior,
                            x0=np.copy(self.beta).flatten(),
                            jac=jacobian(_neg_log_posterior),
                            method='L-BFGS-B',
                            options=dict(
                                maxiter=self.max_iters
                            ))
            beta_map = resp.x.reshape(self.J, self.M+1)
            self.beta = beta_map

        else:
            def ess_LL(param, kwargs):
                phi_X = self.phi(self.X, self.W, add_bias=True)
                F     = phi_X @ param.T
                theta = np.clip(np.exp(F),0,np.finfo('d').max)
                if self.missing:
                    LL    = (self.Y * F ) - theta
                    LL    = LL - gammaln(self.Y + 1) #approxmation for log(n!)
                    LL    = LL.sum()
                else:
                    LL    = ag_poisson.logpmf(self.Y, 
                                              theta).sum()
                return LL                        
    
            beta_ess = ESS(rng=self.rng, init_param=self.beta, 
                        prior_mean=self.beta0, prior_var=self.B0)
            
            beta_ess._log_likelihood = ess_LL
            self.beta = beta_ess.step()

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        # Linear coefficients β in `Poisson(exp(phi(X)*β))`.
        self.beta0 = np.zeros(self.M + 1)
        self.B0 = np.eye(self.M + 1)
        self.beta  = self.rng.multivariate_normal(self.beta0, self.B0,
                                                  size=self.J)
