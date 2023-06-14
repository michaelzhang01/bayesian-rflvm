"""============================================================================
RFLVM with negative binomial observations.

In-comment citations:

    (Polson 2013)  Bayesian inference for logistic models using Polya-Gamma
                   latent variables
    (Zhou 2012)    Augment-and-conquer negative binomial processess
============================================================================"""

import autograd.numpy as np
from   autograd.scipy.special import gammaln as ag_gammaln
from   models._base_logistic_rflvm import _BaseLogisticRFLVM
from   scipy.special import expit as logistic
import pdb

# -----------------------------------------------------------------------------

class NegativeBinomialRFLVM(_BaseLogisticRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize negative binomial RFLVM.
        """
        if kwargs.get('disp_prior') is None:
            self.disp_prior = 1.
        else:
            self.disp_prior = kwargs.get('disp_prior')

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
            bias_var=kwargs.get('bias_var')
        )

# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------

    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W, add_bias=True)
        F     = phi_X @ self.beta.T
        P     = logistic(F)
        Y     = (P*self.R) / (1-P)
        if return_latent:
            K = phi_X @ phi_X.T
            return Y, F, K
        return Y

    def get_params(self):
        """Return model parameters.
        """
        X = self.X_samples if self.t >= self.n_burn else self.X
        return dict(
            X=X,
            W=self.W,
            R=self.R,
            beta=self.beta
        )

# -----------------------------------------------------------------------------
# Sampling.
# -----------------------------------------------------------------------------

    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        self._sample_omega()
        self._sample_beta()
        self._sample_r()

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W_prop`.
        """
        return self.log_likelihood(W=W_prop)

    def _a_func(self, j=None):
        """See parent class.
        """
        if j is not None:
            return self.Y[:, j]
        return self.Y

    def _b_func(self, j=None):
        """See parent class.
        """
        if j is not None:
            return self.Y[:, j] + self.R[:, j]
        return self.Y + self.R

    def _j_func(self):
        """See parent class.
        """
        return self.J

    def _log_c_func(self):
        """See parent class.
        """
        return ag_gammaln(self.Y + self.R) \
               - ag_gammaln(self.Y + 1) \
               - ag_gammaln(self.R)

    def _kappa_func(self, j):
        """See parent class.
        """
        return (self.Y[:, j] - self.R[j]) / 2.

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        LL = self.log_likelihood(X=X)
        LP = self._log_prior_x(X)
        return LL + LP

    def _sample_r(self):
        """Sample negative binomial dispersion parameter `R` based on
        (Zhou 2012). For code, see:

        https://mingyuanzhou.github.io/Softwares/LGNB_Regression_v0.zip
        """
        phi_X  = self.phi(self.X, self.W, add_bias=True)
        F      = phi_X @ self.beta.T
        P      = logistic(F)
        A      = np.array([self._crt_sum(j) for j in range(self.J)])
        maxes  = np.maximum(1 - P, -np.inf)
        B      = 1. / -np.sum(np.log(maxes), axis=0)
        self.R = np.random.gamma(A, B)
#        for j in range(self.J):
#            A = self._crt_sum(j)
#            # `maximum` is element-wise, while `max` is not.
#            maxes = np.maximum(1 - P[:, j], -np.inf)
#            B = 1. / -np.sum(np.log(maxes))
#            self.R[j] = np.random.gamma(A, B)
        # `R` cannot be zero.
        self.R[np.isclose(self.R, 0)] = 0.0000001

    def _crt_sum(self, j):
        """Sum independent Chinese restaurant table random variables.
        """
        if self.missing:
            Y_j = self.Y[:, j][~self.Y[:, j].mask]
        else:
            Y_j = self.Y[:, j]
        r    = self.R[j]
        L    = np.zeros((Y_j.size,int(Y_j.max())))
        L[:] = np.arange(L.shape[1])
        L    = L[L < Y_j[:,None]]
        L    = (r / (r+L))
        L    = np.random.binomial(1,L).sum()
        return L    

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        R = self.disp_prior * np.ones(self.J)
        # The type conversion is not necessary if `disp_prior` is float, but
        # NumPy will fail silently and convert the gamma samples in `sample_r`
        # to integers if this were of type integer.
        self.R = R.astype(np.float64)
