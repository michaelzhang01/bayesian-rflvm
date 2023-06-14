"""============================================================================
Base class for logistic RFLVMs. Logistic models have likelihoods that can be
written as:

    c(y) * [exp(beta * x)^{a(y)} / (1 + exp(beta * x)^{b(y)}]

(Polson 2013) introduce Polya-gamma random variables, which introduces another
function of the data, `kappa(y)`. Sub-classing this model primarily requires to
implementing functions to compute `a(y)`, `b(y)`, `log c(y)`, and `kappa(y)`.

The logic in this class borrows heavily from the Linderman's `PyPolyaGamma`:

    https://github.com/slinderman/pypolyagamma

In-comment citations:

    (Polson 2013)  Bayesian inference for logistic models using Polya-Gamma
                   latent variables
============================================================================"""

import autograd.numpy as np
from   autograd import jacobian
from   autograd.scipy.stats import multivariate_normal as ag_mvn
from   models._base_rflvm import _BaseRFLVM
from   numpy import ma
from   scipy.optimize import minimize
try:
    from   pypolyagamma import PyPolyaGamma
except ImportError:
    from   polyagamma import random_polyagamma

# -----------------------------------------------------------------------------

class _BaseLogisticRFLVM(_BaseRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize base class for logistic RFLVMs.
        """
        # `_BaseRFLVM` will call `_init_specific_params`, and these need to be
        # set first.
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
            optimize=kwargs.get('optimize')

        )
            
        # Polya-gamma augmentation.
        if kwargs.get('bias_var') is None:
            self.bias_var   = 5.
        else:
            self.bias_var   = kwargs.get('bias_var')

        try:
            self.pg         = PyPolyaGamma()
        except:
            self.pg         = None
        prior_Sigma         = np.eye(self.M+1)
        prior_Sigma[-1, -1] = np.sqrt(self.bias_var)
        self.inv_B          = np.linalg.inv(prior_Sigma)
        mu_A_b              = np.zeros(self.M+1)
        self.inv_B_b        = self.inv_B @ mu_A_b
        self.omega          = np.zeros(self.Y.shape)
        if self.missing:
            self.omega = ma.array(self.omega, mask=self.Y.mask)

        # Linear coefficients `beta`.
        self.beta0 = np.zeros(self.M + 1)
        self.B0    = np.eye(self.M + 1)
        self.beta  = self.rng.multivariate_normal(self.beta0, self.B0,
                                                     size=self._j_func())

# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------

    def log_likelihood(self, **kwargs):
        """Generalized, differentiable log likelihood function.
        """
        # This function can be called for two reasons:
        #
        #   1. Optimize the log likelihood w.r.t. `X`.
        #   2. Evaluate the log likelihood w.r.t. a MH-proposed `W`.
        #
        X = kwargs.get('X', self.X)
        W = kwargs.get('W', self.W)
        beta = kwargs.get('beta', self.beta)

        phi_X = self.phi(X, W, add_bias=True)
        psi   = phi_X @ beta.T
        LL    = self._log_c_func() \
                + self._a_func() * psi \
                - self._b_func() * np.log(1 + np.exp(psi))

        return LL.sum()

# -----------------------------------------------------------------------------
# Polya-gamma augmentation.
# -----------------------------------------------------------------------------

    def _sample_beta(self):
        """Sample `β|ω ~ N(m, V)`. See (Polson 2013).
        """
        if self.optimize:
            def _neg_log_posterior(beta_flat):
                beta = beta_flat.reshape(self.J, self.M+1)
                LL   = self.log_likelihood(beta=beta)
                LP   = ag_mvn.logpdf(beta, self.beta0, self.B0).sum()
                return -(LL + LP)
                
            resp = minimize(_neg_log_posterior,
                            x0=np.copy(self.beta),
                            jac=jacobian(_neg_log_posterior),
                            method='L-BFGS-B',
                            options=dict(
                                maxiter=self.max_iters
                            ))
            beta_map = resp.x.reshape(self.J, self.M+1)
            self.beta = beta_map            

        else:
            phi_X = self.phi(self.X, self.W, add_bias=True)    
            for j in range(self.J):
                # This really computes: phi_X.T @ np.diag(omega[:, j]) @ phi_X
                if self.missing:
                    J = ma.dot((phi_X * self.omega[:, j][:, None]).T, phi_X) +\
                        self.B0
                    h = ma.dot(phi_X.T, self._kappa_func(j)) + self.inv_B_b
                else:                    
                    J = (phi_X * self.omega[:, j][:, None]).T @ phi_X + \
                        self.inv_B
                    h = phi_X.T @ self._kappa_func(j) + self.inv_B_b
                joint_sample = self._sample_gaussian(J=J, h=h)
                self.beta[j] = joint_sample

    def _sample_omega(self):
        """Sample `ω|β ~ PG(b, x*β)`. See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)
        psi   = phi_X @ self.beta.T
        b     = self._b_func()
        omega_train = np.empty(self.omega.size)
        if self.pg is None:
            random_polyagamma(b.ravel(),psi.ravel(), out=omega_train,
                              random_state=self.rng.get_state()[1][0])
        else:
            self.pg.pgdrawv(b.ravel(),
                            psi.ravel(),
                            omega_train)
            
        if self.missing:
            self.omega = ma.array(omega_train.reshape(self.omega.shape),
                                  mask=self.Y.mask)
        else:
            self.omega = omega_train.reshape(self.omega.shape)

    def _a_func(self, j=None):
        """This function returns `a(y)`. See the comment at the top of this
        file and (Polson 2013).
        """
        raise NotImplementedError()

    def _b_func(self, j=None):
        """This function returns `b(y)`. See the comment at the top of this
        file and (Polson 2013).
        """
        raise NotImplementedError()

    def _log_c_func(self):
        """This function returns `log c(y)`. This is the normalizer in logistic
        models and is only used in the log likelihood calculation. See the
        comment at the top of this file and (Polson 2013).
        """
        raise NotImplementedError()

    def _j_func(self):
        """Return number of features to iterate over. This is required because
        multinomial models decompose the multinomial distribution into `J-1`
        binomial distributions.
        """
        raise NotImplementedError()

    def _kappa_func(self, j):
        """This function returns `kappa(y)`. See the comment at the top of this
        file and (Polson 2013).
        """
        return self._a_func(j) - (self._b_func(j) / 2.0)

    def _log_likelihood_i(self, X, i):
        """Compute likelihood of X_i.
        """
        phi_X_i = self.phi(X[i].reshape(1,self.D), 
                           self.W, add_bias=True)
        psi   = phi_X_i @ self.beta.T
        LL    = self._log_c_func()[i] \
                + self._a_func()[i] * psi \
                - self._b_func()[i] * np.log(1 + np.exp(psi))
        return LL.sum()