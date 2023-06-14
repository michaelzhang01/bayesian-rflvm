"""============================================================================
RFLVM with multinomial observations.

In-comment citations:

    (Baker 1994)   The multinomial-Poisson transformation
    (Chen 2013)    Scalable inference for logistic-normal topic models
    (Polson 2013)  Bayesian inference for logistic models using Polya-Gamma
                   latent variables
============================================================================"""


import autograd.numpy as np
from   autograd import jacobian
from   autograd.scipy.special import logsumexp as ag_lse
from   autograd.scipy.special import gammaln
from   autograd.scipy.stats import poisson as ag_poisson
from   autograd.scipy.stats import norm as ag_norm
from   ess.ess import ESS
from   numpy import ma
from   models._base_logistic_rflvm import _BaseLogisticRFLVM
from   scipy.optimize import minimize

# -----------------------------------------------------------------------------

class MultinomialRFLVM(_BaseLogisticRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):# disp_prior, bias_var,
                 #A_var):
        """Initialize RFLVM.
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
            bias_var=kwargs.get('bias_var'),
            x_init=kwargs.get('x_init'),
            optimize=kwargs.get('optimize')
        )

        # `Q` is an `(N, J)` matrix in which each component is
        # `Q_n - \sum x_{nj}`.
        self.Q_n = self.Y.sum(axis=1)
        self.Q = self.Q_n[:, None] * np.ones(self.Y.shape)

        # Fix last beta to zero.
        self.beta = np.vstack((self.beta,np.zeros(self.M+1)))
    
        # Prior variance of normalizing constant.
        if kwargs.get('A_var') is None:
            self.A_var = 5.
        else:
            self.A_var = kwargs.get('A_var')
        

# -----------------------------------------------------------------------------
# Public API.
# -------------------------------s---------------------------------------------

    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W, add_bias=True)
        psi   = phi_X @ self.beta.T
        pi    = self.psi_to_pi(psi)
        Y     = self.Q_n[:, None] * pi
        if return_latent:
            F = psi
            K = phi_X @ phi_X.T
            return Y, F, K
        return Y

    def psi_to_pi(self, psi):
        """Log-normalize and exponentiate psi vector        
        """
        return np.exp(psi - ag_lse(psi, axis=1)[:, None])

    def log_likelihood(self, **kwargs):
        """Differentiable log likelihood for the multinomial distribution.

        We have to overwrite `_BaseLogisticRFLVM`'s log likelihood function
        because this model's log likelihood is multinomial, which is not
        in the "logistic family".
        """
        # Optimize the log likelihood w.r.t. `X`. Use MH to evaluate the log
        # likelihood w.r.t. a proposed `W`.
        X = kwargs.get('X', self.X)
        W = kwargs.get('W', self.W)

        phi_X = self.phi(X, W, add_bias=True)
        psi   = phi_X @ self.beta.T  + self.a0[:, None]
        theta = np.clip(np.exp(psi),0,np.finfo('d').max)

        if self.missing:
            LL    = (self.Y * psi ) - theta
            LL    = LL - gammaln(self.Y + 1) #apprixmation for log(n!)
            LL    = np.sum(LL)
        else:
            LL    = ag_poisson.logpmf(self.Y, theta).sum()

        return LL

    def get_params(self):
        """Return model parameters.
        """
        X = self.X_samples if self.t >= self.n_burn else self.X
        return dict(
            X=X,
            W=self.W,
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
        self._sample_a()

    def _sample_a(self):
        """Optimize the nuisance parameter in the multinomial-Poisson
        transformation. See (Baker 1994) for more details.
        """
        
        if self.optimize:
        
            def _neg_log_posterior(a0):
                phi_X = self.phi(self.X, self.W, add_bias=True)
                psi   = phi_X @ self.beta.T 
                # Assume prior mean on a0 is the current normalizing constant.
                prior_mean = -ag_lse(psi, axis=1)
                psi += a0[:, None]
                LL   = ag_poisson.logpmf(self.Y, np.exp(psi)).sum()
                var  = np.sqrt(self.A_var)*np.ones(self.N)
                LL  += ag_norm.logpdf(a0, prior_mean, var).sum()
                return(-1.*LL)
    
            resp = minimize(_neg_log_posterior,
                            x0=np.copy(self.a0),
                            jac=jacobian(_neg_log_posterior),
                            method='L-BFGS-B',
                            options=dict(
                                maxiter=self.max_iters
                            ))
            self.a0 = resp.x.reshape(self.N)

        else:
            def ess_LL(param, kwargs):
                phi_X = self.phi(self.X, self.W, add_bias=True)
                psi   = phi_X @ self.beta.T 
                psi  += param.reshape(-1,1)
                theta = np.clip(np.exp(psi),0,np.finfo('d').max)
                
                if self.missing:
                    LL    = (self.Y * psi ) - theta
                    LL    = LL - gammaln(self.Y + 1) #apprixmation for log(n!)
                    LL    = np.sum(LL)
                else:
                    LL    = ag_poisson.logpmf(self.Y, theta).sum()
                return LL                        

            phi_X = self.phi(self.X, self.W, add_bias=True)
            psi   = phi_X @ self.beta.T 
            prior_mean = -ag_lse(psi, axis=1)
            prior_var  = np.sqrt(self.A_var)*np.ones(self.N)
#            import pdb
#            pdb.set_trace()
            a0_ess = ESS(rng=self.rng, init_param=self.a0.reshape(1,-1), 
                        prior_mean=prior_mean, prior_var=prior_var)
            
            a0_ess._log_likelihood = ess_LL
            self.a0 = a0_ess.step().flatten()

    def _sample_beta(self):
        """Sample β|ω ~ N(m, V). See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)
        for j in range(self._j_func()):
            notj = np.arange(self.J) != j
            ksi = ag_lse(phi_X @ self.beta[notj].T, axis=1)
            # This really computes: phi_X.T @ np.diag(omega[:, j]) @ phi_X
            if self.missing:
                J = ma.dot((phi_X * self.omega[:, j][:, None]).T, phi_X) + \
                    self.inv_B
                h = ma.dot(phi_X.T, (self._kappa_func(j) + ksi*self.omega[:, j])) + \
                    self.inv_B_b 
            else:
                J = (phi_X * self.omega[:, j][:, None]).T @ phi_X + \
                    self.inv_B
                h = phi_X.T @ (self._kappa_func(j) + ksi*self.omega[:, j]) + \
                    self.inv_B_b 

#            L  = np.linalg.cholesky(J)
#            Z  = self.rng.normal(size=self.M+1).T
#            import pdb
#            pdb.set_trace()
#            LZ = solve_triangular(L, Z, lower=True, trans='T')            
#            LZ = np.linalg.solve(L.T,Z)
#            L_mu = dpotrs(L, h, lower=True)[0]
#            self.beta[j] = LZ + L_mu
            self.beta[j] = self._sample_gaussian(J=J, h=h)

    def _sample_omega(self):
        """Sample ω|β ~ PG(y+r, x*β). See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)
        # Is there a faster way to do the below line? It's inefficient.
        Ksi = np.vstack([
            ag_lse(phi_X @ self.beta[np.arange(self.J) != j].T, axis=1)
            for j in range(self.J)
        ]).T
        psi = (phi_X @ self.beta.T) - Ksi
        bb = self._b_func()
        self.pg.pgdrawv(bb.ravel(),
                        psi.ravel(),
                        self.omega.ravel())
        self.omega = self.omega.reshape(self.Y.shape)

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W_prop`.
        """
        return self.log_likelihood(W=W_prop)

    def _j_func(self):
        """See parent class.
        """
        return self.J-1

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
            return self.Q[:,j]
        return self.Q

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        LL = self.log_likelihood(X=X)
        LP = self._log_prior_x(X)
        return LL + LP

    def _log_likelihood_i(self, X, i):
        """Compute likelihood of X_i.
        """
        phi_X_i = self.phi(X[i].reshape(1,self.D), 
                           self.W, add_bias=True)
        psi   = phi_X_i @ self.beta.T  + self.a0[i]
        theta = np.clip(np.exp(psi),0,np.finfo('d').max)

        if self.missing:
            LL    = (self.Y[i] * psi ) - theta
            LL    = LL - gammaln(self.Y[i] + 1) #apprixmation for log(n!)
            LL    = np.sum(LL)
        else:
            LL    = ag_poisson.logpmf(self.Y[i], theta).sum()

        return LL

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        # Initialize nuisance parameters for multinomial-Poisson transform
        self.a0 = np.zeros(self.N)
