"""============================================================================
Base class for dynamic logistic RFLVMs. Logistic models have likelihoods that 
can be written as:

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

import numpy as np
from   models._base_logistic_rflvm import _BaseLogisticRFLVM
from   models._base_dynamic_rflvm import _BaseDynamicRFLVM

# -----------------------------------------------------------------------------

class _BaseDynamicLogisticRFLVM(_BaseDynamicRFLVM, _BaseLogisticRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize base class for dynamic logistic RFLVMs.
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
            disp_prior=kwargs.get('disp_prior'),
            bias_var=kwargs.get('bias_var'),
            hyp_var=kwargs.get('hyp_var'),
            time_idx=kwargs.get('time_idx'),
            x_init=kwargs.get('x_init'),
            kernel=kwargs.get('kernel'),
            test_prop=kwargs.get('test_prop')
        )

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
    
    def _init_common_params(self):
        """Initialize parameters common to logistic RFLVMs.
        """        
        super()._init_common_params()        
