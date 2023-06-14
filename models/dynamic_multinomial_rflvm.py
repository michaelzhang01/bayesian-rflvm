"""============================================================================
Dynamic RFLVM with negative binomial observations.

In-comment citations:

    (Polson 2013)  Bayesian inference for logistic models using Polya-Gamma
                   latent variables
    (Zhou 2012)    Augment-and-conquer negative binomial processess
============================================================================"""

import numpy as np
#import numpy as np
from   models._base_dynamic_logistic_rflvm import _BaseDynamicLogisticRFLVM
from   models.multinomial_rflvm import MultinomialRFLVM
from   scipy.stats import poisson as ag_poisson
from   scipy.special import gammaln
#from   scipy.stats import nbinom
# -----------------------------------------------------------------------------

class DynamicMultinomialRFLVM(MultinomialRFLVM,
                              _BaseDynamicLogisticRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize negative binomial RFLVM.
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
            disp_prior=kwargs.get('disp_prior'),
            bias_var=kwargs.get('bias_var'),
            hyp_var=kwargs.get('hyp_var'),
            time_idx=kwargs.get('time_idx'),
            x_init=kwargs.get('x_init'),
            optimize=kwargs.get('optimize'),
            kernel=kwargs.get('kernel'),
            noiseless=kwargs.get('noiseless'),
            A_var=kwargs.get('A_var'),
            test_prop=kwargs.get('test_prop')
        )

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



    