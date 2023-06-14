"""============================================================================
Dynamic RFLVM with Gaussian observations.
============================================================================"""


import numpy as np
from   models._base_dynamic_rflvm import _BaseDynamicRFLVM
from   models.gaussian_rflvm import GaussianRFLVM
from   numpy import ma
from   scipy.stats import norm as ag_norm

# -----------------------------------------------------------------------------

class DynamicGaussianRFLVM(GaussianRFLVM, _BaseDynamicRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize Dynamic Gaussian RFLVM.
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
            hyp_var=kwargs.get('hyp_var'),
            marginalize=kwargs.get('marginalize'),
            time_idx=kwargs.get('time_idx'),
            x_init=kwargs.get('x_init'),
            optimize=kwargs.get('optimize'),
            kernel=kwargs.get('kernel'),
            noiseless=kwargs.get('noiseless')
        )

#    def _log_likelihood_i(self, X, i):
#        if self.marginalize:
#            return self.log_marginal_likelihood(X, self.W)
#        else:
#            phi_X_i = self.phi(X[i].reshape(1,self.D), 
#                               self.W, add_bias=True)
#            F = phi_X_i @ self.beta.T            
#            if self.missing:
#                LL  = -.5*((self.Y[i]-F)/np.sqrt(self.sigma_y))**2
#                LL  = LL - np.log(np.sqrt(self.sigma_y*2.*np.pi))
#                LL  = ma.sum(LL)
#            else:
#                LL = ag_norm.logpdf(self.Y[i], F, np.sqrt(self.sigma_y)).sum()
#            return LL

