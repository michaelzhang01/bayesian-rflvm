"""============================================================================
RFLVM with Poisson observations.
============================================================================"""

import numpy as np
from   scipy.stats import poisson as ag_poisson
from   models._base_dynamic_rflvm import _BaseDynamicRFLVM
from   models.poisson_rflvm import PoissonRFLVM
import pdb
# -----------------------------------------------------------------------------

class DynamicPoissonRFLVM(PoissonRFLVM, _BaseDynamicRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize Dynamic Poisson RFLVM.
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
    
#    def _log_likelihood_i(self,X,i):
#        """Differentiable log likelihood.
#        """
#        phi_X_i = self.phi(X[i].reshape(1,self.D), 
#                           self.W, add_bias=True)
#        F  = phi_X_i @ self.beta.T            
#        LL = ag_poisson.logpmf(self.Y[i], np.exp(F)).sum()
#        
#        return LL

