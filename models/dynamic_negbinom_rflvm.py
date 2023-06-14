"""============================================================================
Dynamic RFLVM with negative binomial observations.

In-comment citations:

    (Polson 2013)  Bayesian inference for logistic models using Polya-Gamma
                   latent variables
    (Zhou 2012)    Augment-and-conquer negative binomial processess
============================================================================"""

#import autograd.numpy as np
from   models._base_dynamic_logistic_rflvm import _BaseDynamicLogisticRFLVM
from   models.negbinom_rflvm import NegativeBinomialRFLVM
#from   scipy.special import expit as logistic
#from   scipy.stats import nbinom
# -----------------------------------------------------------------------------

class DynamicNegativeBinomialRFLVM(NegativeBinomialRFLVM,
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
            noiseless=kwargs.get('noiseless')
        )