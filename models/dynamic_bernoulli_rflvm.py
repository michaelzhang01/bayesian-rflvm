"""============================================================================
Dynamic RFLVM with Bernoulli observations.
============================================================================"""

from   models.bernoulli import BernoulliRFLVM
from   models._base_dynamic_logistic_rflvm import _BaseDynamicLogisticRFLVM

# -----------------------------------------------------------------------------

class DynamicBernoulliRFLVM(_BaseDynamicLogisticRFLVM, BernoulliRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize Dynamic Bernoulli RFLVM.
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
            num_p=kwargs.get('num_p'),
            bias_var=kwargs.get('bias_var'),
            hyp_var=kwargs.get('hyp_var'),
            time_idx=kwargs.get('time_idx'),
            marginalize=kwargs.get('marginalize'),
            x_init=kwargs.get('x_init'),
            marginalize_IS=kwargs.get('marginalize_IS'),
            optimize=kwargs.get('optimize'),
            kernel=kwargs.get('kernel'),
            sparse=kwargs.get('sparse'),
            noiseless=kwargs.get('noiseless')          
        )            