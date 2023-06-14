"""============================================================================
Base class for Dynamic RFLVMs.
============================================================================"""

#import autograd.numpy as np
import numpy as np
#from   autograd import jacobian
#from   autograd.scipy.linalg import solve_triangular
#from   autograd.scipy.special import logsumexp
#from   autograd.scipy.stats import multivariate_normal as ag_mvn
#from   autograd.scipy.stats import norm as ag_norm
from   ess.ess import ESS
from   GPy.util.linalg import dtrtrs, jitchol, tdot, pdinv
#from   itertools import product
from   models._base_rflvm import _BaseRFLVM
#from   scipy.optimize import minimize
#from   .utils import jitchol_ag, logexp, inv_logexp
from   .utils import logexp, inv_logexp
import pdb


class _BaseDynamicRFLVM(_BaseRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, **kwargs):
        """Initialize base dynamic RFLVM.
        """            
        if kwargs.get('hyp_var') is None:
            self.hyp_var = 3.
        else:
            self.hyp_var = float(kwargs['hyp_var'])
         
        if kwargs.get('noiseless') is None:
            self.noiseless = False
        else:
            self.noiseless = bool(kwargs.get('noiseless'))

        if kwargs.get('cv') is None:
            self.cv = True
        else:
            self.cv = bool(kwargs.get('cv'))

        
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
            marginalize=kwargs.get('marginalize'),
            x_init=kwargs.get('x_init'),
        )

        if kwargs.get('time_idx') is None:
            self.time_idx = np.linspace(-1,1,self.N).reshape(-1,1)
        else:
            self.time_idx = kwargs['time_idx']
 
        if kwargs.get('kernel') is None:
            self.kernel = 'rbf'
        else:
            self.kernel = kwargs.get('kernel')

        self.missing = np.ma.isMaskedArray(self.Y)


    def _conditional_prior_sparse(self, i, hyp):            
        i_filter    = np.ones(self.N).astype(bool)
        i_filter[i] = False
        prior_mean  = np.empty(self.D)
        prior_var   = np.empty(self.D)
        t_i     = self.time_idx[i].reshape(-1,1)
        t_not_i = self.time_idx[i_filter].reshape(-1,1)
        X_not_i = self.X[i_filter]

        if self.kernel == 'rbf_linear':
            kern_fun = self._rbf_linear_kernel
        elif self.kernel == 'rbf':
            kern_fun = self._rbf_kernel
        elif self.kernel == 'cauchy':
            kern_fun = self._cauchy_kernel
        elif self.kernel == 'OU':
            kern_fun = self._OU_kernel
            
        for d, hyp_d in enumerate(hyp):                    
            ls, amp, noise = hyp_d
            KM     = kern_fun(self.time_sparse, hyp=(ls, amp))
            KMN    = kern_fun(self.time_sparse, t_not_i, hyp=(ls, amp))
            KMn    = kern_fun(self.time_sparse, t_i, hyp=(ls, amp))

            Lambda_inv = np.diag(kern_fun(t_not_i, hyp=(ls, amp)) \
                                 - np.dot(KMN.T, np.linalg.solve(KM,KMN)))
            Lambda_inv = 1. / (Lambda_inv + noise)
            Q_m    = np.dot(KMN, np.multiply(Lambda_inv[:,None], KMN.T)) + KM 
            prior_mean[d] = np.dot(np.dot(KMn.T, np.linalg.solve(Q_m, KMN)),
                      np.multiply(Lambda_inv[:,None], X_not_i[:,d,np.newaxis]))        
            prior_var[d]  = kern_fun(t_i, hyp=(ls, amp))
            prior_var[d] -= KMn.T @ np.linalg.solve(KM, KMn)
            prior_var[d] += KMn.T @ np.linalg.solve(Q_m, KMn)
            prior_var[d] += noise
            if (prior_var[d] <= 0.):
                 # this happens because of numerical error, so set variance to obs. noise
                prior_var[d] = noise
        return prior_mean, prior_var

    def _conditional_prior_control_var(self, i, hyp):            
        i_filter    = np.ones(self.N).astype(bool)
        i_filter[i] = False
        prior_mean  = np.empty(self.D)
        prior_var   = np.empty(self.D)
        t_i     = self.time_idx[i].reshape(-1,1)
        t_not_i = self.time_idx[i_filter].reshape(-1,1)
        X_not_i = self.X[i_filter]

        if self.kernel == 'rbf_linear':
            kern_fun = self._rbf_linear_kernel
        elif self.kernel == 'rbf':
            kern_fun = self._rbf_kernel
        elif self.kernel == 'cauchy':
            kern_fun = self._cauchy_kernel
        elif self.kernel == 'OU':
            kern_fun = self._OU_kernel
            
        for d, hyp_d in enumerate(hyp):                    
            ls, amp, noise = hyp_d
            KM     = kern_fun(self.time_sparse, hyp=(ls, amp))
            KMN    = kern_fun(self.time_sparse, t_not_i, hyp=(ls, amp))
            KMn    = kern_fun(self.time_sparse, t_i, hyp=(ls, amp))

            Lambda_inv = np.diag(kern_fun(t_not_i, hyp=(ls, amp)) \
                                 - np.dot(KMN.T, np.linalg.solve(KM,KMN)))
            Lambda_inv = 1. / (Lambda_inv + noise)
            Q_m    = np.dot(KMN, np.multiply(Lambda_inv[:,None], KMN.T)) + KM 

            f_var  = np.dot(KM, np.linalg.solve(Q_m, KM))
            f_mean = np.dot(np.dot(KM, np.linalg.solve(Q_m,KMN)), 
                            np.multiply(Lambda_inv[:,None], 
                                        X_not_i[:,d,np.newaxis])).flatten()
#            pdb.set_trace()
            f      = self.rng.multivariate_normal(f_mean,f_var)
            prior_mean[d] = np.dot(KMn.T, np.linalg.solve(KM, f))
#            prior_mean[d] = np.dot(np.dot(KMn.T, np.linalg.solve(Q_m, KMN)),
#                      np.multiply(Lambda_inv[:,None], X_not_i[:,d,np.newaxis]))        
            prior_var[d]  = kern_fun(t_i, hyp=(ls, amp))
            prior_var[d] -= KMn.T @ np.linalg.solve(KM, KMn)
#            prior_var[d] += KMn.T @ np.linalg.solve(Q_m, KMn)
            prior_var[d] += noise
            if (prior_var[d] <= 0.):
                 # this happens because of numerical error, so set variance to obs. noise
                prior_var[d] = noise
        return prior_mean, prior_var
            
    def _conditional_prior(self, block, hyp):
        N_block = len(block)
        i_filter = [i for i in np.arange(self.N) if i not in block]

        if N_block ==1:
            prior_mean = np.zeros(self.D)
            prior_var = np.empty(self.D)
        else:
            prior_mean = np.zeros((self.D,N_block))
            prior_var = np.empty((self.D,N_block,N_block))

        t_i = self.time_idx[block].reshape(-1,1)
        t_not_i = self.time_idx[i_filter].reshape(-1,1)
        X_not_i = self.X[i_filter]
        kern_fun = self._rbf_linear_kernel

        if self.kernel == 'rbf_linear':
            kern_fun = self._rbf_linear_kernel
        elif self.kernel == 'rbf':
            kern_fun = self._rbf_kernel
        elif self.kernel == 'cauchy':
            kern_fun = self._cauchy_kernel
        elif self.kernel == 'OU':
            kern_fun = self._OU_kernel
            
        for d, hyp_d in enumerate(hyp):                    
            ls, amp, noise = hyp_d                  
            Kxx_i = kern_fun(t_not_i,t_i,hyp=(ls, amp)) 
            Kx    = kern_fun(t_not_i,hyp=(ls, amp)) + noise*np.eye(t_not_i.size)
            prior_var[d]  = kern_fun(t_i, hyp=(ls, amp))
            prior_var[d] -= Kxx_i.T @ np.linalg.solve(Kx, Kxx_i)
            prior_mean[d] = Kxx_i.T @ np.linalg.solve(Kx, X_not_i[:,d])
        return prior_mean, prior_var

    def _rbf_kernel(self, X, X_i=None, hyp=None):
        """Radial basis function.
        
        RBF kernel was taken from `autograd`: 
        https://github.com/HIPS/autograd/blob/master/examples/
                gaussian_process.py
        diag_xx is fixed to 1e-6
        """
        len_xx, var_xx = hyp
        diag_xx = 1e-6 # jitter term
        X = X.reshape(-1,1)
        if X_i is None:
            diffs = np.expand_dims(X, 1) \
                    - np.expand_dims(X, 0)
            if diffs.size == 1:
                A = np.exp(-len_xx * (diffs ** 2))
                B = float(diag_xx)
            else:
                A = np.exp(-len_xx * np.sum(diffs ** 2, axis=2))
                B = diag_xx * np.eye(len(X))
            return (var_xx * A) + B

        else:
            X_i = X_i.reshape(-1,1)
            diffs = np.expand_dims(X, 1) \
                    - np.expand_dims(X_i, 0)
            if diffs.size == 1:
                A = np.exp(-len_xx * (diffs ** 2))
            else:
                A = np.exp(-len_xx * np.sum(diffs ** 2, axis=2))
            return var_xx * A
                
    def _rbf_linear_kernel(self, X, X_i=None, hyp=None):
        # TODO: generalize this to different kernels
        """Radial basis function and linear kernel.
        
        RBF kernel was taken from `autograd`: 
        https://github.com/HIPS/autograd/blob/master/examples/
                gaussian_process.py
        diag_xx is fixed to 1e-6
        """
        len_xx, var_xx = hyp
        diag_xx = 1e-6 # jitter term
#        var_xx = 0.5 # amplitude, fixed so that diagonal variance is 1
        X = X.reshape(-1,1)
        if X_i is None:
            diffs = np.expand_dims(X, 1) \
                    - np.expand_dims(X, 0)
            if diffs.size == 1:
                A = np.exp(-len_xx * (diffs ** 2))
                B = float(diag_xx)
                C = X * X
            else:
                A = np.exp(-len_xx * np.sum(diffs ** 2, axis=2))
                B = diag_xx * np.eye(len(X))
                C = np.outer(X, X.T)
            return var_xx * (A + C) + B

        else:
            X_i = X_i.reshape(-1,1)
            diffs = np.expand_dims(X, 1) \
                    - np.expand_dims(X_i, 0)
            if diffs.size == 1:
                A = np.exp(-len_xx * (diffs ** 2))
                C = X * X_i
            else:
                A = np.exp(-len_xx * np.sum(diffs ** 2, axis=2))
                C = X @ X_i.T
            return var_xx * (A + C)
        
    def _cauchy_kernel(self, X, X_i=None, hyp=None):
        len_xx, var_xx = hyp
        diag_xx = 1e-6 # jitter term
        X = X.reshape(-1,1)
        if X_i is None:
            diffs = np.expand_dims(X, 1) \
                    - np.expand_dims(X, 0)
            if diffs.size == 1:
                A = 1. / (1. + len_xx * (diffs ** 2))
                B = float(diag_xx)
            else:
                A = 1. / (1. +  len_xx * np.sum(diffs ** 2, axis=2))
                B = diag_xx * np.eye(len(X))
            return var_xx*A + B

        else:
            X_i = X_i.reshape(-1,1)
            diffs = np.expand_dims(X, 1) \
                    - np.expand_dims(X_i, 0)
            if diffs.size == 1:
                A = 1. / (1. + len_xx * (diffs ** 2))
            else:
                A = 1. / (1. + len_xx * np.sum(diffs ** 2, axis=2))
            return var_xx*A

    def _OU_kernel(self, X, X_i=None, hyp=None):
        len_xx, var_xx = hyp
        diag_xx = 1e-6 # jitter term
        X = X.reshape(-1,1)
        if X_i is None:
            diffs = np.expand_dims(X, 1) \
                    - np.expand_dims(X, 0)
            if diffs.size == 1:
                A = np.exp(-len_xx * np.abs(diffs))
                B = float(diag_xx)
            else:
                A = np.exp(-len_xx * np.sum(np.abs(diffs), axis=2))
                B = diag_xx * np.eye(len(X))
            return var_xx*A + B

        else:
            X_i = X_i.reshape(-1,1)
            diffs = np.expand_dims(X, 1) \
                    - np.expand_dims(X_i, 0)
            if diffs.size == 1:
                A = np.exp(-len_xx * np.abs(diffs))
            else:
                A = np.exp(-len_xx * np.sum(np.abs(diffs), axis=2))
            return var_xx*A 

    def _sparse_marginal_likelihood_GP(self, hyp, pseudo_input, d): 
        """Calculates the marginal likelihood of the FITC GP model
        """
        if self.kernel == 'rbf_linear':
            kern_fun = self._rbf_linear_kernel
        elif self.kernel == 'rbf':
            kern_fun = self._rbf_kernel
        elif self.kernel == 'cauchy':
            kern_fun = self._cauchy_kernel
        elif self.kernel == 'OU':
            kern_fun = self._OU_kernel
     
        hyp = logexp(hyp)
        ls, amp, noise = hyp
        X_id = self.X[:,d].reshape(-1,1)
        # sparse GP calculation from GPy
        
        Kmm = kern_fun(pseudo_input, hyp=(ls, amp)) \
                + (1e-6)*np.eye(self.M_div_2)
        Knn = np.diag(kern_fun(self.time_idx, hyp=(ls, amp)))
        Knm = kern_fun(self.time_idx, pseudo_input, hyp=(ls, amp))

        #factor Kmm
        Kmmi, L, Li, _ = pdinv(Kmm)

        #compute beta_star, the effective noise precision
        LiUT = np.dot(Li, Knm.T)
        sigma_star = Knn - np.sum(np.square(LiUT),0) + noise
        beta_star = 1./sigma_star

        # Compute and factor A
        A = tdot(LiUT*np.sqrt(beta_star)) + np.eye(self.M_div_2)
        LA = jitchol(A)

        # back substutue to get b, P, v
        URiy = np.dot(Knm.T*beta_star,X_id)
        tmp, _ = dtrtrs(L, URiy, lower=1)
        b, _ = dtrtrs(LA, tmp, lower=1)
        tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
        v, _ = dtrtrs(L, tmp, lower=1, trans=1)
        tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)

        #compute log marginal
        LL     = -0.5*self.N*np.log(2*np.pi) + \
                 -np.sum(np.log(np.diag(LA))) + \
                 0.5*np.sum(np.log(beta_star)) + \
                 -0.5*np.sum(np.square(X_id.T*np.sqrt(beta_star))) + \
                 0.5*np.sum(np.square(b))

        return LL
                    
    def _sample_hyp(self):
        """Update GP hyperparams with ESS
        """
        def marg_GP(param, **kwargs):
            return np.sum([self._sparse_marginal_likelihood_GP(param[d], self.time_sparse, d) for d in range(self.D)])

        new_hyp = np.empty(self.gp_hyp.shape)

        prior_mean = np.zeros(3)
        prior_var  = self.hyp_var*np.ones(3)
        if self.noiseless:
            prior_mean[1] = inv_logexp(1)
            prior_mean[2] = inv_logexp(1e-6)
            prior_var[1] = .0000001
            prior_var[2] = .0000001
        hyp_ess = ESS(rng=self.rng, 
                      init_param=self.gp_hyp, 
                      prior_mean=prior_mean, 
                      prior_var=prior_var)
        
        hyp_ess._transform = logexp
        hyp_ess._inv_transform = inv_logexp
        hyp_ess._log_likelihood = marg_GP
        new_hyp = hyp_ess.step()

        def sparse_marg_GP(param, **kwargs):
            param = np.ravel(param)
            return np.sum([self._sparse_marginal_likelihood_GP(inv_logexp(new_hyp[d]),
                                                               param, d) \
                            for d in range(self.D)])

        sparse_ess = ESS(rng=self.rng, 
                         init_param=self.time_sparse.reshape(1,-1), 
                         prior_mean=self.time_idx.mean()*np.ones(self.M_div_2), 
                         prior_var=self.time_idx.var()*np.ones(self.M_div_2))
    
        sparse_ess._log_likelihood = sparse_marg_GP
        new_pseudo = np.ravel(sparse_ess.step())
        return (new_hyp, new_pseudo)        
        
    def _sample_x(self):
        """Sample `X` according to dynamical model.
        """
        self.gp_hyp, self.time_sparse = self._sample_hyp()

        for i in range(self.N):
            self.X[i] = self._sample_x_i_ess(i)
        
    def _sample_x_i_ess(self, i):
        if self.cv:
#            print("control variates")
            cond_mean, cond_var = self._conditional_prior_control_var([i], self.gp_hyp)
        else:
            cond_mean, cond_var = self._conditional_prior_sparse([i], self.gp_hyp)
        def ess_LL(param, kwargs):
            i = kwargs['i']
            X = kwargs['X']
            X[i] = param
            return self._log_likelihood_i(X,i)
        
        x_ess = ESS(rng=self.rng, init_param=self.X[i].reshape(1,-1), 
                    prior_mean=cond_mean, prior_var=cond_var)
        
        x_ess._log_likelihood = ess_LL
        return x_ess.step(X=np.copy(self.X), i=i)
        
    def _log_likelihood_i(self,X,i):
        """Compute likelihood of `X_i`.
        """
        raise NotImplementedError()
    
    def _init_common_params(self):
        """Initialize parameters common to RFLVMs.
        """       
        super()._init_common_params()
        self.gp_hyp = np.ones((self.D,3)) # ls, amplitude, noise
        if self.noiseless: # fix amplitude to 1 and noise to 1e-6
            self.gp_hyp[:,-1] = 1e-6
        self.time_sparse = self.rng.normal(size=self.M_div_2)