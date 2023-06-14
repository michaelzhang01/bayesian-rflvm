# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:35:05 2021

"""
from   datasets import load_dataset
from   logger import (format_number,
                      Logger)
from   models import (GaussianRFLVM,
                      PoissonRFLVM,
                      DynamicGaussianRFLVM,
                      DynamicPoissonRFLVM)
from   metrics import (knn_classify,
                       mean_squared_error,
                       r_squared)
from   new_gpdm.gpdm import GPDM
from   ppca.ppca_ma import PPCA
from   numpy.random import RandomState
from   numpy import savetxt
from   pathlib import Path
from   time import perf_counter
# from   visualizer import Visualizer
import os
import pickle

if __name__ == '__main__':
    seed = 10
    plot_every = 1
    rng = RandomState(seed)
    emissions = 'gaussian'
#    emissions = 'gaussian'
#    emissions = 'negbinom'
    name = 'congress'
    test_split = .2
    ds  = load_dataset(rng=rng, 
                       name=name, 
                       emissions=emissions, 
                       test_split=test_split)
    directory = os.path.abspath("./temp/"+name+"_"+emissions+"/")
    p = Path(directory)
    if not p.exists():
        p.mkdir()

#    ds  = load_dataset(rng, 'hippo', emissions)
#    ds  = load_dataset(rng, 's-curve', emissions)
#    ds  = load_dataset(rng, 'spike', None)
#    ds  = load_dataset(rng, 'slds', emissions)
#    ds  = load_dataset(rng, 'mnist', None)
    # viz = Visualizer(directory, ds)
    log = Logger(directory=directory)
    log.log(f'Initializing RNG with seed {seed}.')    
    n_clusters     = 1
    n_iters        = 100
    n_rffs         = 100
    n_burn         = int(n_iters / 2)  # Recommended in Gelman's BDA.
    dp_prior_obs   = ds.latent_dim
    dp_df          = ds.latent_dim + 1
    marginalize    = True
    log_every      = 50.
    time_idx       = None
    hyp_var        = 5.
    num_p          = 5
    disp_prior     = 1.
    bias_var       = 5.
    x_init         = 'pca'
    optimize       = False
    marginalize_IS = False
    kernel         = 'rbf'
    sparse         = True
    gmm_X          = False
    latent_states  = 20
    noiseless      = False
#    model = NegativeBinomialRFLVM(
#    model = DynamicGaussianRFLVM(
    model = PoissonRFLVM(
#    model = PPCA(
#    model = DynamicPoissonRFLVM(
#    model = StateSpaceNegativeBinomialRFLVM(
#    model = StateSpacePoissonRFLVM(
#    model = StateSpaceGaussianRFLVM(
            rng=rng,
            data=ds.Y_ma,
            n_burn=n_burn,
            n_iters=n_iters,
            latent_dim=ds.latent_dim,
            n_clusters=n_clusters,
            n_rffs=n_rffs,
            dp_prior_obs=dp_prior_obs,
            dp_df=dp_df,
            marginalize=marginalize,
            time_idx=time_idx,
            hyp_var=hyp_var,
            num_p=num_p,
            disp_prior=disp_prior,
            bias_var=bias_var,
            x_init=x_init,
            optimize=optimize,
            marginalize_IS=marginalize_IS,
            kernel=kernel,
            sparse=sparse,
            gmm_X=gmm_X,
            latent_states=latent_states,
            noiseless=noiseless,
            linear=False
        )
    
    # viz.plot_X_init(model.X)

    s_start = perf_counter()
    for t in range(n_iters):
        s = perf_counter()
        model.step()
        e = perf_counter() - s

        if t == model.n_burn:
            log.log_hline()
            log.log(f'Burn in complete on iter = {t}. Now plotting using mean '
                    f'of `X` samples after burn in.')
        if (t % plot_every == 0):
            assert(model.t-1 == t)
            Y_pred, F_pred, K_pred = model.predict(model.X, return_latent=True)
            # viz.plot_iteration(t, Y_pred, F_pred, K_pred, model.X)
            log.log_hline()
            log.log(t)

            mse_Y = mean_squared_error(Y_pred[ds.mask], ds.Y[ds.mask])
            log.log_pair('MSE Y', mse_Y)
        
            if ds.has_true_F:
                mse_F = mean_squared_error(F_pred[ds.mask], ds.F[ds.mask])
                log.log_pair('MSE F', mse_F)
        
            if ds.has_true_K:
                mse_K = mean_squared_error(K_pred, ds.K)
                log.log_pair('MSE K', mse_K)
        
            if ds.has_true_X:
                r2_X = r_squared(model.X, ds.X)
                log.log_pair('R2 X', r2_X)
        
            if ds.is_categorical:
                knn_acc = knn_classify(model.X, ds.labels, rng)
                log.log_pair('KNN acc', knn_acc)
        
#            log.log_pair('DPMM LL', model.calc_dpgmm_ll())
#            log.log_pair('K', model.Z_count.tolist())
#            log.log_pair('alpha', model.alpha)
#            n_mh_iters = (model.t + 1) * model.M
#            log.log_pair('W MH acc', model.mh_accept / n_mh_iters)

            log.log_pair('time', e)
    elapsed_time = (perf_counter() - s_start) / 3600
    log.log_hline()
    log.log(f'Finished job in {format_number(elapsed_time)} (hrs).')
    log.log_hline()
    Y_pred, F_pred, K_pred = model.predict(model.X, return_latent=True)
#    fpath = os.path.join(directory,'nb_rflvm.pickle')
#    pickle.dump(params, open(fpath, 'wb'))

#    savetxt(os.path.join(directory,'bunny_nb_Y_pred.txt'),Y_pred)
    
