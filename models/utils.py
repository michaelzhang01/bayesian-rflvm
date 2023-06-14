"""Utility functions"""

import numpy as np 


def logexp(x): # reals -> positive reals
    _log_lim_val = np.log(np.finfo(np.float64).max)
    _lim_val = 36.0
    return np.where(x>_lim_val, x, np.log1p(np.exp(np.clip(x, -_log_lim_val, _lim_val))))


def inv_logexp(f): # positive reals -> reals
    _lim_val = 36.0
    return np.where(f>_lim_val, f, np.log(np.expm1(f)))


def jitchol_ag(A, maxtries=5): # autograd friendly version of jitchol
    diagA = np.diag(A)
    if np.any(diagA <= 0.):
        raise np.linalg.LinAlgError("not pd: non-positive diagonal elements")
    jitter = diagA.mean() * 1e-6
    num_tries = 1
    while num_tries <= maxtries and np.isfinite(jitter):
        try:
            L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter)
            return L
        except:
            jitter *= 10
        finally:
            num_tries += 1
    raise np.linalg.LinAlgError("not positive definite, even with jitter.")
