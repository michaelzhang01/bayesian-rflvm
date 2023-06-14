"""============================================================================
Dataset loading functions.
============================================================================"""

from   constants import (LOCAL_DIR,
                         REMOTE_DIR,
                         DATASETS)
from   datasets.dataset import Dataset
from   GPy import kern
import numpy as np
import os
import pandas as pd
from   scipy.io import loadmat
from   scipy.special import (expit as logistic,
                             logsumexp)
from   sklearn.datasets import make_s_curve

# -----------------------------------------------------------------------------

if 'gwg3' in os.getcwd():
    BASE_DIR = f'{REMOTE_DIR}/datasets'
else:
    BASE_DIR = f'{LOCAL_DIR}/datasets'

# -----------------------------------------------------------------------------

def load_dataset(rng, name, emissions, test_split=0):
    """Given a dataset string, returns data and possibly true generative
    parameters.
    """
    loader = {
        'bridges'       : load_bridges,
        'congress'      : load_congress,
        'lorenz'        : load_lorenz,
        's-curve'       : gen_s_curve,
        'spikes'        : load_spikes,
    }[name]

    if name == 's-curve' or name == 's-curve-batch':
        return loader(rng, emissions, test_split)
    else:
        return loader(rng, test_split)

# -----------------------------------------------------------------------------

def load_bridges(rng, test_split):
    """Load NYC bridges dataset:
    https://data.cityofnewyork.us/Transportation/
      Bicycle-Counts-for-East-River-Bridges/gua4-p9wg
    """
    data   = np.load(f'{BASE_DIR}/bridges/bridges.npy', allow_pickle=True)
    data   = data[()]
    Y      = data['Y']
    labels = data['labels']
    return Dataset(rng, 'bridges', True, Y=Y, labels=labels, 
                   test_split=test_split)

# -----------------------------------------------------------------------------

def load_lorenz(rng, test_split):
    def dyn_lorenz(T, dt=0.01):

        stepCnt = T

        def lorenz(x, y, z, s=10, r=28, b=2.667):
            x_dot = s*(y - x)
            y_dot = r*x - y - x*z
            z_dot = x*y - b*z
            return x_dot, y_dot, z_dot

        # Need one more for the initial values
        xs = np.empty((stepCnt + 1,))
        ys = np.empty((stepCnt + 1,))
        zs = np.empty((stepCnt + 1,))

        # Setting initial values
        xs[0], ys[0], zs[0] = (0., 1., 1.05)

        # Stepping through "time".
        for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)

        z = np.zeros((T, 3))
        z[:,0] = xs[:-1]
        z[:,1] = ys[:-1]
        z[:,2] = zs[:-1]
        return z

    def map_tanh(z, D, J):
        Wz_true = np.random.normal(0, 1,[D,J])
        mu = np.dot(z, Wz_true)
        return np.tanh(mu)

    N = 500
    J = 50
    D = 3

    z_all = dyn_lorenz(N)
    z_sim = z_all[-N:,:]
    z_sim_norm  = z_sim - z_sim.mean(axis=0)
    z_sim_norm /= np.linalg.norm(z_sim_norm, axis=0, ord=np.inf)
    X  = np.copy(z_sim_norm)

    F  = 10.*map_tanh(X, D, J)
    F  -= F.mean(axis=0)
    F  /= F.std(axis=0)
    Y  = F + rng.normal(size=(F.shape))
    
    t  = np.linspace(-1,1,Y.shape[0])

    return Dataset(rng, 'lorenz', False, Y=Y, X=X, F=F, latent_dim=D, labels=t,
                   test_split=test_split)

# -----------------------------------------------------------------------------

def load_congress(rng, test_split):
    """Congress 109 data:
    https://github.com/jgscott/STA380/blob/master/data/congress109.csv
    https://github.com/jgscott/STA380/blob/master/data/congress109members.csv
    """
    df1 = pd.read_csv(f'{BASE_DIR}/congress109.csv')
    df2 = pd.read_csv(f'{BASE_DIR}/congress109members.csv')
    assert (len(df1) == len(df2))

    # Ensure same ordering.
    df1 = df1.sort_values(by='name')
    df2 = df2.sort_values(by='name')

    Y = df1.values[:, 1:].astype(int)
    labels = np.array([0 if x == 'R' else 1 for x in df2.party.values])
    return Dataset(rng, 'congress109', True, Y, labels=labels,
                   test_split=test_split)

# -----------------------------------------------------------------------------

def load_spikes(rng, test_split):
    """Synthetic data of grid cell responses during 2D random walks in real
    space.
    """
    data = np.load(f'{BASE_DIR}/spks.npy', allow_pickle=True)
    data = data[()]
    Y    = data['Y']
    X    = data['X']
    return Dataset(rng, 'synthspikes', False, Y, X=X,
                   test_split=test_split)

# -----------------------------------------------------------------------------

def gen_s_curve(rng, emissions, test_split):
    """Generate synthetic data from datasets generating process.
    """
    N = 500
    J = 100
    D = 2

    # Generate latent manifold.
    # -------------------------
    X, t = make_s_curve(N, random_state=rng)
    X    = np.delete(X, obj=1, axis=1)
    X    = X / np.std(X, axis=0)
    inds = t.argsort()
    X    = X[inds]
    t    = t[inds]

    # Generate kernel `K` and latent GP-distributed maps `F`.
    # -------------------------------------------------------

    K = kern.RBF(input_dim=D, lengthscale=1).K(X)
    F = rng.multivariate_normal(np.zeros(N), K, size=J).T

    # Generate emissions using `F` and/or `K`.
    # ----------------------------------------
    if emissions == 'bernoulli':
        P = logistic(F)
        Y = rng.binomial(1, P).astype(np.double)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, 
                       labels=t, test_split=test_split)
    if emissions == 'gaussian':
        Y = F + np.random.normal(0, scale=0.5, size=F.shape)
        return Dataset(rng, 's-curve',False, Y=Y, X=X, F=F, latent_dim=D, 
                       labels=t, test_split=test_split)
    elif emissions == 'multinomial':
        C = 100
        pi = np.exp(F - logsumexp(F, axis=1)[:, None])
        Y = np.zeros(pi.shape)
        for n in range(N):
            Y[n] = rng.multinomial(C, pi[n])
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t,
                   test_split=test_split)
    elif emissions == 'negbinom':
        P = logistic(F)
        R = np.arange(1, J+1, dtype=float)
        Y = rng.negative_binomial(R, 1-P)
        return Dataset(rng, 's-curve', False, False, Y=Y, X=X, F=F, R=R, 
                       latent_dim=D, labels=t, test_split=test_split)
    else:
        assert(emissions == 'poisson')
        theta = np.exp(F)
        Y = rng.poisson(theta)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, 
                       labels=t, test_split=test_split)



