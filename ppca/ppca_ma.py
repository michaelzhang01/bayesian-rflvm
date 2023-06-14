import numpy as np
from scipy import special
#from numpy.random import randn

## updating W, X and tau when doing inference
class PPCA:
    ## Y: input continuous data with shape (N, M)
    ## D: number of ppca components
    # todo: take in rng
    def __init__(self, rng, data, latent_dim = 2, n_iters = 100, verbose = False,
                 tol=1e-3, **kwargs):
        self.rng = rng
        self.Y = data 
        self.N, self.M = self.Y.shape
        self.D = latent_dim
        self.tol = tol
        self.n_iters = n_iters
        self.verbose = verbose
        self._init_paras()
        
    def _init_paras(self):
        self.a = 1.0
        self.b = 1.0
        self.e_tau = self.a / self.b
        self.W = self.rng.normal(size=(self.M, self.D))
        self.e_wwt = np.zeros((self.D, self.D, self.M))
        for m in range(self.M):
            ## use np.newaxis here to transfer numpy array from 1D to 2D
            self.e_wwt[:, :, m] = np.eye(self.D) + self.W[m, :][np.newaxis].T.dot(self.W[m, :][np.newaxis])

        self.lbs = np.zeros(self.n_iters)
        self.X = np.zeros((self.N, self.D))
        self.e_XXt = np.zeros((self.D, self.D, self.N))

    def _update_X(self):
        self.sigx = np.linalg.inv(np.eye(self.D) + self.e_tau * np.sum(self.e_wwt, axis = 2))
        for n in range(self.N):
            self.X[n, :] = self.e_tau * np.ma.dot(self.sigx, (np.ma.sum(self.W * np.tile(self.Y[n, :], (self.D, 1)).T, axis = 0)))
            self.e_XXt[:, :, n] = self.sigx + self.X[n, :][np.newaxis].T.dot(self.X[n, :][np.newaxis])

    def _update_W(self):
        self.sigw = np.linalg.inv(np.eye(self.D) + self.e_tau * np.sum(self.e_XXt, axis = 2))
        for m in range(self.M):
            self.W[m, :] = self.e_tau * np.ma.dot(self.sigw, np.ma.sum(self.X * np.tile(self.Y[:, m], (self.D, 1)).T, axis = 0))
            self.e_wwt[:, :, m] = self.sigw + self.W[m, :][np.newaxis].T.dot(self.W[m, :][np.newaxis])

    def _update_tau(self):
        self.e = self.a +self.N * self.M * 1.0 / 2
        outer_expect = 0
        for n in range(self.N):
            for m in range(self.M):
                outer_expect = outer_expect \
                                + np.trace(self.e_wwt[:, :, m].dot(self.sigx)) \
                                + self.X[n, :][np.newaxis].dot(self.e_wwt[:, :, m]).dot(self.X[n, :][np.newaxis].T)[0][0]
        self.f = self.b + 0.5 * np.ma.sum(self.Y ** 2) - np.ma.sum(self.Y * self.W.dot(self.X.T).T) + 0.5 * outer_expect
        self.e_tau = self.e / self.f
        self.e_log_tau = np.mean(np.log(np.random.gamma(self.e, 1/self.f, size=1000)))
    
    def lower_bound(self):
        LB = self.a * np.log(self.b) + (self.a - 1) * self.e_log_tau - self.b * self.e_tau - special.gammaln(self.a)
        LB = LB - (self.e * np.log(self.f) + (self.e - 1) * self.e_log_tau - self.f * self.e_tau - special.gammaln(self.e))
        for n in range(self.N):
            LB = LB + (-(self.D*0.5)*np.log(2*np.pi) - 0.5 * (np.trace(self.sigx) + self.X[n, :][np.newaxis].dot(self.X[n, :][np.newaxis].T)[0][0]))
            LB = LB - (-(self.D*0.5)*np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(self.sigx)) - 0.5 * self.D)
        for m in range(self.M):
            LB = LB + (-(self.D*0.5)*np.log(2*np.pi) - 0.5 * (np.trace(self.sigw) + self.W[m, :][np.newaxis].dot(self.W[m, :][np.newaxis].T)[0][0]))
            LB = LB - (-(self.D*0.5)*np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(self.sigw)) - 0.5 * self.D)
        outer_expect = 0
        for n in range(self.N):
            for m in range(self.M):
                outer_expect = outer_expect \
                                + np.trace(self.e_wwt[:, :, m].dot(self.sigx)) \
                                + self.X[n, :][np.newaxis].dot(self.e_wwt[:, :, m]).dot(self.X[n, :][np.newaxis].T)[0][0]

        LB = LB + ( \
            -(self.N * self.M * 1.0 / 2) * np.log(2 * np.pi) + (self.N * self.M * 1.0 / 2) * self.e_log_tau \
            - 0.5 * self.e_tau * (np.ma.sum(self.Y**2) - 2 * np.ma.sum(self.Y * self.W.dot(self.X.T).T) + outer_expect))
        return LB
    
    def step(self):
        self._update_X()
        self._update_W()
        self._update_tau()
        
    def fit(self):    
        for it in range(self.n_iters):
            self.step()
            self.lbs[it] = self.lower_bound()
            if it >= 1:                
                if np.abs(self.lbs[it] - self.lbs[it - 1]) < self.tol:
                    break

    def fit_transform(self):
        self.fit()
        return self.X
    
    def predict(self, X=None):
        if X is None:
            return self.X.dot(self.W.T)
        else:
            return X.dot(self.W.T)        

if __name__ == '__main__':
    from   sklearn.datasets import make_s_curve
    import matplotlib.pyplot as plt
    from   sklearn.metrics.pairwise import rbf_kernel
    from   numpy.random import RandomState
    
    def gen_data():
        T    = 200
        J    = 40
        X, t = make_s_curve(T)
        X    = np.delete(X, obj=1, axis=1)
        X    = X / np.std(X, axis=0)
        inds = t.argsort()
        X    = X[inds]
        t    = t[inds]
        K    = rbf_kernel(X)
#        K    = X @ X.T
        F    = np.random.multivariate_normal(np.zeros(T), K, size=J).T
        Y    = F + np.random.normal(0, scale=1, size=F.shape)
        return X, Y, t
    seed = 10
    rng = RandomState(seed)

    X, Y, t = gen_data()
    mask = rng.binomial(1,.2,size=Y.shape).astype(bool)
    Y_ma = np.copy(Y)
    Y_ma[mask] = np.nan
    Y_ma = np.ma.array(Y_ma,mask=mask)
    Y_ma = Y_ma.harden_mask()
    ppca = PPCA(rng, Y)
    ppca.fit()
    Y_pred = ppca.predict()
    MSE = np.mean(((Y-Y_pred)[mask])**2)
    print('mse: %.2f' % MSE)
    plt.scatter(ppca.X[:,0],ppca.X[:,1],c=t)
    plt.show()
#        ppca._update()
