#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:09:56 2021

Base class for Elliptical Slice Sampling

@author: Michael Zhang
"""

import numpy as np
import pdb

class ESS(object):    
    def __init__(self, rng, init_param, prior_mean, prior_var):
        self.rng   = rng        
        self.param = init_param
        self.N,self.D     = self.param.shape
        self.prior_mean = prior_mean
        assert(self.prior_mean.size==self.D)
        
        self.prior_var = prior_var
        if self.prior_var.shape == (self.D,self.D):
            self.square = True
        elif self.prior_var.size == self.D:
            self.square = False
        else:
            raise ValueError
    
    def _log_likelihood(self, param, kwargs):
        raise NotImplementedError()
    
    def _transform(self, x): # transforms reals to parameter space
        return 1. * x

    def _inv_transform(self, x): # transforms parameter space to reals
        return 1. * x

    def _proposal(self, theta, nu):
        transform_x = (((self._inv_transform(self.param) \
                         - self.prior_mean)*np.cos(theta)) \
                         + (nu * np.sin(theta))) + self.prior_mean
        return transform_x
    
    def step(self, **kwargs):
        if self.square:
            nu = self.rng.multivariate_normal(mean=np.zeros(self.D),
                                              cov=self.prior_var,
                                              size=self.N)
        else:
            nu = self.rng.normal(scale=np.sqrt(self.prior_var),
                                 size=(self.N,self.D))

        log_u          = np.log(self.rng.uniform())
        init_param     = self._inv_transform(self.param)
        LL             = self._log_likelihood(param=init_param, 
                                              kwargs=kwargs) + log_u
        theta          = self.rng.uniform(0.,2.*np.pi)
        theta_min      = theta - 2.*np.pi
        theta_max      = float(theta)
        proposal_param = self._proposal(theta, nu)
        proposal_LL    = self._log_likelihood(param=proposal_param, 
                                              kwargs=kwargs)
        if np.isnan(proposal_LL):
            pdb.set_trace()
        while proposal_LL < LL:
            if theta < 0:
                theta_min = float(theta)
            else:
                theta_max = float(theta)
            theta          = self.rng.uniform(theta_min,theta_max)
            proposal_param = self._proposal(theta, nu)
            proposal_LL    = self._log_likelihood(param=proposal_param, 
                                                  kwargs=kwargs)
            if np.isnan(proposal_LL):
                pdb.set_trace()


        self.param = self._transform(proposal_param)

        return self.param
    
if __name__ == '__main__':
    pass