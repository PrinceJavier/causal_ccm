import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import pearsonr
from causal_ccm import ccm

class pai(ccm):
    
    def __init__(self, X, Y, tau=1, manifold_pattern=[[0, -1],[0]], L=None):
        '''
        X: timeseries for variable X that could cause Y
        Y: timeseries for variable Y that could be caused by X
        tau: time lag, assumed equal for both X and Y manifolds
        E: shadow manifold embedding dimension of X, derived from manifold_pattern[0]
        Ey: shadow manifold embedding dimension of Y, derived from manifold_pattern[1]
        manifold_pattern: a list of 2 lists that contains the lag vectors for X and Y shadow manifolds.
            For example, manifold_pattern=[[0, -1, -2],[0]], tau=2 is the same as the shadow manifold 
            (X_t, X_{t-1*2}, X_{t-2*2}, Y_t).
            A CCM of E=2, tau=2 can be represented as [[],[0, -1]], tau=2 or (Y_t, Y_{t-2}).
            Note that the lag vectors must have 1-decrement. Tau will handle the time delta.
        L: time period/duration to consider (longer = more data)
        We're checking for X -> General shadow manifold constructed from X, Y, or both X and Y
        '''
        self.E = len(manifold_pattern[0])
        self.Ey = len(manifold_pattern[1])
        self.manifold_pattern = manifold_pattern
        
        self.X = X
        self.Y = Y
        self.tau = tau
        
        if L == None:
            self.L = len(X)
        else:
            self.L = L
            
        self.My = self.shadow_manifold() # shadow manifold for Y (we want to know if info from X is in Y)
        self.t_steps, self.dists = self.get_distances(self.My) # for distances between points in manifold
        
    def shadow_manifold(self):
        
        """
        Given
            manifold_pattern: a list of 2 lists that contains the lag vectors for X and Y shadow manifolds.
                For example, manifold_pattern=[[0, -1, -2],[0]], tau=2 is the same as the shadow manifold 
                (X_t, X_{t-1*2}, X_{t-2*2}, Y_t).
                A CCM of E=2, tau=2 can be represented as [[],[0, -1]], tau=2 or (Y_t, Y_{t-2}).
                Note that the lag vectors must have 1-decrement. Tau will handle the time delta.
            tau: lag step
            L: max time step to consider - 1 (starts from 0)
        Returns
            {t:[X_t, X_{t-1*2}, X_{t-2*2}, Y_t]} = Shadow attractor manifold, dictionary of vectors
        """
    
        M = {t:[] for t in range((self.E-1) * self.tau, self.L)} # shadow manifold
        for t in range((self.E-1) * self.tau, self.L):
            x_lag = [] # lagged values of X
            y_lag = [] # lagged values of Y
            
            for tx in self.manifold_pattern[0]: # get lags, we add 1 to E-1 because we want to include E
                x_lag.append(self.X[t+tx*self.tau])
                
            for ty in self.manifold_pattern[1]: # get lags, we add 1 to E-1 because we want to include E
                x_lag.append(self.Y[t+ty*self.tau])
            M[t] = x_lag + y_lag
        
    
        return M