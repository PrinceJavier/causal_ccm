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
        Ey: shadow manifold embedding dimension of Y, derived from manifold_pattern[1], Ey should be >= E
        manifold_pattern: a list of 2 lists that contains the lag vectors for X and Y shadow manifolds.
            For example, manifold_pattern=[[0, -1, -2],[0]], tau=2 is the same as the shadow manifold 
            (X_t, X_{t-1*2}, X_{t-2*2}, Y_t).
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
            
        self.My = self.shadow_manifold(self.X, self.Y) # shadow manifold for Y (we want to know if info from X is in Y)
        self.t_steps, self.dists = self.get_distances(self.My) # for distances between points in manifold
        
    def shadow_manifold(self, X, Y):
        
        """
        Given
            X: X time series
            Y: Y time series
            manifold_pattern: a list of 2 lists that contains the lag vectors for X and Y shadow manifolds.
                For example, manifold_pattern=[[0, -1, -2],[0]], tau=2 is the same as the shadow manifold 
                (X_t, X_{t-1*2}, X_{t-2*2}, Y_t).
                Note that the lag vectors must have 1-decrement. Tau will handle the time delta.
            tau: lag step
            L: max time step to consider - 1 (starts from 0)
        Returns
            {t:[X_t, X_{t-1*2}, X_{t-2*2}, Y_t]} = Shadow attractor manifold following the manifold_pattern, dictionary of vectors
        """
    
        M = {t:[] for t in range((self.E-1) * self.tau, self.L)} # shadow manifold
        for t in range((self.E-1) * self.tau, self.L):
            x_lag = [] # lagged values of X
            y_lag = [] # lagged values of Y
            
            for tx in self.manifold_pattern[0]: # get lags, we add 1 to E-1 because we want to include E
                x_lag.append(X[t+tx*self.tau])
                
            for ty in self.manifold_pattern[1]: # get lags, we add 1 to E-1 because we want to include E
                x_lag.append(Y[t+ty*self.tau])
            M[t] = x_lag + y_lag
        
    
        return M
    
    def get_nearest_distances(self, t, t_steps, dists):
        """
        Args:
            t: timestep of vector whose nearest neighbors we want to compute
            t_teps: time steps of all vectors in the manifold M, output of get_distances()
            dists: distance matrix showing distance of each vector (row) from other vectors (columns). output of get_distances()
            E: embedding dimension of shadow manifold M
        Returns:
            nearest_timesteps: array of timesteps of E+1 vectors that are nearest to vector at time t
            nearest_distances: array of distances corresponding to vectors closest to vector at time t
        """
        t_ind = np.where(t_steps == t) # get the index of time t
        dist_t = dists[t_ind].squeeze() # distances from vector at time t (this is one row)

        # get top closest vectors
        nearest_inds = np.argsort(dist_t)[1:self.E + self.Ey + 1 + 1] # get indices sorted, we exclude 0 which is distance from itself
        nearest_timesteps = t_steps[nearest_inds] # index column-wise, t_steps are same column and row-wise
        nearest_distances = dist_t[nearest_inds]
       
        return nearest_timesteps, nearest_distances
    
    def visualize_cross_mapping(self,size=3):
        """
        Visualize the shadow manifolds and some cross mappings
        """
        # we want to check cross mapping from Mx to My and My to Mx

        f, axs = plt.subplots(1, 2, figsize=(12, 6))

        for i, ax in zip((0, 1), axs): # i will be used in switching Mx and My in Cross Mapping visualization
            #===============================================
            # Shadow Manifolds Visualization

            Mxy = self.shadow_manifold(self.X, self.Y)
            Myx = self.shadow_manifold(self.Y, self.X)
            
            #Use PCA to visualize any E dimension manifold
            pca = PCA(2)
            Mxy_pca = pca.fit_transform(list(Mxy.values()))
            Myx_pca = pca.fit_transform(list(Myx.values()))

            ax.scatter(Mxy_pca[:,0], Mxy_pca[:,1], s=5, label='$M_{xy}$')
            ax.scatter(Myx_pca[:,0], Myx_pca[:,1], s=5, label='$M_{yx}$', c='y')

            #===============================================
            # Cross Mapping Visualization

            A, B = [(self.X, self.Y), (self.Y, self.X)][i]
            cm_direction = ['Mxy to Myx', 'Myx to Mxy'][i]

            Ma = self.shadow_manifold(A, B)
            Mb = self.shadow_manifold(B, A)
            
            #Use PCA to visualize any E dimension manifold
            pca = PCA(2)
            
            Ma_pca = pca.fit_transform(list(Ma.values()))
            Mb_pca = pca.fit_transform(list(Mb.values()))

                
            t_steps_A, dists_A = self.get_distances(Ma) # for distances between points in manifold
            t_steps_B, dists_B = self.get_distances(Mb) # for distances between points in manifold

            # Plot cross mapping for different time steps
            timesteps = list(Ma.keys())
            for t in np.random.choice(timesteps, size=size, replace=False):
                Ma_t = Ma[t]
                near_t_A, near_d_A = self.get_nearest_distances(t, t_steps_A, dists_A)

                for i in range(self.E+  self.Ey + 1):
                    # points on Ma
                    A_t = Ma_pca[near_t_A[i],0]
                    A_lag = Ma_pca[near_t_A[i],1]
                    ax.scatter(A_t, A_lag, c='b', marker='s')

                    # corresponding points on Mb
                    B_t = Mb_pca[near_t_A[i],0]
                    B_lag = Mb_pca[near_t_A[i],1]
                    ax.scatter(B_t, B_lag, c='r', marker='*', s=50)

                    # connections
                    ax.plot([A_t, B_t], [A_lag, B_lag], c='r', linestyle=':')

            ax.set_title(f'{cm_direction} cross mapping. time lag, tau = {self.tau}, E = 2')
            ax.legend(prop={'size': 14})

            ax.set_xlabel('$M_{xy}$ PCA 0, $M_{yx}$ PCA 0', size=15)
            ax.set_ylabel('$M_{xy}$ PCA 1, $M_{yx}$ PCA 1', size=15)
        plt.show()