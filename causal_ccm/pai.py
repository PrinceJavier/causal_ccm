import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import pearsonr

class ccm:
    """
    We're checking causality X -> Y       
    Args
        X: timeseries for variable X that could cause Y
        Y: timeseries for variable Y that could be caused by X
        tau: time lag. default = 1
        E: shadow manifold embedding dimension. default = 2
        L: time period/duration to consider (longer = more data). default = length of X
    """
    def __init__(self, X, Y, tau=1, E=2, L=None):
        '''
        X: timeseries for variable X that could cause Y
        Y: timeseries for variable Y that could be caused by X
        tau: time lag
        E: shadow manifold embedding dimension
        L: time period/duration to consider (longer = more data)
        We're checking for X -> Y
        '''
        self.X = X
        self.Y = Y
        self.tau = tau
        self.E = E
        if L == None:
            self.L = len(X)
        else:
            self.L = L
        self.My = self.shadow_manifold(Y) # shadow manifold for Y (we want to know if info from X is in Y)
        self.t_steps, self.dists = self.get_distances(self.My) # for distances between points in manifold

    def shadow_manifold(self, V):
        """
        Given
            V: some time series vector
            tau: lag step
            E: shadow manifold embedding dimension
            L: max time step to consider - 1 (starts from 0)
        Returns
            {t:[t, t-tau, t-2*tau ... t-(E-1)*tau]} = Shadow attractor manifold, dictionary of vectors
        """
        V = V[:self.L] # make sure we cut at L
        M = {t:[] for t in range((self.E-1) * self.tau, self.L)} # shadow manifold
        for t in range((self.E-1) * self.tau, self.L):
            v_lag = [] # lagged values
            for t2 in range(0, self.E-1 + 1): # get lags, we add 1 to E-1 because we want to include E
                v_lag.append(V[t-t2*self.tau])
            M[t] = v_lag
        return M

    # get pairwise distances between vectors in the time series
    def get_distances(self, M):
        """
        Args
            M: The shadow manifold from the time series
        Returns
            t_steps: timesteps
            dists: n x n matrix showing distances of each vector at t_step (rows) from other vectors (columns)
        """

        # we extract the time indices and vectors from the manifold M
        # we just want to be safe and convert the dictionary to a tuple (time, vector)
        # to preserve the time inds when we separate them
        t_vec = [(k, v) for k,v in M.items()]
        t_steps = np.array([i[0] for i in t_vec])
        vecs = np.array([i[1] for i in t_vec])
        dists = distance.cdist(vecs, vecs)
        return t_steps, dists

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
        nearest_inds = np.argsort(dist_t)[1:self.E+1 + 1] # get indices sorted, we exclude 0 which is distance from itself
        nearest_timesteps = t_steps[nearest_inds] # index column-wise, t_steps are same column and row-wise
        nearest_distances = dist_t[nearest_inds]

        return nearest_timesteps, nearest_distances

    def predict(self, t):
        """
        Args
            t: timestep at manifold of y, My, to predict X at same time step
        Returns
            X_true: the true value of X at time t
            X_hat: the predicted value of X at time t using the manifold My
        """
        eps = 0.000001 # epsilon minimum distance possible
        t_ind = np.where(self.t_steps == t) # get the index of time t
        dist_t = self.dists[t_ind].squeeze() # distances from vector at time t (this is one row)
        nearest_timesteps, nearest_distances = self.get_nearest_distances(t, self.t_steps, self.dists)

        # get weights
        u = np.exp(-nearest_distances/np.max([eps, nearest_distances[0]])) # we divide by the closest distance to scale
        w = u / np.sum(u)

        # get prediction of X
        X_true = self.X[t] # get corresponding true X
        X_cor = np.array(self.X)[nearest_timesteps] # get corresponding Y to cluster in Mx
        X_hat = (w * X_cor).sum() # get X_hat

    #     DEBUGGING
    #     will need to check why nearest_distances become nan
    #     if np.isnan(X_hat):
    #         print(nearest_timesteps)
    #         print(nearest_distances)

        return X_true, X_hat


    def causality(self):
        '''
        Args:
            None
        Returns:
            (r, p): how much X causes Y. as a correlation between predicted X and true X and the p-value (significance)
        '''

        # run over all timesteps in M
        # X causes Y, we can predict X using My
        # X puts some info into Y that we can use to reverse engineer X from Y via My
        X_true_list = []
        X_hat_list = []

        for t in list(self.My.keys()): # for each time step in My
            X_true, X_hat = self.predict(t) # predict X from My
            X_true_list.append(X_true)
            X_hat_list.append(X_hat)

        x, y = X_true_list, X_hat_list
        r, p = pearsonr(x, y)

        return r, p

    def visualize_cross_mapping(self):
        """
        Visualize the shadow manifolds and some cross mappings
        """
        # we want to check cross mapping from Mx to My and My to Mx

        f, axs = plt.subplots(1, 2, figsize=(12, 6))

        for i, ax in zip((0, 1), axs): # i will be used in switching Mx and My in Cross Mapping visualization
            #===============================================
            # Shadow Manifolds Visualization

            X_lag, Y_lag = [], []
            for t in range(1, len(self.X)):
                X_lag.append(self.X[t-self.tau])
                Y_lag.append(self.Y[t-self.tau])
            X_t, Y_t = self.X[1:], self.Y[1:] # remove first value

            ax.scatter(X_t, X_lag, s=5, label='$M_x$')
            ax.scatter(Y_t, Y_lag, s=5, label='$M_y$', c='y')

            #===============================================
            # Cross Mapping Visualization

            A, B = [(self.Y, self.X), (self.X, self.Y)][i]
            cm_direction = ['Mx to My', 'My to Mx'][i]

            Ma = self.shadow_manifold(A)
            Mb = self.shadow_manifold(B)

            t_steps_A, dists_A = self.get_distances(Ma) # for distances between points in manifold
            t_steps_B, dists_B = self.get_distances(Mb) # for distances between points in manifold

            # Plot cross mapping for different time steps
            timesteps = list(Ma.keys())
            for t in np.random.choice(timesteps, size=3, replace=False):
                Ma_t = Ma[t]
                near_t_A, near_d_A = self.get_nearest_distances(t, t_steps_A, dists_A)

                for i in range(self.E+1):
                    # points on Ma
                    A_t = Ma[near_t_A[i]][0]
                    A_lag = Ma[near_t_A[i]][1]
                    ax.scatter(A_t, A_lag, c='b', marker='s')

                    # corresponding points on Mb
                    B_t = Mb[near_t_A[i]][0]
                    B_lag = Mb[near_t_A[i]][1]
                    ax.scatter(B_t, B_lag, c='r', marker='*', s=50)

                    # connections
                    ax.plot([A_t, B_t], [A_lag, B_lag], c='r', linestyle=':')

            ax.set_title(f'{cm_direction} cross mapping. time lag, tau = {self.tau}, E = 2')
            ax.legend(prop={'size': 14})

            ax.set_xlabel('$X_t$, $Y_t$', size=15)
            ax.set_ylabel('$X_{t-1}$, $Y_{t-1}$', size=15)
        plt.show()

    def plot_ccm_correls(self):
        """
        Args
            X: X time series
            Y: Y time series
            tau: time lag
            E: shadow manifold embedding dimension
            L: time duration
        Returns
            None. Just correlation plots between predicted X|M_y and true X
        """
        X_My_true, X_My_pred = [], []
        for t in range(self.tau, self.L):
            true, pred = self.predict(t)
            X_My_true.append(true)
            X_My_pred.append(pred)

        # predicting X from My
        r, p = np.round(pearsonr(X_My_true, X_My_pred), 4)

        plt.scatter(X_My_true, X_My_pred, s=10)
        plt.xlabel('$X(t)$ (observed)', size=15)
        plt.ylabel('$\hat{X}(t)|M_y$ (estimated)', size=15)
        plt.title(f'tau={self.tau}, E={self.E}, L={self.L}, Correlation coeff = {r}')

        plt.show()
