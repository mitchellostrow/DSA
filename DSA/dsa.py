from DSA.dmd import DMD
from DSA.simdist import SimilarityTransformDist
from typing import Literal
import torch
import numpy as np

class DSA:
    """
    Computes the Dynamical Similarity Analysis (DSA) for two data matrices
    """
    def __init__(self,
                X,
                Y=None,
                n_delays=1,
                delay_interval=1,
                rank=None,
                rank_thresh=None,
                rank_explained_variance = None,
                lamb = 0.0,
                iters = 200,
                score_method: Literal["angular", "euclidean"] = "angular",
                lr = 0.01,
                device = 'cpu',
                verbose = False):
        """
        Parameters
        __________

        X : np.array or torch.tensor or list of np.arrays or torch.tensors
            first data matrix/matrices
        
        Y : None or np.array or torch.tensor or list of np.arrays or torch.tensors
            second data matrix/matrices. 
            * If Y is None, X is compared to itself pairwise 
            (must be a list)
            * If Y is a single matrix, all matrices in X are compared to Y
            * If Y is a list, all matrices in X are compared to all matrices in Y
        
        n_delays : int
            number of delays to use in constructing the Hankel matrix
        
        delay_interval : int
            interval between samples taken in constructing Hankel matrix

        rank : int
            rank of DMD matrix fit in reduced-rank regression
        
        rank_thresh : float
            Parameter that controls the rank of V in fitting HAVOK DMD by dictating a threshold
            of singular values to use. Explicitly, the rank of V will be the number of singular
            values greater than rank_thresh. Defaults to None.
        
        rank_explained_variance : float
            Parameter that controls the rank of V in fitting HAVOK DMD by indicating the percentage of
            cumulative explained variance that should be explained by the columns of V. Defaults to None.
        
        lamb : float
            L-1 regularization parameter in DMD fit
        
        iters : int
            number of optimization iterations in Procrustes over vector fields
        
        score_method : {'angular','euclidean'}
            type of metric to compute, angular vs euclidean distance
        
        lr : float
            learning rate of the Procrustes over vector fields optimization
        
        device : 'cpu' or 'cuda' or int
            hardware to use in both DMD and PoVF
        
        verbose : bool
            whether or not print when sections of the analysis is completed
        """
        self.X = X
        self.Y = Y
        if self.X is None and isinstance(self.Y,list):
            self.X, self.Y = self.Y, self.X #swap so code is easy

        self.check_method()
        if self.method == 'self-pairwise':
            self.data = [self.X]
        else:
            self.data = [self.X, self.Y]

        self.n_delays = n_delays
        self.delay_interval = delay_interval
        self.rank = rank
        self.rank_thresh = rank_thresh
        self.rank_explained_variance = rank_explained_variance
        self.lamb = lamb
        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.device = device
        self.verbose = verbose

        #get a list of all DMDs here
        self.dmds = [[DMD(Xi,n_delays,delay_interval,rank,rank_thresh,rank_explained_variance,lamb,device,verbose) for Xi in dat] for dat in self.data]
        
        self.simdist = SimilarityTransformDist(iters,score_method,lr,device,verbose)

    def check_method(self):
        tensor_or_np = lambda x: isinstance(x,np.ndarray) or isinstance(x,torch.Tensor)

        if isinstance(self.X,list):
            if self.Y is None:
                self.method = 'self-pairwise'
            elif isinstance(self.Y,list):
                self.method = 'bipartite-pairwise'
            elif tensor_or_np(self.Y):
                self.method = 'list-to-one'
                self.Y = [self.Y] #wrap in a list for iteration
            else:
                raise ValueError('unknown type of Y')
        elif tensor_or_np(self.X):
            self.X = [self.X]
            if self.Y is None:
                raise ValueError('only one element provided')
            elif isinstance(self.Y,list):
                self.method = 'one-to-list'
            elif tensor_or_np(self.Y):
                self.method = 'default'
                self.Y = [self.Y]
            else:
                raise ValueError('unknown type of Y')
        else:
            raise ValueError('unknown type of X')
        
    def fit_dmds(self,
                 X=None,
                 Y=None,
                 n_delays=None,
                 delay_interval=None,
                 rank=None,
                 lamb = None):
        """
        Recomputes only the DMDs. This will not compare, that will need to be done with the full procedure
        """
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        n_delays = self.n_delays if n_delays is None else n_delays
        delay_interval = self.delay_interval if delay_interval is None else delay_interval
        rank = self.rank if rank is None else rank
        lamb = self.lamb if lamb is None else lamb
        data = []
        if isinstance(X,list):
            data.append(X)
        else:
            data.append([X])
        if Y is not None:
            if isinstance(Y,list):
                data.append(Y)
            else:
                data.append([Y])
    
        dmds = [[DMD(Xi,n_delays,delay_interval,rank,lamb=lamb,device=self.device) for Xi in dat] for dat in data]
            
        for dmd_sets in dmds:
            for dmd in dmd_sets:
                dmd.fit()

        return dmds

    def fit_score(self):
        """
        Standard fitting function for both DMDs and PoVF
        
        Parameters
        __________

        Returns
        _______

        sims : np.array
            data matrix of the similarity scores between the specific sets of data     
        """
        for dmd_sets in self.dmds:
            for dmd in dmd_sets:
                dmd.fit()

        return self.score()
    
    def score(self,iters=None,lr=None,score_method=None):
        """
        Rescore DSA with precomputed dmds if you want to try again

        Parameters
        __________
        iters : int or None
            number of optimization steps, if None then resorts to saved self.iters
        lr : float or None
            learning rate, if None then resorts to saved self.lr
        score_method : None or {'angular','euclidean'}
            overwrites the score method in the object for this application

        Returns
        ________
        score : float
            similarity score of the two precomputed DMDs
        """

        iters = self.iters if iters is None else iters
        lr = self.lr if lr is None else lr
        score_method = self.score_method if score_method is None else score_method

        ind2 = 1 - int(self.method == 'self-pairwise') 
        # 0 if self.pairwise (want to compare the set to itself)

        sims = np.zeros((len(self.dmds[0]),len(self.dmds[ind2])))
        for i,dmd1 in enumerate(self.dmds[0]):
            for j,dmd2 in enumerate(self.dmds[ind2]):
                if self.method == 'self-pairwise':
                    if i == j: 
                        continue
                    if j > i:
                        continue
                    sims[i,j] = sims[j,i] = self.simdist.fit_score(dmd1.A_v,dmd2.A_v,iters,lr,score_method)
                else:
                    sims[i,j] = self.simdist.fit_score(dmd1.A_v,dmd2.A_v,iters,lr,score_method)

        return sims

