from DSA.dmd import DMD as DefaultDMD
from DSA.simdist import SimilarityTransformDist
from typing import Literal
import torch
import numpy as np
from omegaconf.listconfig import ListConfig
import tqdm
from joblib import Parallel, delayed


CAST_TYPES = {
    "n_delays": int,
    "delay_interval": int,
    "rank": int,
    "rank_thresh": float,
    "rank_explained_variance": float,
    "lamb": float,
    "steps_ahead": int,
    "reduced_rank_reg": bool,
    "send_to_cpu": bool,
}


class DSA:
    """
    Computes the Dynamical Similarity Analysis (DSA) for two data tensors
    """

    def __init__(
        self,
        X,
        Y=None,
        dmd_class=DefaultDMD,
        iters=1500,
        score_method: Literal["angular", "euclidean", "wasserstein"] = "angular",
        lr=5e-3,
        zero_pad=False,
        device="cpu",
        wasserstein_compare: Literal["sv", "eig", None] = "eig",
        n_jobs: int = 1,
        dsa_verbose=False,
        **dmd_kwargs,
    ):
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

        DMD parameters :

        n_delays : int or list or tuple/list: (int,int), (list,list),(list,int),(int,list)
            number of delays to use in constructing the Hankel matrix

        delay_interval : int or list or tuple/list: (int,int), (list,list),(list,int),(int,list)
            interval between samples taken in constructing Hankel matrix

        rank : int or list or tuple/list: (int,int), (list,list),(list,int),(int,list)
            rank of DMD matrix fit in reduced-rank regression

        rank_thresh : float or list or tuple/list: (float,float), (list,list),(list,float),(float,list)
            Parameter that controls the rank of V in fitting HAVOK DMD by dictating a threshold
            of singular values to use. Explicitly, the rank of V will be the number of singular
            values greater than rank_thresh. Defaults to None.

        rank_explained_variance : float or list or tuple: (float,float), (list,list),(list,float),(float,list)
            Parameter that controls the rank of V in fitting HAVOK DMD by indicating the percentage of
            cumulative explained variance that should be explained by the columns of V. Defaults to None.

        lamb : float
            L-1 regularization parameter in DMD fit

        send_to_cpu: bool
            If True, will send all tensors in the object back to the cpu after everything is computed.
            This is implemented to prevent gpu memory overload when computing multiple DMDs.

        NOTE: for all of these above, they can be single values or lists or tuples,
            depending on the corresponding dimensions of the data
            If at least one of X and Y are lists, then if they are a single value
                it will default to the rank of all DMD matrices.
            If they are (int,int), then they will correspond to an individual dmd matrix
                OR to X and Y respectively across all matrices
            If it is (list,list), then each element will correspond to an individual
                dmd matrix indexed at the same position

        SimDist parameters:

        iters : int
            number of optimization iterations in Procrustes over vector fields

        score_method : {'angular','euclidean'}
            type of metric to compute, angular vs euclidean distance

        lr : float
            learning rate of the Procrustes over vector fields optimization

        zero_pad : bool
            whether or not to zero-pad if the dimensions are different

        device : 'cpu' or 'cuda' or int
            hardware to use in both DMD and PoVF

        dsa_verbose : bool
            whether or not print when sections of the analysis is completed

        wasserstein_compare : {'sv','eig',None}
            specifies whether to compare the singular values or eigenvalues
            if score_method is "wasserstein", or the shapes are different
        """
        self.X = X
        self.Y = Y
        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.device = device
        self.zero_pad = zero_pad
        self.n_jobs = n_jobs
        self.dsa_verbose = dsa_verbose
        self.dmd_class = dmd_class

        if self.X is None and isinstance(self.Y, list):
            self.X, self.Y = self.Y, self.X  # swap so code is easy

        self.check_method()
        if self.method == "self-pairwise":
            self.data = [self.X]
        else:
            self.data = [self.X, self.Y]

        # Process DMD keyword arguments from **dmd_kwargs
        # These are parameters like n_delays, rank, etc., that are specific to DMDs
        # and need to be broadcasted according to X and Y data structure.
        self.dmd_kwargs = (
            {}
        )  # This will store {'param_name': broadcasted_value_list_of_lists}

        if dmd_kwargs:
            for key, value in dmd_kwargs.items():
                cast_type = CAST_TYPES.get(key)

                if cast_type is not None:
                    broadcasted_value = self.broadcast_params(value, cast=cast_type)
                else:
                    broadcasted_value = self.broadcast_params(value)

                setattr(
                    self, key, broadcasted_value
                )  # e.g., self.n_delays = [[v,v],[v,v]]
                self.dmd_kwargs[key] = (
                    broadcasted_value  # Store in dict for DMD instantiation
                )


        self._dmd_api_source(dmd_class)
        self._initiate_dmds()

        self.simdist = SimilarityTransformDist(
            iters, score_method, lr, device, dsa_verbose, wasserstein_compare
        )

    def _initiate_dmds(self):
        if self.dmd_api_source == "local_dsa_dmd":
            self.dmds = [
                [
                    self.dmd_class(Xi, **{k: v[i][j] for k, v in self.dmd_kwargs.items()})
                    for j, Xi in enumerate(dat)
                ]
                for i, dat in enumerate(self.data)
            ]
        else:
            self.dmds = [
                [self.dmd_class(**{k: v[i][j] for k, v in self.dmd_kwargs.items()}) for j, Xi in enumerate(dat)]
                for i, dat in enumerate(self.data)
            ]

    def _dmd_api_source(self, dmd_class):
        module_name = dmd_class.__module__
        if "pydmd" in module_name:
            self.dmd_api_source = "pydmd"
            raise ValueError("DSA is not currently directly compatible with pydmd due to \
                 data structure incompatibility. Please use pykoopman instead. \
                 Note that you can pass in pydmd objects through pykoopman's Koopman class.")
        elif "pykoopman" in module_name:
            self.dmd_api_source = "pykoopman"
        elif "DSA.dmd" in module_name:
            self.dmd_api_source = "local_dsa_dmd"
        else:
            self.dmd_api_source = "unknown"
            raise ValueError(
                f"dmd_class {dmd_class.__name__} from unknown module {module_name}"
            )

    def fit_dmds(self):
        if self.n_jobs != 1:
            n_jobs = self.n_jobs if self.n_jobs > 0 else -1  # -1 means use all available cores
            
            if self.dmd_api_source == "local_dsa_dmd":
                for dmd_sets in self.dmds:
                    if self.dsa_verbose:
                        print(f"Fitting {len(dmd_sets)} DMDs in parallel with {n_jobs} jobs")
                    Parallel(n_jobs=n_jobs)(
                        delayed(lambda dmd: dmd.fit())(dmd) for dmd in dmd_sets
                    )
            else:
                for dmd_list, dat in zip(self.dmds, self.data):
                    if self.dsa_verbose:
                        print(f"Fitting {len(dmd_list)} DMDs in parallel with {n_jobs} jobs")
                    Parallel(n_jobs=n_jobs)(
                        delayed(lambda dmd, X: dmd.fit(X))(dmd, Xi) for dmd, Xi in zip(dmd_list, dat)
                    )
        else:
            # Sequential processing
            if self.dmd_api_source == "local_dsa_dmd":
                for dmd_sets in self.dmds:
                    loop = dmd_sets if not self.dsa_verbose else tqdm.tqdm(dmd_sets, desc="Fitting DMDs")
                    for dmd in loop:
                        dmd.fit()
            else:
                for dmd_list, dat in zip(self.dmds, self.data):
                    loop = zip(dmd_list, dat) if not self.dsa_verbose else tqdm.tqdm(zip(dmd_list, dat), desc="Fitting DMDs")
                    for dmd, Xi in loop:
                        dmd.fit(Xi)

    def check_method(self):
        """
        helper function to identify what type of dsa we're running
        """
        tensor_or_np = lambda x: isinstance(x, (np.ndarray, torch.Tensor))

        if isinstance(self.X, list):
            if self.Y is None:
                self.method = "self-pairwise"
            elif isinstance(self.Y, list):
                self.method = "bipartite-pairwise"
            elif tensor_or_np(self.Y):
                self.method = "list-to-one"
                self.Y = [self.Y]  # wrap in a list for iteration
            else:
                raise ValueError("unknown type of Y")
        elif tensor_or_np(self.X):
            self.X = [self.X]
            if self.Y is None:
                raise ValueError("only one element provided")
            elif isinstance(self.Y, list):
                self.method = "one-to-list"
            elif tensor_or_np(self.Y):
                self.method = "default"
                self.Y = [self.Y]
            else:
                raise ValueError("unknown type of Y")
        else:
            raise ValueError("unknown type of X")

    def broadcast_params(self, param, cast=None):
        """
        aligns the dimensionality of the parameters with the data so it's one-to-one
        """
        out = []
        if isinstance(param, (tuple, list, np.ndarray, ListConfig)):
            if self.method == "self-pairwise" and len(param) >= len(self.X):
                out = [param]
            else:
                assert len(param) <= 2  # only 2 elements max

                # if the inner terms are singly valued, we broadcast, otherwise needs to be the same dimensions
                for i, data in enumerate([self.X, self.Y]):
                    if data is None:
                        continue
                    if isinstance(param[i], (int, float)):
                        out.append([param[i]] * len(data))
                    elif isinstance(param[i], (list, np.ndarray, tuple)):
                        assert len(param[i]) >= len(data)
                        out.append(param[i][: len(data)])
        elif (
            isinstance(param, (int, float, np.integer)) or param in {None,'None','none'} or
            (hasattr(param, '__module__') and ('pykoopman' in param.__module__ or 'pydmd' in param.__module__))
        ):  # self.X has already been mapped to [self.X]
            if param in {'None','none'}:
                param = None
            out.append([param] * len(self.X))
            if self.Y is not None:
                out.append([param] * len(self.Y))
        else:
            raise ValueError("unknown type entered for parameter")

        if cast is not None and param is not None:
            out = [[cast(x) for x in dat] for dat in out]

        return out

    def fit_score(self):
        """
        Standard fitting function for both DMDs and PAVF

        Parameters
        __________

        Returns
        _______

        sims : np.array
            data matrix of the similarity scores between the specific sets of data
        """
        self.fit_dmds()
        return self.score()

    def get_dmd_matrix(self, dmd):
        if self.dmd_api_source == "local_dsa_dmd":
            return dmd.A_v
        elif self.dmd_api_source == "pykoopman":
            return dmd.A
        elif self.dmd_api_source == "pydmd":
            raise ValueError("DSA is not currently compatible with pydmd due to \
                data structure incompatibility. Please use pykoopman instead.")

    def score(self, iters=None, lr=None, score_method=None):
        """
        Score DSA with precomputed dmds 
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

        ind2 = 1 - int(self.method == "self-pairwise")
        # 0 if self.pairwise (want to compare the set to itself)

        self.sims = np.zeros((len(self.dmds[0]), len(self.dmds[ind2])))

        if self.dsa_verbose:
           print('comparing dmds')
        
        def compute_similarity(i, j):
            if self.method == "self-pairwise" and j >= i:
                return None

            if self.dsa_verbose and self.n_jobs != 1:
                print(f"computing similarity between DMDs {i} and {j}")
                
            sim = self.simdist.fit_score(
                self.get_dmd_matrix(self.dmds[0][i]),
                self.get_dmd_matrix(self.dmds[ind2][j]),
                iters,
                lr,
                score_method,
                zero_pad=self.zero_pad,
            )
            if self.dsa_verbose and self.n_jobs != 1:
                print(f"computing similarity between DMDs {i} and {j}")
            
            return (i, j, sim)
        
        pairs = []
        for i in range(len(self.dmds[0])):
            for j in range(len(self.dmds[ind2])):
                if not (self.method == "self-pairwise" and j >= i):
                    pairs.append((i, j))
        
        if self.n_jobs != 1:
            n_jobs = self.n_jobs if self.n_jobs > 0 else -1
            if self.dsa_verbose:
                print(f"Computing {len(pairs)} DMD similarities in parallel with {n_jobs} jobs")
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_similarity)(i, j) for i, j in pairs
            )
        else:
            loop = pairs if not self.dsa_verbose else tqdm.tqdm(pairs, desc="Computing DMD similarities")
            results = [compute_similarity(i, j) for i, j in loop]
        
        for result in results:
            if result is not None:
                i, j, sim = result
                self.sims[i, j] = sim
                if self.method == "self-pairwise":
                    self.sims[j, i] = sim
        
        if self.method == "default":
            return self.sims[0, 0]

        return self.sims
