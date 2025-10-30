from DSA.dmd import DMD as DefaultDMD
from DSA.simdist_controllability import ControllabilitySimilarityTransformDist
from DSA.dmdc import DMDc as DefaultDMDc
from DSA.subspace_dmdc import SubspaceDMDc
from DSA.simdist import SimilarityTransformDist
from typing import Literal
import torch
import numpy as np
from omegaconf.listconfig import ListConfig
import tqdm
from joblib import Parallel, delayed
from dataclasses import dataclass, is_dataclass, asdict
import DSA.pykoopman as pykoopman
import pydmd
from DSA.pykoopman.regression import DMDc, EDMDc
from typing import Union, Mapping, Any, ClassVar, Final
import warnings


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


# ___Example config dataclasses for DMD #
@dataclass()
class DefaultDMDConfig:
    """
    Configuration dataclass for DefaultDMD (standard DMD without control).

    This configuration is used to set parameters for the DefaultDMD class when
    performing Dynamical Mode Decomposition on time series data.

    Attributes:
        n_delays (int): Number of time delays to use in the Hankel matrix construction.
            Default is 1 (no delays).
        delay_interval (int): Interval between delays in the Hankel matrix.
            Default is 1 (consecutive time steps).
        rank (int): Rank for SVD truncation. If None, no truncation is performed.
            Default is None.
        lamb (float): Regularization parameter for ridge regression.
            Default is 0 (no regularization).
        send_to_cpu (bool): Whether to move computations to CPU.
            Default is False (use GPU if available).
    """

    n_delays: int = 1
    delay_interval: int = 1
    rank: int = None
    lamb: float = 0
    send_to_cpu: bool = False


@dataclass()
class pyKoopmanDMDConfig:
    """
    Configuration dataclass for pyKoopman DMD models.

    This configuration is used to set up pyKoopman observables and regressors
    for performing DMD analysis with the pyKoopman library.

    Attributes:
        observables: Observable function from pykoopman. Default is TimeDelay with n_delays=1.
        regressor: Regressor model from pydmd. Default is DMD with svd_rank=2.
    """

    observables = pykoopman.observables.TimeDelay(n_delays=1)
    regressor = pydmd.DMD(svd_rank=2)


@dataclass()
class SubspaceDMDcConfig:
    """
    Configuration dataclass for SubspaceDMDc (DMD with control using subspace identification).

    This configuration is used to set parameters for the SubspaceDMDc class when
    performing Dynamical Mode Decomposition on controlled systems.

    Attributes:
        n_delays (int): Number of time delays to use in the Hankel matrix construction.
            Default is 1 (no delays).
        delay_interval (int): Interval between delays in the Hankel matrix.
            Default is 1 (consecutive time steps).
        rank (int): Rank for SVD truncation. If None, no truncation is performed.
            Default is None.
        lamb (float): Regularization parameter for ridge regression.
            Default is 0 (no regularization).
        backend (str): Subspace identification backend to use.
            Options: 'n4sid', 'custom'.
    """

    n_delays: int = 1
    rank: int = None
    lamb: float = 0
    backend: str = "n4sid"


# __Example config dataclasses for similarity transform distance #
@dataclass
class SimilarityTransformDistConfig:
    """
    Configuration dataclass for SimilarityTransformDist (standard similarity transform distance).

    This configuration is used to compute the similarity transform distance between
    two DMD matrices, which measures how similar two dynamical systems are.

    Attributes:
        iters (int): Number of optimization iterations for finding the similarity transform.
            Default is 1500.
        score_method (Literal["angular", "euclidean","wasserstein"]): Method for computing the distance score.
            'angular' uses angular distance, 'euclidean' uses Euclidean distance.
            Default is "angular".
        lr (float): Learning rate for the optimization algorithm.
            Default is 5e-3.
    """

    iters: int = 1500
    score_method: Literal["angular", "euclidean", "wasserstein"] = "angular"
    lr: float = 5e-3
    #class variable, set as final to indicate that it's fixed and immutable
    compare: ClassVar[Final] = "state" 

@dataclass()
class ControllabilitySimilarityTransformDistConfig:
    """
    Configuration dataclass for ControllabilitySimilarityTransformDist (similarity transform distance with control).

    This configuration is used to compute the similarity transform distance between
    two controlled DMD systems, comparing both state and control operators.

    Attributes:
        score_method (Literal["euclidean", "angular"]): Method for computing the distance score.
            'angular' uses angular distance, 'euclidean' uses Euclidean distance.
            Default is "euclidean".
        compare (str): What to compare between systems.
            'control' compares only control operators,
            'joint' compares both control and state operators simultaneousl. 
            Default is 'joint'.
            If you pass in 'state', it will throw an error -> use SimilarityTransformDistConfig Instead
        align_inputs (bool): whether to learn a C_u transformation that aligns the input representations as well
        return_distance_components (bool): Whether to return individual distance components
            (state, control, joint) separately. Default is False.
    """

    score_method: Literal["euclidean", "angular"] = "euclidean"
    compare: Literal["joint","control"] = "joint"
    align_inputs: bool = False
    return_distance_components: bool = False

class GeneralizedDSA:
    """
    Computes the Generalized Dynamical Similarity Analysis (DSA) for two data tensors.

    This class performs Dynamical Mode Decomposition (DMD) on input data and then computes
    similarity scores between the resulting DMD models using similarity transform distances.
    It supports various comparison modes including pairwise comparisons, bipartite comparisons,
    and comparisons with control inputs.

    The class handles:
    - Multiple data formats (single arrays, lists of arrays)
    - Different DMD implementations (local DMD, pyKoopman, etc.)
    - Control inputs for controlled systems
    - Parallel processing for efficiency
    - Various similarity metrics

    Example usage:
        # Compare two datasets
        dsa = GeneralizedDSA(X=data1, Y=data2, dmd_class=DefaultDMD)
        similarity_score = dsa.fit_score()

        # Pairwise comparison of multiple datasets
        dsa = GeneralizedDSA(X=[data1, data2, data3], Y=None)
        similarity_matrix = dsa.fit_score()
    """

    def __init__(
        self,
        X,
        Y=None,
        X_control=None,
        Y_control=None,
        dmd_class=DefaultDMD,
        similarity_class=SimilarityTransformDist,
        dmd_config: Union[Mapping[str, Any], dataclass] = DefaultDMDConfig(),
        simdist_config: Union[
            Mapping[str, Any], dataclass
        ] = SimilarityTransformDistConfig(),
        device="cpu",
        verbose=False,
        n_jobs=1,
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

        X_control : None or np.array or torch.tensor or list of np.arrays or torch.tensors
            control data matrix/matrices.
            Must be the same shape as X.
            If None, then no control data is used.

        Y_control : None or np.array or torch.tensor or list of np.arrays or torch.tensors
            control data matrix/matrices.
            Must be the same shape as Y.
            If None, then no control data is used.

        dmd_class : class
            DMD class to use for decomposition. Default is DefaultDMD.

        similarity_class : class
            Similarity transform distance class to use. Default is SimilarityTransformDist.

        dmd_config : Union[Mapping[str, Any], dataclass]
            Configuration for DMD parameters. Can be a dict or dataclass.

        simdist_config : Union[Mapping[str, Any], dataclass]
            Configuration for similarity transform distance parameters. Can be a dict or dataclass.

        device : str
            Device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.

        verbose : bool
            Whether to print verbose output during computation. Default is False.

        n_jobs : int
            Number of parallel jobs to use. Default is 1 (sequential).
            Set to -1 to use all available cores.

        NOTE: for all of these above, they can be single values or lists or tuples,
            depending on the corresponding dimensions of the data
            If at least one of X and Y are lists, then if they are a single value
                it will default to the rank of all DMD matrices.
            If they are (int,int), then they will correspond to an individual dmd matrix
                OR to X and Y respectively across all matrices
            If it is (list,list), then each element will correspond to an individual
                dmd matrix indexed at the same position

        """
        self.X = X
        self.Y = Y
        self.X_control = X_control
        self.Y_control = Y_control
        self.simdist_config = simdist_config

        if is_dataclass(simdist_config):
            self.simdist_config = asdict(self.simdist_config)

        self.device = device
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.dmd_class = dmd_class

        if self.X is None and isinstance(self.Y, list):
            self.X, self.Y = self.Y, self.X  # swap so code is easy
            self.X_control, self.Y_control = (
                self.Y_control,
                self.X_control,
            )  # swap control too

        self.check_method()
        if self.method == "self-pairwise":
            self.data = [self.X]
            self.control_data = [self.X_control]
        else:
            self.data = [self.X, self.Y]
            self.control_data = [self.X_control, self.Y_control]

        # Process DMD keyword arguments from **dmd_kwargs
        # These are parameters like n_delays, rank, etc., that are specific to DMDs
        # and need to be broadcasted according to X and Y data structure.
        if is_dataclass(dmd_config):
            dmd_config = asdict(dmd_config)
        self.dmd_config = (
            {}
        )  # This will store {'param_name': broadcasted_value_list_of_lists}

        for key, value in dmd_config.items():
            cast_type = CAST_TYPES.get(key)

            if cast_type is not None:
                broadcasted_value = self.broadcast_params(value, cast=cast_type)
            else:
                broadcasted_value = self.broadcast_params(value)

            setattr(self, key, broadcasted_value)  # e.g., self.n_delays = [[v,v],[v,v]]
            self.dmd_config[key] = (
                broadcasted_value  # Store in dict for DMD instantiation
            )

        self._check_dmd_simdist_compatibility(dmd_class, similarity_class)
        self._dmd_api_source(dmd_class)
        self._initiate_dmds()
        self.simdist = similarity_class(**self.simdist_config)

    def _initiate_dmds(self):
        if self.dmd_has_control and self.X_control is None and self.Y_control is None:
            raise ValueError(
                "Error: You are using a DMD model that fits a control operator but no control data is provided for either X or Y"
            )

        if not self.dmd_has_control and (
            self.X_control is not None or self.Y_control is not None
        ):
            warnings.warn(
                "You are using a DMD model with no control but control data is provided, will be ignored"
            )

        if self.dmd_api_source == "local_dmd":
            self.dmds = []
            # TODO: test this for single numpy array
            for i, (dat, control_dat) in enumerate(zip(self.data, self.control_data)):
                dmd_list = []
                if control_dat is None:
                    control_dat = [None] * len(dat)
                for j, (Xi, Xi_control) in enumerate(zip(dat, control_dat)):
                    config = {k: v[i][j] for k, v in self.dmd_config.items()}

                    #
                    if self.dmd_has_control:
                        dmd_obj = self.dmd_class(Xi, control_data=Xi_control, **config)
                    else:
                        dmd_obj = self.dmd_class(Xi, **config)

                    dmd_list.append(dmd_obj)
                self.dmds.append(dmd_list)
        else:
            self.dmds = [
                [
                    self.dmd_class(**{k: v[i][j] for k, v in self.dmd_config.items()})
                    for j, Xi in enumerate(dat)
                ]
                for i, dat in enumerate(self.data)
            ]

    def _check_dmd_simdist_compatibility(self, dmd_class, similarity_class):
        self.dmd_has_control = dmd_class in [DefaultDMDc, SubspaceDMDc] or (
            "pykoopman" in dmd_class.__module__
            and self.dmd_config.get("regressor") in [DMDc, EDMDc]
        )
        self.simdist_has_control = similarity_class in [
            ControllabilitySimilarityTransformDist
        ]

        if self.dmd_has_control and not self.simdist_has_control:
            warnings.warn(
                "Warning: You are using a DMD model that fits a control operator but comparing with a DSA metric that does not compare control operators"
            )
        if not self.dmd_has_control and self.simdist_has_control:
            raise ValueError(
                "Error: Your DMD model does not fit a control operator but comparing with a DSA metric that compares control operators"
            )

    def _dmd_api_source(self, dmd_class):
        module_name = dmd_class.__module__

        if "pydmd" in module_name:
            self.dmd_api_source = "pydmd"
            raise ValueError(
                "DSA is not currently directly compatible with pydmd due to \
                 data structure incompatibility. Please use pykoopman instead. \
                 Note that you can pass in pydmd objects through pykoopman's Koopman class."
            )
        elif "pykoopman" in module_name:
            self.dmd_api_source = "pykoopman"
            if self.dmd_has_control and self.dmd_config.get("regressor") in [
                DMDc,
                EDMDc,
            ]:
                raise ValueError(
                    "Pykoopman DMDc and EDMDc are not currently compatible with DSA"
                )
        elif (
            "DSA.dmd" in module_name
            or "DSA.subspace_dmdc" in module_name
            or "DSA.dmdc" in module_name
        ):
            self.dmd_api_source = "local_dmd"
        else:
            self.dmd_api_source = "unknown"
            raise ValueError(
                f"dmd_class {dmd_class.__name__} from unknown module {module_name}"
            )

    def fit_dmds(self):
        if self.n_jobs != 1:
            n_jobs = (
                self.n_jobs if self.n_jobs > 0 else -1
            )  # -1 means use all available cores

            if self.dmd_api_source == "local_dmd":
                for dmd_sets in self.dmds:
                    if self.verbose:
                        print(
                            f"Fitting {len(dmd_sets)} DMDs in parallel with {n_jobs} jobs"
                        )
                    Parallel(n_jobs=n_jobs)(
                        delayed(lambda dmd: dmd.fit())(dmd) for dmd in dmd_sets
                    )
            else:
                for dmd_list, dat in zip(self.dmds, self.data):
                    if self.verbose:
                        print(
                            f"Fitting {len(dmd_list)} DMDs in parallel with {n_jobs} jobs"
                        )
                    Parallel(n_jobs=n_jobs)(
                        delayed(lambda dmd, X: dmd.fit(X))(dmd, Xi)
                        for dmd, Xi in zip(dmd_list, dat)
                    )
        else:
            # Sequential processing
            if self.dmd_api_source == "local_dmd":
                for dmd_sets in self.dmds:
                    loop = (
                        dmd_sets
                        if not self.verbose
                        else tqdm.tqdm(dmd_sets, desc="Fitting DMDs")
                    )
                    for dmd in loop:
                        dmd.fit()
            else:
                for dmd_list, dat in zip(self.dmds, self.data):
                    loop = (
                        zip(dmd_list, dat)
                        if not self.verbose
                        else tqdm.tqdm(zip(dmd_list, dat), desc="Fitting DMDs")
                    )
                    for dmd, Xi in loop:
                        dmd.fit(Xi)

    def check_method(self):
        """
        helper function to identify what type of dsa we're running
        """
        tensor_or_np = lambda x: isinstance(x, (np.ndarray, torch.Tensor))

        if isinstance(self.X, list):
            # Ensure X_control is also a list
            if self.X_control is not None and not isinstance(self.X_control, list):
                self.X_control = [self.X_control]

            if self.Y is None:
                self.method = "self-pairwise"
            elif isinstance(self.Y, list):
                self.method = "bipartite-pairwise"
                if self.Y_control is not None and not isinstance(self.Y_control, list):
                    self.Y_control = [self.Y_control]
            elif tensor_or_np(self.Y):
                self.method = "list-to-one"
                self.Y = [self.Y]  # wrap in a list for iteration
                if self.Y_control is not None:
                    self.Y_control = [self.Y_control]
            else:
                raise ValueError("unknown type of Y")
        elif tensor_or_np(self.X):
            self.X = [self.X]
            if self.X_control is not None:
                self.X_control = [self.X_control]
            if self.Y is None:
                raise ValueError("only one element provided")
            elif isinstance(self.Y, list):
                self.method = "one-to-list"
                if self.Y_control is not None and not isinstance(self.Y_control, list):
                    self.Y_control = [self.Y_control]
            elif tensor_or_np(self.Y):
                self.method = "default"
                self.Y = [self.Y]
                if self.Y_control is not None:
                    self.Y_control = [self.Y_control]
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
            isinstance(param, (int, float, np.integer,str))
            or param in {None, "None", "none"}
            or (
                hasattr(param, "__module__")
                and ("pykoopman" in param.__module__ or "pydmd" in param.__module__)
            )
        ):  # self.X has already been mapped to [self.X]
            if param in {"None", "none"}:
                param = None
            out.append([param] * len(self.X))
            if self.Y is not None:
                out.append([param] * len(self.Y))
        else:
            raise ValueError("unknown type entered for parameter")

        if cast is not None and param is not None:
            out = [[cast(x) if x is not None else None for x in dat] for dat in out]

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
        if self.dmd_api_source == "local_dmd":
            return dmd.A_v
        elif self.dmd_api_source == "pykoopman":
            return dmd.A
        elif self.dmd_api_source == "pydmd":
            raise ValueError(
                "DSA is not currently compatible with pydmd due to \
                data structure incompatibility. Please use pykoopman instead."
            )

    def get_dmd_control_matrix(self, dmd):
        if self.dmd_api_source == "local_dmd":
            return dmd.B_v
        elif self.dmd_api_source == "pykoopman":
            return dmd.B
        elif self.dmd_api_source == "pydmd":
            raise ValueError(
                "DSA is not currently compatible with pydmd due to \
                data structure incompatibility. Please use pykoopman instead."
            )

    def score(self):
        """
        Score DSA with precomputed dmds
        Parameters
        __________

        Returns
        ________
        score : float or numpy array
            similarity score of the two precomputed DMDs
            if array is d x d, it is a standard DSA (or whatever you set it to be)
            if array is d x d x 3 , you ran simdist_controllability with return_distance_components = True
                This means that you returned the following similarity scores:
                    joint similarity scores (both state and control)
                    state similarity score (optimized jointly)
                    control similarity score (optimized jointly)
        """

        ind2 = 0 if self.method == "self-pairwise" else 1
        # 0 if self.pairwise (want to compare the set to itself)
        n_sims = (
            3
            if (
                self.simdist_has_control
                and self.simdist_config.get("return_distance_components")
                and self.simdist_config.get("compare") == "joint"
            )
            else 1
        )

        self.sims = np.zeros((len(self.dmds[0]), len(self.dmds[ind2]), n_sims))


        def compute_similarity(i, j):
            if self.method == "self-pairwise" and j >= i:
                return None

            if self.verbose and self.n_jobs != 1:
                print(f"computing similarity between DMDs {i} and {j}")

            simdist_args = [
                self.get_dmd_matrix(self.dmds[0][i]),
                self.get_dmd_matrix(self.dmds[ind2][j]),
            ]

            if self.simdist_has_control and self.dmd_has_control:
                simdist_args.extend(
                    [
                        self.get_dmd_control_matrix(self.dmds[0][i]),
                        self.get_dmd_control_matrix(self.dmds[ind2][j]),
                    ]
                )
            sim = self.simdist.fit_score(*simdist_args)

            if self.verbose and self.n_jobs != 1:
                print(f"computing similarity between DMDs {i} and {j}")

            return (i, j, sim)

        pairs = []
        for i in range(len(self.dmds[0])):
            for j in range(len(self.dmds[ind2])):
                if not (self.method == "self-pairwise" and j >= i):
                    pairs.append((i, j))

        if self.n_jobs != 1:
            n_jobs = self.n_jobs if self.n_jobs > 0 else -1
            if self.verbose:
                print(
                    f"Computing {len(pairs)} DMD similarities in parallel with {n_jobs} jobs"
                )

            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_similarity)(i, j) for i, j in pairs
            )
        else:
            loop = (
                pairs
                if not self.verbose
                else tqdm.tqdm(pairs, desc="Computing DMD similarities")
            )
            results = [compute_similarity(i, j) for i, j in loop]

        for result in results:
            if result is not None:
                i, j, sim = result
                self.sims[i, j] = sim
                if self.method == "self-pairwise":
                    self.sims[j, i] = sim

        if self.method == "default":
            return self.sims[0, 0].squeeze()

        return self.sims.squeeze()


class DSA(GeneralizedDSA):
    def __init__(
        self,
        X,
        Y=None,
        dmd_class=DefaultDMD,
        device="cpu",
        verbose=False,
        n_jobs=1,
        # simdist parameters
        score_method: Literal["angular", "euclidean","wasserstein"] = "angular",
        iters: int = 1500,
        lr: float = 5e-3,
        **dmd_kwargs,
    ):
        # TODO: add readme
        simdist_config = {
            "score_method": score_method,
            "iters": iters,
            "lr": lr,
        }

        dmd_config = dmd_kwargs

        super().__init__(
            X=X,
            Y=Y,
            X_control=None,
            Y_control=None,
            dmd_class=dmd_class,
            similarity_class=SimilarityTransformDist,
            dmd_config=dmd_config,
            simdist_config=simdist_config,
            device=device,
            verbose=verbose,
            n_jobs=n_jobs,
        )


class InputDSA(GeneralizedDSA):
    def __init__(
        self,
        X,
        X_control,
        Y=None,
        Y_control=None,
        dmd_class=SubspaceDMDc,
        dmd_config: Union[Mapping[str, Any], dataclass] = SubspaceDMDcConfig(),
        simdist_config: Union[
            Mapping[str, Any], dataclass
        ] = ControllabilitySimilarityTransformDistConfig(),
        device="cpu",
        verbose=False,
        n_jobs=1,
    ):
        #TODO: fix based on making compare argument explicit
        # check if simdist_config has 'compare', and if it's 'state', use the standard SimilarityTransformDist,
        # otherwise use ControllabilitySimilarityTransformDistConfig
        if isinstance(simdist_config, dict):
            compare = simdist_config.get("compare", None)
        else:
            compare = simdist_config.compare
        simdist = self.update_compare_method(compare)

        super().__init__(
            X,
            Y,
            X_control,
            Y_control,
            dmd_class,
            simdist,
            dmd_config,
            simdist_config,
            device,
            verbose,
            n_jobs,
        )

        assert X_control is not None
        assert self.dmd_has_control

    def update_compare_method(self,compare='joint'):
        if compare == "state":
            simdist = SimilarityTransformDist
            #TODO: check simdist config to make sure it aligns
        else:
            simdist = ControllabilitySimilarityTransformDist
            #TODO: check simdist config to make sure it aligns
        return simdist

