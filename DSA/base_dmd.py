"""Base class for DMD implementations."""

import numpy as np
import torch
from abc import ABC, abstractmethod


class BaseDMD(ABC):
    """Base class for DMD implementations with common functionality."""
    
    def __init__(
        self,
        device="cpu",
        verbose=False,
        send_to_cpu=False,
        lamb=0,
    ):
        """
        Parameters
        ----------
        device: string, int, or torch.device
            A string, int or torch.device object to indicate the device to torch.
        verbose: bool
            If True, print statements will be provided about the progress of the fitting procedure.
        send_to_cpu: bool
            If True, will send all tensors in the object back to the cpu after everything is computed.
            This is implemented to prevent gpu memory overload when computing multiple DMDs.
        lamb : float
            Regularization parameter for ridge regression. Defaults to 0.
        """
        self.device = device
        self.verbose = verbose
        self.send_to_cpu = send_to_cpu
        self.lamb = lamb
        
        # Common attributes
        self.data = None
        self.n = None
        self.ntrials = None
        self.is_list_data = False
        
        # SVD attributes - will be set by subclasses
        self.cumulative_explained_variance = None
        
    def _process_single_dataset(self, data):
        """Process a single dataset, handling numpy arrays, tensors, and lists."""
        if isinstance(data, list):
            try:
                # Attempt to convert to a single tensor if possible (non-ragged)
                processed_data = [
                    torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                    for d in data
                ]
                return torch.stack(processed_data), False  
            except (RuntimeError, ValueError):
                # Handle ragged lists
                processed_data = [
                    torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                    for d in data
                ]
                # Check for consistent last dimension
                n_features = processed_data[0].shape[-1]
                if not all(d.shape[-1] == n_features for d in processed_data):
                    raise ValueError(
                        "All tensors in the list must have the same number of features (last dimension)."
                    )
                return processed_data, True  
                
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data), False
        
        return data, False

    def _init_single_data(self, data):
        """Initialize data attributes for a single dataset."""
        processed_data, is_ragged = self._process_single_dataset(data)
        
        if is_ragged:
            # Set attributes for ragged data
            n_features = processed_data[0].shape[-1]
            self.n = n_features
            self.ntrials = sum(
                d.shape[0] if d.ndim == 3 else 1 for d in processed_data
            )
            self.trial_counts = [
                d.shape[0] if d.ndim == 3 else 1 for d in processed_data
            ]
            self.is_list_data = True
        else:
            # Set attributes for non-ragged data
            if processed_data.ndim == 3:
                self.ntrials = processed_data.shape[0]
                self.n = processed_data.shape[2]
            else:
                self.n = processed_data.shape[1]
                self.ntrials = 1
            self.is_list_data = False
            
        return processed_data

    def _compute_explained_variance(self, S):
        """Compute cumulative explained variance from singular values."""
        exp_variance = S**2 / torch.sum(S**2)
        return torch.cumsum(exp_variance, 0)

    def _compute_rank_from_params(self, S, cumulative_explained_variance, max_rank, 
                                 rank=None, rank_thresh=None, rank_explained_variance=None):
        """
        Compute rank based on provided parameters.
        
        Parameters
        ----------
        S : torch.Tensor
            Singular values
        cumulative_explained_variance : torch.Tensor
            Cumulative explained variance
        max_rank : int
            Maximum possible rank
        rank : int, optional
            Explicit rank specification
        rank_thresh : float, optional
            Threshold for singular values
        rank_explained_variance : float, optional
            Explained variance threshold
            
        Returns
        -------
        int
            Computed rank
        """
        parameters_provided = [rank is not None, rank_thresh is not None, rank_explained_variance is not None]
        num_parameters_provided = sum(parameters_provided)
        
        if num_parameters_provided > 1:
            raise ValueError(
                "More than one rank parameter was provided. Please provide only one of rank, rank_thresh, or rank_explained_variance."
            )
        elif num_parameters_provided == 0:
            computed_rank = len(S)
        else:
            if rank is not None:
                computed_rank = rank
            elif rank_thresh is not None:
                # Find the number of singular values greater than the threshold
                computed_rank = int((S > rank_thresh).sum().item())
                if computed_rank == 0:
                    computed_rank = 1  # Ensure at least rank 1
            elif rank_explained_variance is not None:
                cumulative_explained_variance_cpu = cumulative_explained_variance.cpu().numpy()
                computed_rank = int(np.searchsorted(cumulative_explained_variance_cpu, rank_explained_variance) + 1)
                if computed_rank > len(S):
                    computed_rank = len(S)
        
        # Ensure rank doesn't exceed maximum possible
        if computed_rank > max_rank:
            computed_rank = max_rank
            
        return computed_rank

    def all_to_device(self, device="cpu"):
        """Move all tensor attributes to specified device."""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                self.__dict__[k] = [tensor.to(device) for tensor in v]

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the DMD model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Make predictions with the DMD model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def compute_hankel(self, *args, **kwargs):
        """Compute Hankel matrix. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def compute_svd(self, *args, **kwargs):
        """Compute SVD. Must be implemented by subclasses."""
        pass
