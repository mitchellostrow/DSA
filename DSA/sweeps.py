"""
Sweeps: Parameter sweep utilities for DMD-type models.

This module provides object-oriented sweeping over hyperparameters
for both local DMD and PyKoopman models.
"""

import numpy as np
from tqdm import tqdm
from .dmd import DMD
from .dmdc import DMDc
from .subspace_dmdc import SubspaceDMDc
from .stats import measure_nonnormality_transpose, compute_all_stats
from .resdmd import ResidualComputer
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import warnings
import torch

# Import pykoopman
from . import pykoopman as pk
from .pykoopman.observables import TimeDelay, Identity


def split_train_test(data, train_frac=0.8):
    """Split data into train and test sets."""
    if isinstance(data, list):
        train_data = [d for i, d in enumerate(data) if i < int(train_frac * len(data))]
        test_data = [d for i, d in enumerate(data) if i >= int(train_frac * len(data))]
        dim = data[0].shape[-1]
    elif data.ndim == 3 and data.shape[0] == 1:
        train_data = data[:, int(train_frac * data.shape[1]):]
        test_data = data[:, :int(train_frac * data.shape[1])]
        dim = data.shape[-1]
    else:
        train_data = data[:int(train_frac * data.shape[0])]
        test_data = data[int(train_frac * data.shape[0]):] if train_frac < 1.0 else train_data
        dim = data.shape[-1]
    return train_data, test_data, dim


class BaseSweeper(ABC):
    """Abstract base class for parameter sweeps over DMD-type models.
    
    This class provides the core infrastructure for sweeping over two hyperparameters
    of DMD-type models, computing various metrics (AIC, MASE, MSE, non-normality),
    and optionally computing residuals. Subclasses must implement the abstract methods
    to define how models are created, how predictions are made, and how to extract
    model properties.
    
    Parameters
    ----------
    data : np.ndarray
        Input data for the sweep. Can be 2D (time, features) or 3D (trials, time, features).
        Will be automatically split into train/test sets based on `train_frac`.
    param1_name : str
        Name of the first parameter to sweep over.
    param1_values : list or np.ndarray
        Values to sweep over for the first parameter.
    param2_name : str
        Name of the second parameter to sweep over.
    param2_values : list or np.ndarray
        Values to sweep over for the second parameter.
    train_frac : float, optional
        Fraction of data to use for training (default: 0.8).
    reseed : int, optional
        Number of reseeding steps for prediction (default: 1).
    compute_residuals : bool, optional
        Whether to compute residuals using ResDMD (default: False).
    control_data : np.ndarray, optional
        Control/actuation data for controlled systems (default: None).
        Will be split into train/test sets matching the data split.
    **model_kwargs
        Additional keyword arguments passed to the model constructor.
    
    Attributes
    ----------
    train_data : np.ndarray
        Training portion of the input data.
    test_data : np.ndarray
        Test portion of the input data.
    train_control : np.ndarray or None
        Training portion of control data (if provided).
    test_control : np.ndarray or None
        Test portion of control data (if provided).
    dim : int
        Dimensionality of the data (number of features).
    aics : np.ndarray
        AIC values for each parameter combination (after sweep).
    mases : np.ndarray
        MASE values for each parameter combination (after sweep).
    mses : np.ndarray
        MSE values for each parameter combination (after sweep).
    nnormals : np.ndarray
        Non-normality values for each parameter combination (after sweep).
    residuals : np.ndarray or None
        Residual values for each parameter combination (after sweep, if computed).
    fitted_models : list of lists
        Fitted model objects for each parameter combination (after sweep).
    """
    
    def __init__(
        self,
        data: np.ndarray,
        param1_name: str,
        param1_values: Union[List, np.ndarray],
        param2_name: str,
        param2_values: Union[List, np.ndarray],
        train_frac: float = 0.8,
        reseed: int = 1,
        compute_residuals: bool = False,
        control_data: np.ndarray = None,
        **model_kwargs
    ):
        self.data = data
        self.param1_name = param1_name
        self.param1_values = np.array(param1_values)
        self.param2_name = param2_name
        self.param2_values = np.array(param2_values)
        self.train_frac = train_frac
        self.reseed = reseed
        self.compute_residuals = compute_residuals
        self.model_kwargs = model_kwargs
        self.control_data = control_data
        
        self.train_data, self.test_data, self.dim = split_train_test(data, train_frac)
        
        # Split control data if provided
        if control_data is not None:
            self.train_control, self.test_control, _ = split_train_test(control_data, train_frac)
        else:
            self.train_control = self.test_control = None
        
        self._aics: np.ndarray = None
        self._mases: np.ndarray = None
        self._mses: np.ndarray = None
        self._nnormals: np.ndarray = None
        self._residuals: np.ndarray = None
        self._fitted_models: List[List] = None
        self._swept = False
    
    def _validate_control_data(self, model_class) -> None:
        """Validate control data against the model class, warning if incompatible.
        
        Args:
            model_class: The model class to validate against.
        """
        if self.control_data is not None and model_class is not None:
            if not self._is_control_model_class(model_class):
                warnings.warn(
                    f"control_data was provided, but model class ({model_class.__name__}) "
                    f"may not support control input. Control data may be ignored.",
                    UserWarning
                )
    
    @abstractmethod
    def _is_control_model_class(self, model_class) -> bool:
        """Check if a model class supports control input.
        
        Args:
            model_class: The model class to check.
            
        Returns:
            True if the model class supports control input, False otherwise.
        """
        pass
    
    @abstractmethod
    def make_model(self, p1_val, p2_val):
        pass
    
    @abstractmethod
    def get_state_matrix(self, model) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_rank(self, model) -> int:
        pass
    
    @abstractmethod
    def predict(self, model, test_data: np.ndarray, n_steps: int) -> np.ndarray:
        pass
    
    def _is_valid_param_combo(self, p1_val, p2_val) -> bool:
        return True
    
    def sweep(self) -> "BaseSweeper":
        """Run the parameter sweep."""
        n1, n2 = len(self.param1_values), len(self.param2_values)
        
        self._aics = np.full((n1, n2), np.nan)
        self._mases = np.full((n1, n2), np.nan)
        self._mses = np.full((n1, n2), np.nan)
        self._nnormals = np.full((n1, n2), np.nan)
        self._residuals = np.full((n1, n2), np.nan) if self.compute_residuals else None
        self._fitted_models = [[None for _ in range(n2)] for _ in range(n1)]
        
        for i, p1 in tqdm(enumerate(self.param1_values), total=n1, desc="Sweeping"):
            for j, p2 in enumerate(self.param2_values):
                if not self._is_valid_param_combo(p1, p2):
                    continue
                # try:
                model = self.make_model(p1, p2)
                self._fitted_models[i][j] = model
                
                pred = self.predict(model, self.test_data, self.reseed)
                pred_np = self._to_numpy(pred)
                test_np = self._to_numpy(self.test_data)
                pred_np, test_np = self._align_predictions(pred_np, test_np)
                
                rank = self.get_rank(model)
                stats = compute_all_stats(test_np, pred_np, rank)
                
                self._aics[i, j] = stats["AIC"]
                self._mases[i, j] = stats["MASE"]
                self._mses[i, j] = stats["MSE"]
                
                A = self.get_state_matrix(model)
                A_np = self._to_numpy(A)
                self._nnormals[i, j] = measure_nonnormality_transpose(A_np)
                
                if self.compute_residuals:
                    try:
                        rc = ResidualComputer(model, self.test_data)
                        self._residuals[i, j] = rc.get_average_residual()
                    except Exception as e:
                        warnings.warn(f"Residual computation failed: {e}")
                        self._residuals[i, j] = np.nan
                # except Exception as e:
                    # warnings.warn(f"Failed for {self.param1_name}={p1}, {self.param2_name}={p2}: {e}")
                    # continue
        
        self._swept = True
        return self
    
    def _to_numpy(self, x) -> np.ndarray:
        if hasattr(x, 'cpu'):
            return x.cpu().detach().numpy()
        if isinstance(x, list):
            return np.concatenate([self._to_numpy(xi) for xi in x], axis=0)
        return np.array(x)
    
    def _align_predictions(self, pred: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pred.ndim == 3 and test.ndim == 3:
            min_t = min(pred.shape[1], test.shape[1])
            return pred[:, :min_t], test[:, :min_t]
        elif pred.ndim == 2 and test.ndim == 2:
            min_t = min(pred.shape[0], test.shape[0])
            return pred[:min_t], test[:min_t]
        return pred, test
    
    @property
    def aics(self) -> np.ndarray:
        self._check_swept()
        return self._aics
    
    @property
    def mases(self) -> np.ndarray:
        self._check_swept()
        return self._mases
    
    @property
    def mses(self) -> np.ndarray:
        self._check_swept()
        return self._mses
    
    @property
    def nnormals(self) -> np.ndarray:
        self._check_swept()
        return self._nnormals
    
    @property
    def residuals(self) -> Optional[np.ndarray]:
        self._check_swept()
        return self._residuals
    
    @property
    def fitted_models(self) -> List[List]:
        self._check_swept()
        return self._fitted_models
    
    def _check_swept(self):
        if not self._swept:
            raise RuntimeError("Must call sweep() before accessing results.")
    
    def get_results(self) -> Dict[str, Any]:
        self._check_swept()
        return {
            "param1_name": self.param1_name,
            "param1_values": self.param1_values,
            "param2_name": self.param2_name,
            "param2_values": self.param2_values,
            "aics": self._aics,
            "mases": self._mases,
            "mses": self._mses,
            "nnormals": self._nnormals,
            "residuals": self._residuals,
            "train_frac": self.train_frac,
            "reseed": self.reseed,
        }
    
    def plot(
        self,
        x_axis_param: str = None,
        legend_param: str = None,
        metrics: List[str] = None,
        figsize: Tuple[float, float] = None,
        cmap: str = "viridis",
        save_path: str = None,
        mase_scale: str = "log",
        title: str = None,
    ):
        """Plot sweep results."""
        self._check_swept()
        
        if x_axis_param is None:
            x_axis_param = self.param2_name
        if legend_param is None:
            legend_param = self.param1_name
        
        if x_axis_param == self.param2_name:
            x_values = self.param2_values
            legend_values = self.param1_values
            transpose = False
        else:
            x_values = self.param1_values
            legend_values = self.param2_values
            transpose = True
        
        if metrics is None:
            metrics = ["AIC", "MASE", "nnormal"]
            if self._residuals is not None:
                metrics.append("residual")
        
        metric_data = {
            "AIC": self._aics, "MASE": self._mases, "MSE": self._mses,
            "nnormal": self._nnormals, "residual": self._residuals,
        }
        metric_labels = {
            "AIC": "AIC", "MASE": "MASE", "MSE": "MSE",
            "nnormal": "Non-normality", "residual": "Avg. Residual",
        }
        
        metrics = [m for m in metrics if metric_data.get(m) is not None]
        n_metrics = len(metrics)
        
        if figsize is None:
            figsize = (4 * n_metrics, 4)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        cmap_obj = plt.cm.get_cmap(cmap)
        n_legend = len(legend_values)
        
        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]
            data = metric_data[metric]
            if transpose:
                data = data.T
            
            for leg_idx, leg_val in enumerate(legend_values):
                color = cmap_obj(leg_idx / (n_legend + 2))
                y_data = data[leg_idx, :]
                ax.plot(x_values, y_data, label=f"{leg_val}", color=color, marker='o', markersize=3)
            
            if metric == "MASE":
                ax.axhline(1, color="black", linestyle="--", linewidth=0.7)
            if metric in ["MASE", "MSE"]:
                ax.set_yscale(mase_scale)
            
            ax.set_xlabel(x_axis_param)
            ax.set_ylabel(metric_labels[metric])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        
        axes[-1].legend(title=legend_param, loc="upper right", bbox_to_anchor=(1.4, 1), fontsize=8)
        
        if title:
            fig.suptitle(title, y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        return fig, axes


class DefaultSweeper(BaseSweeper):
    """Sweeper for the local DSA DMD model classes (DMD, DMDc, SubspaceDMDc).
    
    This sweeper performs parameter sweeps over the local DMD implementations,
    typically sweeping over `n_delays` and `rank` parameters. It supports both
    autonomous systems (DMD) and controlled systems (DMDc, SubspaceDMDc).
    
    Parameters
    ----------
    data : np.ndarray
        Input data for the sweep. See `BaseSweeper` for details.
    param1_name : str, optional
        Name of the first parameter to sweep (default: "n_delays").
    param1_values : list or np.ndarray, optional
        Values to sweep for the first parameter.
    param2_name : str, optional
        Name of the second parameter to sweep (default: "rank").
    param2_values : list or np.ndarray, optional
        Values to sweep for the second parameter.
    model_class : str, optional
        Model class to use: "DMD", "DMDc", or "SubspaceDMDc" (default: "DMD").
    control_data : np.ndarray, optional
        Control/actuation data. Required for "DMDc" and "SubspaceDMDc".
    **kwargs
        Additional arguments passed to `BaseSweeper` (train_frac, reseed, 
        compute_residuals) and to the model constructor.
    
    Examples
    --------
    >>> sweeper = DefaultSweeper(
    ...     data=X,
    ...     param1_values=np.arange(1, 5),
    ...     param2_values=np.arange(1, 15),
    ...     model_class="DMD"
    ... )
    >>> sweeper.sweep()
    >>> print(sweeper.mases)
    
    >>> # With control input
    >>> sweeper = DefaultSweeper(
    ...     data=X,
    ...     control_data=U,
    ...     param1_values=np.arange(1, 5),
    ...     param2_values=np.arange(1, 15),
    ...     model_class="DMDc"
    ... )
    
    See Also
    --------
    BaseSweeper : Base class with full parameter documentation.
    PyKoopmanSweeper : Sweeper for PyKoopman models.
    """
    
    # Model class names that support control input
    CONTROL_MODEL_NAMES = ('DMDc', 'SubspaceDMDc')
    
    def __init__(
        self,
        data: np.ndarray,
        param1_name: str = "n_delays",
        param1_values: Union[List, np.ndarray] = None,
        param2_name: str = "rank",
        param2_values: Union[List, np.ndarray] = None,
        model_class: str = "DMD",
        control_data: np.ndarray = None,
        **kwargs
    ):
        super().__init__(data=data, param1_name=param1_name, param1_values=param1_values,
                        param2_name=param2_name, param2_values=param2_values, 
                        control_data=control_data, **kwargs)
        self.model_class_name = model_class
        
        # Validate control data
        if model_class in self.CONTROL_MODEL_NAMES and control_data is None:
            raise ValueError(f"control_data required for {model_class}")
        if control_data is not None and model_class not in self.CONTROL_MODEL_NAMES:
            warnings.warn(
                f"control_data was provided, but model class '{model_class}' "
                f"may not support control input. Control data will be ignored.",
                UserWarning
            )
    
    def _is_control_model_class(self, model_class) -> bool:
        """Check if a model class supports control input."""
        if isinstance(model_class, str):
            return model_class in self.CONTROL_MODEL_NAMES
        return model_class.__name__ in self.CONTROL_MODEL_NAMES
    
    def _is_valid_param_combo(self, p1_val, p2_val) -> bool:
        n_delays = p1_val if self.param1_name == "n_delays" else (p2_val if self.param2_name == "n_delays" else None)
        rank = p1_val if self.param1_name == "rank" else (p2_val if self.param2_name == "rank" else None)
        if n_delays is None or rank is None:
            return True
        return rank <= n_delays * self.dim
    
    def make_model(self, p1_val, p2_val):
        kwargs = {self.param1_name: p1_val, self.param2_name: p2_val, **self.model_kwargs}
        if self.model_class_name == 'DMD':
            model = DMD(data=self.train_data, **kwargs)
        elif self.model_class_name == 'DMDc':
            model = DMDc(self.train_data, self.train_control, **kwargs)
        elif self.model_class_name == 'SubspaceDMDc':
            model = SubspaceDMDc(self.train_data, self.train_control, **kwargs)
        else:
            raise ValueError(f"Unknown model class: {self.model_class_name}")
        model.fit()
        return model
    
    def get_state_matrix(self, model) -> np.ndarray:
        A = model.A_v
        return A.cpu().detach().numpy() if hasattr(A, 'cpu') else A
    
    def get_rank(self, model) -> int:
        return getattr(model, 'rank', getattr(model, 'rank_output', 1))
    
    def predict(self, model, test_data: np.ndarray, n_steps: int) -> np.ndarray:
        if self.model_class_name == 'DMD':
            return model.predict(test_data, reseed=n_steps)
        return model.predict(test_data, self.test_control, reseed=n_steps)


class PyKoopmanSweeper(BaseSweeper):
    """Sweeper for PyKoopman models with dotted parameter names.
    
    This sweeper performs parameter sweeps over PyKoopman Koopman models,
    allowing sweeps over both observable parameters (e.g., time delays) and
    regressor parameters (e.g., SVD rank). Parameters are specified using
    dotted notation: "component.parameter" where component is "observables"
    (or "observable", "obs"), "regressor" (or "reg"), or "extra_obs.{index}"
    for extra observables.
    
    Parameters
    ----------
    data : np.ndarray
        Input data for the sweep. See `BaseSweeper` for details.
    param1_name : str, optional
        Name of the first parameter in "component.param" format
        (default: "observables.n_delays"). Supported components:
        - "observables", "observable", "obs": base observable parameters
        - "regressor", "reg": regressor parameters
        - "extra_obs.{index}": extra observable parameters (e.g., "extra_obs.0.D")
    param1_values : list or np.ndarray, optional
        Values to sweep for the first parameter.
    param2_name : str, optional
        Name of the second parameter in "component.param" format
        (default: "regressor.svd_rank").
    param2_values : list or np.ndarray, optional
        Values to sweep for the second parameter.
    base_observable_class : class, optional
        Observable class to use (default: TimeDelay).
    base_observable_kwargs : dict, optional
        Additional kwargs for the observable constructor.
    base_regressor_class : class, optional
        Regressor class to use. If None and control_data is provided,
        defaults to pk.regression.DMDc. Otherwise uses PyDMDRegressor.
    base_regressor_kwargs : dict, optional
        Additional kwargs for the regressor constructor.
    extra_observables : list, optional
        Additional observable objects or classes to combine with the base observable.
        Can be a mix of:
        - Observable instances (fixed, not swept)
        - Observable classes (can be swept via "extra_obs.{index}.param" syntax)
    extra_observable_kwargs : list of dict, optional
        Default kwargs for each extra observable class. Only used when the
        corresponding entry in extra_observables is a class (not an instance).
        Length must match extra_observables.
    control_data : np.ndarray, optional
        Control/actuation data for controlled systems.
    **kwargs
        Additional arguments passed to `BaseSweeper` (train_frac, reseed,
        compute_residuals).
    
    Examples
    --------
    >>> # Sweep over time delays and SVD rank
    >>> sweeper = PyKoopmanSweeper(
    ...     data=X,
    ...     param1_name="observable.n_delays",
    ...     param1_values=[1, 2, 3, 4],
    ...     param2_name="regressor.svd_rank",
    ...     param2_values=[5, 10, 15, 20],
    ... )
    >>> sweeper.sweep()
    >>> sweeper.plot()
    
    >>> # Sweep TimeDelay n_delays and RandomFourierFeatures D
    >>> sweeper = PyKoopmanSweeper(
    ...     data=X,
    ...     param1_name="obs.n_delays",
    ...     param1_values=[3, 5, 7],
    ...     param2_name="extra_obs.0.D",
    ...     param2_values=[50, 100, 200],
    ...     base_observable_class=pk.observables.TimeDelay,
    ...     extra_observables=[pk.observables.RandomFourierFeatures],  # class, not instance
    ...     extra_observable_kwargs=[{"include_state": True}],  # default kwargs
    ... )
    
    >>> # With control input
    >>> import pykoopman as pk
    >>> sweeper = PyKoopmanSweeper(
    ...     data=X,
    ...     control_data=U,
    ...     param1_name="observable.n_delays",
    ...     param1_values=[1, 2, 3],
    ...     param2_name="regressor.svd_rank",
    ...     param2_values=[5, 10],
    ...     base_regressor_class=pk.regression.DMDc,
    ... )
    
    See Also
    --------
    BaseSweeper : Base class with full parameter documentation.
    DefaultSweeper : Sweeper for local DMD models.
    
    Notes
    -----
    The component names in param1_name/param2_name are flexible:
    - For observables: "observable", "observables", or "obs"
    - For regressor: "regressor" or "reg"
    - For extra observables: "extra_obs.{index}" (e.g., "extra_obs.0", "extra_obs.1")
    """
    
    def __init__(
        self,
        data: np.ndarray,
        param1_name: str = "observables.n_delays",
        param1_values: Union[List, np.ndarray] = None,
        param2_name: str = "regressor.svd_rank",
        param2_values: Union[List, np.ndarray] = None,
        base_observable_class=None,
        base_observable_kwargs: dict = None,
        base_regressor_class=None,
        base_regressor_kwargs: dict = None,
        extra_observables: Union[List,Any] = None,
        extra_observable_kwargs: List[dict] = None,
        control_data: np.ndarray = None,
        **kwargs
    ):
        super().__init__(data=data, param1_name=param1_name, param1_values=param1_values,
                        param2_name=param2_name, param2_values=param2_values,
                        control_data=control_data, **kwargs)
        
        self.base_observable_class = base_observable_class or TimeDelay
        self.base_observable_kwargs = base_observable_kwargs or {}
        self.base_regressor_class = base_regressor_class
        self.base_regressor_kwargs = base_regressor_kwargs or {}
        self.extra_observables = extra_observables or []
        if not isinstance(self.extra_observables, list):
            self.extra_observables = [self.extra_observables]
        self.extra_observable_kwargs = extra_observable_kwargs or [{} for _ in self.extra_observables]
        
        # Validate extra_observable_kwargs length
        if len(self.extra_observable_kwargs) != len(self.extra_observables):
            raise ValueError(
                f"extra_observable_kwargs length ({len(self.extra_observable_kwargs)}) must match "
                f"extra_observables length ({len(self.extra_observables)})"
            )
        
        # Warn if the regressor class doesn't support control input
        if control_data is not None and base_regressor_class is not None:
            if not self._is_control_model_class(base_regressor_class):
                warnings.warn(
                    f"control_data was provided, but base_regressor_class ({base_regressor_class.__name__}) "
                    f"may not support control input. Consider using a DMDc-type regressor (e.g., pk.regression.DMDc, "
                    f"pk.regression.EDMDc, or pydmd.DMDc). Control data may be ignored.",
                    UserWarning
                )
        
        # Parse parameter names into components
        self._param1_parsed = self._parse_param_name(param1_name)
        self._param2_parsed = self._parse_param_name(param2_name)
        # Keep legacy attributes for backward compatibility
        self._param1_component, self._param1_attr = param1_name.split('.', 1)
        self._param2_component, self._param2_attr = param2_name.split('.', 1)
        # Handle extra_obs.N.param format for legacy attrs
        if self._param1_parsed['type'] == 'extra_obs':
            self._param1_attr = self._param1_parsed['attr']
        if self._param2_parsed['type'] == 'extra_obs':
            self._param2_attr = self._param2_parsed['attr']
    
    def _is_control_model_class(self, model_class) -> bool:
        """Check if a model/regressor class supports control input.
        
        Args:
            model_class: The model class to check.
            
        Returns:
            True if the model class supports control input, False otherwise.
        """
        # Simple check: class name ends with 'DMDc' (covers DMDc, EDMDc, etc.)
        return model_class.__name__.endswith('DMDc')
    
    @staticmethod
    def _is_observable_component(component: str) -> bool:
        """Check if a component name refers to observables."""
        return component in ("observable", "observables", "obs")
    
    @staticmethod
    def _is_regressor_component(component: str) -> bool:
        """Check if a component name refers to regressor."""
        return component in ("regressor", "reg")
    
    @staticmethod
    def _is_extra_obs_component(component: str) -> bool:
        """Check if a component name refers to extra observables."""
        return component.startswith("extra_obs")
    
    def _parse_param_name(self, param_name: str) -> dict:
        """Parse a parameter name into its components.
        
        Handles formats:
        - "obs.n_delays" -> {'type': 'observable', 'attr': 'n_delays', 'index': None}
        - "reg.svd_rank" -> {'type': 'regressor', 'attr': 'svd_rank', 'index': None}
        - "extra_obs.0.D" -> {'type': 'extra_obs', 'attr': 'D', 'index': 0}
        
        Args:
            param_name: Parameter name in dotted notation.
            
        Returns:
            Dict with 'type', 'attr', and 'index' keys.
        """
        parts = param_name.split('.')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid parameter name: {param_name}. Expected 'component.param' format.")
        
        component = parts[0]
        
        if self._is_observable_component(component):
            return {'type': 'observable', 'attr': parts[1], 'index': None}
        elif self._is_regressor_component(component):
            return {'type': 'regressor', 'attr': parts[1], 'index': None}
        elif self._is_extra_obs_component(component):
            # Format: "extra_obs.{index}.{attr}" or "extra_obs{index}.{attr}"
            if len(parts) == 3:
                # "extra_obs.0.D"
                try:
                    index = int(parts[1])
                except ValueError:
                    raise ValueError(f"Invalid extra_obs index in: {param_name}. Expected integer.")
                return {'type': 'extra_obs', 'attr': parts[2], 'index': index}
            elif len(parts) == 2:
                # Try "extra_obs0.D" format
                import re
                match = re.match(r'extra_obs(\d+)', component)
                if match:
                    return {'type': 'extra_obs', 'attr': parts[1], 'index': int(match.group(1))}
                raise ValueError(f"Invalid extra_obs format: {param_name}. Use 'extra_obs.N.param' or 'extra_obsN.param'.")
            else:
                raise ValueError(f"Invalid extra_obs format: {param_name}. Use 'extra_obs.N.param'.")
        else:
            raise ValueError(f"Unknown component: {component}. Expected 'obs', 'reg', or 'extra_obs'.")
            
    @staticmethod
    def _cast_to_int(val):
        return int(val) if isinstance(val, np.integer) else val

    def _build_observable(self, p1_val, p2_val):
        """Build the composite observable with swept parameters.
        
        Args:
            p1_val: Value for param1.
            p2_val: Value for param2.
            
        Returns:
            Combined observable (base + extras).
        """
        # Build base observable
        kwargs = dict(self.base_observable_kwargs)
        if self._param1_parsed['type'] == 'observable':
            kwargs[self._param1_parsed['attr']] = self._cast_to_int(p1_val)
        if self._param2_parsed['type'] == 'observable':
            kwargs[self._param2_parsed['attr']] = self._cast_to_int(p2_val)
        obs = self.base_observable_class(**kwargs)

        # Build and add extra observables
        for i, extra in enumerate(self.extra_observables):
            extra_kwargs = dict(self.extra_observable_kwargs[i])
            
            # Check if param1 targets this extra observable
            if self._param1_parsed['type'] == 'extra_obs' and self._param1_parsed['index'] == i:
                extra_kwargs[self._param1_parsed['attr']] = p1_val
            # Check if param2 targets this extra observable
            if self._param2_parsed['type'] == 'extra_obs' and self._param2_parsed['index'] == i:
                extra_kwargs[self._param2_parsed['attr']] = p2_val
            
            # Build extra observable: if it's a class, instantiate it; if instance, use as-is
            if isinstance(extra, type):
                # It's a class, instantiate with kwargs
                extra_obs = extra(**extra_kwargs)
            else:
                # It's already an instance
                if extra_kwargs:
                    warnings.warn(
                        f"extra_observables[{i}] is an instance, not a class. "
                        f"Cannot apply swept parameters. Pass a class instead to sweep its parameters.",
                        UserWarning
                    )
                extra_obs = extra
            
            obs = obs + extra_obs
        return obs
    
    def _build_regressor(self, p1_val, p2_val):
        kwargs = dict(self.base_regressor_kwargs)
        if self._is_regressor_component(self._param1_component):
            kwargs[self._param1_attr] = self._cast_to_int(p1_val)
        if self._is_regressor_component(self._param2_component):
            kwargs[self._param2_attr] = self._cast_to_int(p2_val)
        if self.base_regressor_class:
            # Return the regressor instance directly - pk.Koopman handles pydmd regressors
            return self.base_regressor_class(**kwargs)
        # Default regressor: DMDc if control data is provided, else pydmd.DMD
        if self.control_data is not None:
            return pk.regression.DMDc(**kwargs)
        from pydmd import DMD as PyDMD_DMD
        svd_rank = int(kwargs.pop('svd_rank', -1))
        return PyDMD_DMD(svd_rank=svd_rank, **kwargs)
    
    def make_model(self, p1_val, p2_val):
        obs = self._build_observable(p1_val, p2_val)
        reg = self._build_regressor(p1_val, p2_val)
        model = pk.Koopman(observables=obs, regressor=reg)
        train = self.train_data
        if isinstance(train, torch.Tensor):
            train = train.cpu().numpy()
        if train.ndim == 3:
            train = train.reshape(-1, train.shape[-1])
        
        # Prepare control data for fitting if provided
        train_u = None
        if self.train_control is not None:
            train_u = self.train_control
            if isinstance(train_u, torch.Tensor):
                train_u = train_u.cpu().numpy()
            if train_u.ndim == 3:
                train_u = train_u.reshape(-1, train_u.shape[-1])
        
        model.fit(train, u=train_u)
        return model
    
    def get_state_matrix(self, model) -> np.ndarray:
        """Get the reduced-order state transition matrix.
        
        Returns the state matrix A from the regressor, which should be
        of shape (svd_rank, svd_rank) when SVD truncation is applied.
        """
        A = model.A
        if hasattr(A, 'cpu'):
            A = A.cpu().detach().numpy()
        return np.asarray(A)
    
    def get_rank(self, model) -> int:
        reg = model._regressor()
        if hasattr(reg, 'svd_rank') and reg.svd_rank > 0:
            return reg.svd_rank
        return model.A.shape[0]
    
    def predict(self, model, test_data: np.ndarray, reseed: int) -> np.ndarray:
        """Generate one-step predictions using model.predict.
        
        Uses PyKoopman's predict method which does one-step-ahead prediction.
        The predictions are offset by n_consumed samples due to time delays.
        
        Args:
            model: Fitted PyKoopman model.
            test_data: Test data to predict on.
            reseed: Not used currently (kept for API compatibility).
        
        Returns:
            Predictions aligned with test_data.
        """
        test_np = test_data.cpu().numpy() if isinstance(test_data, torch.Tensor) else test_data
        
        # Prepare test control data if available
        test_u = None
        if self.test_control is not None:
            test_u = self.test_control
            if isinstance(test_u, torch.Tensor):
                test_u = test_u.cpu().numpy()
        
        if test_np.ndim == 3:
            if test_u is not None and test_u.ndim == 3:
                return np.stack([self._predict_single(model, t, u=test_u[i]) 
                               for i, t in enumerate(test_np)])
            return np.stack([self._predict_single(model, t, u=test_u) for t in test_np])
        return self._predict_single(model, test_np, u=test_u)
    
    def _predict_single(self, model, data: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """One-step predictions for a single trajectory.
        
        Args:
            model: Fitted PyKoopman model.
            data: 2D array of shape (time, features).
            u: Optional control data.
        
        Returns:
            Predictions array of same shape as data.
        """
        n_consumed = getattr(model.observables, 'n_consumed_samples', 0)
        n_samples = data.shape[0]
        
        # Need at least n_consumed + 1 samples to predict
        if n_samples <= n_consumed + 1:
            return data.copy()
        
        # Use model.predict for one-step-ahead predictions
        # Input: data[:-1], Output: predictions for data[1:]
        # But with time delays, we lose n_consumed samples
        try:
            # Predict uses all but the last sample to predict the next
            x_input = data[:-1]  # Shape: (n_samples - 1, n_features)
            u_input = u[:-1] if u is not None else None
            
            pred = model.predict(x_input, u=u_input)
            
            # pred has shape (n_samples - 1 - n_consumed, n_features)
            # We need to align this with the original data
            predictions = np.zeros_like(data)
            predictions[:n_consumed + 1] = data[:n_consumed + 1]
            predictions[n_consumed + 1:] = pred
            
            return predictions
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            return data.copy()


def sweep_local_dmd(data, n_delays_values, rank_values, **kwargs) -> DefaultSweeper:
    """Convenience function to create and run a DefaultSweeper."""
    sweeper = DefaultSweeper(data=data, param1_values=n_delays_values, param2_values=rank_values, **kwargs)
    sweeper.sweep()
    return sweeper


def sweep_pykoopman(data, param1_name, param1_values, param2_name, param2_values, **kwargs) -> PyKoopmanSweeper:
    """Convenience function to create and run a PyKoopmanSweeper."""
    sweeper = PyKoopmanSweeper(data=data, param1_name=param1_name, param1_values=param1_values,
                               param2_name=param2_name, param2_values=param2_values, **kwargs)
    sweeper.sweep()
    return sweeper


def sweep_ranks_delays(data, n_delays, ranks, control_data=None, train_frac=0.8, reseed=5,
                       return_residuals=True, model_class='DMD', **model_kwargs):
    """Backward-compatible wrapper around DefaultDMDSweeper."""
    sweeper = DefaultSweeper(data=data, param1_values=n_delays, param2_values=ranks,
                              model_class=model_class, control_data=control_data,
                              train_frac=train_frac, reseed=reseed,
                              compute_residuals=return_residuals, **model_kwargs)
    sweeper.sweep()
    result = (sweeper.aics, sweeper.mases, sweeper.nnormals)
    if return_residuals:
        result += (sweeper.residuals if sweeper.residuals is not None else np.full_like(sweeper.aics, np.nan),)
    return result
