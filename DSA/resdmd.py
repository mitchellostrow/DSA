"""
ResDMD: Residual-based analysis of DMD eigenvalues.

This module provides tools for computing and analyzing eigenvalue residuals
for DMD-type models, including both local DMD and PyKoopman models.

Supports DMD with control (DMDc) models by subtracting out control effects
before computing residuals on the autonomous dynamics.
"""

import warnings
import numpy as np
from typing import Literal, Tuple

try:
    from .dmd import DMD, embed_signal_torch
    from .dmdc import embed_data_DMDc
    from .subspace_dmdc import SubspaceDMDc
except ImportError:
    from dmd import DMD, embed_signal_torch
    try:
        from dmdc import embed_data_DMDc
        from subspace_dmdc import SubspaceDMDc
    except ImportError:
        DMDc = None
        SubspaceDMDc = None
        embed_data_DMDc = None

import torch


# =============================================================================
# Control Input Handling Helpers
# =============================================================================

def _requires_control(model) -> bool:
    """
    Check if a model requires control input for residual computation.
    
    Parameters
    ----------
    model : DMD, DMDc, SubspaceDMDc, or pk.Koopman
        Fitted model.
    
    Returns
    -------
    bool
        True if the model is a control model requiring control_data.
    """
    # Local DMDc or SubspaceDMDc
    if hasattr(model, 'B_v') and model.B_v is not None:
        return True
    
    # PyKoopman with control (check via B property)
    if hasattr(model, '_pipeline'):
        if getattr(model, 'B', None) is not None:
            return True
    
    return False


def _validate_control_input(model, control_data, data) -> None:
    """
    Validate control data matches model requirements.
    
    Parameters
    ----------
    model : fitted model
        The DMD-type model.
    control_data : array-like or None
        Control input data.
    data : array-like
        State data (for shape validation).
    
    Raises
    ------
    ValueError
        If model requires control but none provided.
    
    Warns
    -----
    UserWarning
        If control_data provided but model doesn't use control.
    """
    requires = _requires_control(model)
    
    if requires and control_data is None:
        raise ValueError(
            f"Model of type {type(model).__name__} requires control input, "
            "but control_data was not provided."
        )
    
    if not requires and control_data is not None:
        warnings.warn(
            f"control_data was provided, but model of type {type(model).__name__} "
            "does not use control input. Control data will be ignored.",
            UserWarning
        )
        return
    
    if control_data is not None:
        # Validate temporal structure matches
        data_arr = data.cpu().numpy() if hasattr(data, 'cpu') else np.asarray(data)
        ctrl_arr = control_data.cpu().numpy() if hasattr(control_data, 'cpu') else np.asarray(control_data)
        
        # Check dimensionality matches
        if data_arr.ndim != ctrl_arr.ndim:
            raise ValueError(
                f"data and control_data must have the same number of dimensions. "
                f"Got data.ndim={data_arr.ndim}, control_data.ndim={ctrl_arr.ndim}"
            )
        
        # Check temporal dimension matches (axis 0 for 2D, axis 1 for 3D)
        if data_arr.ndim == 2:
            if data_arr.shape[0] != ctrl_arr.shape[0]:
                raise ValueError(
                    f"data and control_data must have the same number of time points. "
                    f"Got data.shape[0]={data_arr.shape[0]}, control_data.shape[0]={ctrl_arr.shape[0]}"
                )
        elif data_arr.ndim == 3:
            if data_arr.shape[0] != ctrl_arr.shape[0] or data_arr.shape[1] != ctrl_arr.shape[1]:
                raise ValueError(
                    f"data and control_data must have the same number of trials and time points. "
                    f"Got data.shape={data_arr.shape[:2]}, control_data.shape={ctrl_arr.shape[:2]}"
                )


def _subtract_control_effects(
    Y: np.ndarray, 
    B: np.ndarray, 
    U: np.ndarray
) -> np.ndarray:
    """
    Subtract control effects to isolate autonomous dynamics.
    
    Computes Y_corrected = Y - B @ U.T (or Y - U @ B.T depending on shape).
    
    Parameters
    ----------
    Y : np.ndarray
        Output data matrix. Shape (T, rank).
    B : np.ndarray
        Control matrix. Shape (rank, control_dim) or compatible.
    U : np.ndarray
        Control input matrix. Shape (T, control_dim) or compatible.
    
    Returns
    -------
    Y_corrected : np.ndarray
        Y with control effects removed. Shape (T, rank).
    """
    # Y is (T, rank), U is (T, control_dim), B is (rank, control_dim)
    # We want Y_corrected = Y - (B @ U.T).T = Y - U @ B.T
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    
    # Compute control contribution: (T, control_dim) @ (control_dim, rank) = (T, rank)
    control_contribution = U @ B.T
    Y_corrected = Y - control_contribution
    
    return Y_corrected


# =============================================================================
# Core Residual Computation
# =============================================================================

def _compute_residuals_from_matrices(
    X: np.ndarray,
    Y: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> Tuple:
    """
    Core residual computation from projected data matrices and eigendecomposition.
    
    shared implementation used by both local DMD and PyKoopman.
    
    Parameters
    ----------
    X : np.ndarray
        Projected input data matrix. Shape (T, rank).
    Y : np.ndarray
        Projected output data matrix. Shape (T, rank).
    eigenvalues : np.ndarray
        Eigenvalues of the state matrix. Shape (rank,).
    eigenvectors : np.ndarray
        Eigenvectors of the state matrix. Shape (rank, rank).
    
    Returns
    -------
    residuals : np.ndarray
        Residuals for each eigenpair.
    normalized_residuals : np.ndarray
        Normalized residuals (relative to persistence baseline).
    """
    rank = len(eigenvalues)
    
    # Compute Gram matrices
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, Y)
    YtY = np.dot(Y.T, Y)
    YtX = np.dot(Y.T, X)
    
    residuals = np.zeros(rank, dtype=np.complex128)
    persistence_residuals = np.zeros(rank, dtype=np.complex128)
    
    for i in range(rank):
        g = eigenvectors[:, i]
        lam = eigenvalues[i]
        
        denominator = np.dot(g.conj().T, np.dot(XtX, g))
        numerator = np.dot(
            g.conj().T,
            np.dot(
                YtY - np.conj(lam) * XtY - lam * YtX + np.abs(lam) ** 2 * XtX,
                g,
            ),
        )
        residuals[i] = numerator / denominator
        
        # Persistence baseline (lambda = 1)
        persistence_numerator = np.dot(
            g.conj().T, np.dot(YtY - XtY - YtX + XtX, g)
        )
        persistence_residuals[i] = persistence_numerator / denominator
    
    normalized_residuals = np.abs(residuals) / (np.abs(persistence_residuals) + 1e-10)
    
    return residuals, normalized_residuals


def compute_residuals(
    dmd: "DMD",
    data: "np.ndarray | torch.Tensor",
    Y: "np.ndarray | torch.Tensor" = None,
    control_data: "np.ndarray | torch.Tensor" = None,
    matrix: Literal["A_v", "A_havok_dmd"] = "A_v",
):
    """
    Compute DMD eigenvalues, eigenvectors, and residuals for each mode.

    Parameters
    ----------
    dmd : DMD or DMDc object
        Fitted DMD object (with A_v or A_havok_dmd, U, S_mat_inv, rank, n_delays, delay_interval).
        For DMDc models, also requires B_v and control-related SVD matrices.
    data : np.ndarray or torch.Tensor
        Input data matrix. Can be 2D (T x N) or 3D (K x T x N).
        If Y is not provided, X and Y will be constructed by splitting along the time axis.
    Y : np.ndarray or torch.Tensor, optional
        Right-hand side data matrix. If not provided, will be constructed from data.
    control_data : np.ndarray or torch.Tensor, optional
        Control input data. Required for DMDc models. Must have the same temporal
        structure as data (same number of time points and trials).
    matrix : Literal["A_v", "A_havok_dmd"], optional
        Matrix to compute residuals on. Must be either "A_v" or "A_havok_dmd". Default is "A_v".

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues of the state matrix.
    eigenvectors : np.ndarray
        Eigenvectors of the state matrix.
    residuals : np.ndarray
        Residuals for each eigenpair.
    normalized_residuals : np.ndarray
        Normalized residuals.
    
    Notes
    -----
    For DMDc models (when control_data is provided), residuals are computed on the
    autonomous dynamics by subtracting out the control effects:
    
        Y_corrected = Y - B @ U
    
    This allows assessment of eigenvalue quality for the A matrix alone.
    """
    # Validate control input
    _validate_control_input(dmd, control_data, data)
    
    # Check if this is a DMDc model
    is_dmdc = hasattr(dmd, 'B_v') and dmd.B_v is not None and control_data is not None
    
    # Get parameters from the DMD model
    n_delays = dmd.n_delays
    delay_interval = dmd.delay_interval if hasattr(dmd, 'delay_interval') else 1
    steps_ahead = dmd.steps_ahead if dmd.steps_ahead is not None and hasattr(dmd,'steps_ahead') else 1
    device = dmd.device
    
    # For DMDc, we need rank_output for state and rank_input for control
    if is_dmdc:
        rank_output = dmd.rank_output if hasattr(dmd, 'rank_output') else dmd.rank
        rank_input = dmd.rank_input if hasattr(dmd, 'rank_input') else dmd.rank
        rank = rank_output
    else:
        rank = dmd.rank

    # Convert to torch tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device)
    else:
        data = data.to(device)
    
    if control_data is not None:
        if isinstance(control_data, np.ndarray):
            control_data = torch.from_numpy(control_data).to(device)
        else:
            control_data = control_data.to(device)

    # Construct X and Y if Y is not provided
    if Y is None:
        if data.ndim == 3:
            X_data = data[:, :-steps_ahead]
            Y_data = data[:, steps_ahead:]
            if control_data is not None:
                # Control at time t affects transition from x_t to x_{t+1}
                U_data = control_data[:, :-steps_ahead]
        else:
            X_data = data[:-steps_ahead]
            Y_data = data[steps_ahead:]
            if control_data is not None:
                U_data = control_data[:-steps_ahead]
    else:
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).to(device)
        else:
            Y = Y.to(device)
        X_data = data
        Y_data = Y
        if control_data is not None:
            U_data = control_data

    # Compute delay embeddings for state data
    H_X = embed_signal_torch(X_data, n_delays, delay_interval)
    H_Y = embed_signal_torch(Y_data, n_delays, delay_interval)
    
    # Compute delay embeddings for control data if present
    if is_dmdc:
        n_control_delays = dmd.n_control_delays if hasattr(dmd, 'n_control_delays') else 1
        H_U = embed_data_DMDc(
            U_data, 
            n_delays=n_delays,
            n_control_delays=n_control_delays,
            delay_interval=delay_interval,
            control=True
        )
        if isinstance(H_U, np.ndarray):
            H_U = torch.from_numpy(H_U).to(device)

    # Flatten if 3D (combine trials and time)
    if H_X.ndim == 3:
        H_X = H_X.reshape(-1, H_X.shape[-1])
        H_Y = H_Y.reshape(-1, H_Y.shape[-1])
        if is_dmdc:
            H_U = H_U.reshape(-1, H_U.shape[-1])

    # Get the appropriate matrix and compute eigendecomposition
    A = getattr(dmd, matrix)
    if hasattr(A, "cpu"):
        A = A.cpu().detach().numpy()
    # Note: The DMD convention uses Y ≈ X @ A^T, so for the residual
    # ||Yg - λXg||^2 to be small, we need g to be an eigenvector of A^T
    eigenvalues, eigenvectors = np.linalg.eig(A.T)

    # Project into appropriate coordinates based on matrix type
    if matrix == "A_havok_dmd":
        # For A_havok_dmd, data is already in the right space (observation space)
        X = H_X.cpu().detach().numpy() if hasattr(H_X, "cpu") else H_X
        Y = H_Y.cpu().detach().numpy() if hasattr(H_Y, "cpu") else H_Y
        
        if is_dmdc:
            # Get B_havok_dmd and project control
            B = dmd.B_havok_dmd
            if hasattr(B, "cpu"):
                B = B.cpu().detach().numpy()
            U = H_U.cpu().detach().numpy() if hasattr(H_U, "cpu") else H_U
            # Subtract control effects
            Y = _subtract_control_effects(Y, B, U)
            
    elif matrix == "A_v":
        # For A_v, project into V coordinates: X_proj = X @ U @ S_mat_inv[:rank, :rank]
        # DMDc uses Uh/Sh_mat_inv, regular DMD uses U/S_mat_inv
        if is_dmdc:
            U_proj_mat = dmd.Uh[:, :rank]
            S_inv_mat = dmd.Sh_mat_inv[:rank, :rank]
        else:
            U_proj_mat = dmd.U[:, :rank]
            S_inv_mat = dmd.S_mat_inv[:rank, :rank]
        
        X_proj = H_X @ U_proj_mat @ S_inv_mat
        Y_proj = H_Y @ U_proj_mat @ S_inv_mat
        
        X = X_proj.cpu().detach().numpy() if hasattr(X_proj, "cpu") else X_proj
        Y = Y_proj.cpu().detach().numpy() if hasattr(Y_proj, "cpu") else Y_proj
        
        if is_dmdc:
            # Project control into its reduced space
            Uu = dmd.Uu[:, :rank_input].cpu().numpy()
            Su_mat_inv = dmd.Su_mat_inv[:rank_input, :rank_input].cpu().numpy()
            U_proj = H_U @ Uu @ Su_mat_inv
            U = U_proj.cpu().detach().numpy() if hasattr(U_proj, "cpu") else U_proj
            
            # Get B_v and subtract control effects
            B_v = dmd.B_v
            if hasattr(B_v, "cpu"):
                B_v = B_v.cpu().detach().numpy()
            
            # Subtract control effects: Y_corrected = Y - B_v @ U.T
            Y = _subtract_control_effects(Y, B_v, U)
    else:
        raise ValueError(f"matrix must be 'A_v' or 'A_havok_dmd', got {matrix}")

    # Truncate eigenvalues/vectors to rank
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]

    # Use shared core computation
    residuals, normalized_residuals = _compute_residuals_from_matrices(
        X, Y, eigenvalues, eigenvectors
    )
    
    return eigenvalues, eigenvectors, residuals, normalized_residuals


def compute_residuals_pykoopman(
    model,
    test_data: np.ndarray,
    control_data: np.ndarray = None,
):
    """
    Compute residuals for a fitted PyKoopman model.
    
    Parameters
    ----------
    model : pk.Koopman
        Fitted PyKoopman model. Can use DMD, EDMD, DMDc, EDMDc, or PyDMD regressors.
    test_data : np.ndarray
        Test data. Shape (T, N) or (K, T, N).
    control_data : np.ndarray, optional
        Control input data. Required for models with control (DMDc, EDMDc, PyDMD DMDc).
        Must have the same temporal structure as test_data.
    
    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues of the state matrix.
    eigenvectors : np.ndarray
        Eigenvectors of the state matrix.
    residuals : np.ndarray
        Residuals for each eigenpair.
    normalized_residuals : np.ndarray
        Normalized residuals.
    
    Notes
    -----
    For control models (DMDc, EDMDc, PyDMD DMDc), residuals are computed on the
    autonomous dynamics by subtracting out the control effects:
    
        Y_corrected = Y - B @ U
    
    This allows assessment of eigenvalue quality for the A matrix alone.
    """
    # Ensure test_data is numpy
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.cpu().numpy()
    if control_data is not None and isinstance(control_data, torch.Tensor):
        control_data = control_data.cpu().numpy()
    
    # Validate control input
    _validate_control_input(model, control_data, test_data)
    
    # Check if this is a control model
    is_control = _requires_control(model) and control_data is not None
    
    # Split data into X and Y with temporal offset BEFORE transforming
    # This correctly handles trial boundaries - each (X[k,t], Y[k,t]) pair is aligned
    if test_data.ndim == 3:
        X_data = test_data[:, :-1]  # (K, T-1, N)
        Y_data = test_data[:, 1:]   # (K, T-1, N)
        if is_control:
            U_data = control_data[:, :-1]  # (K, T-1, control_dim)
    else:
        X_data = test_data[:-1]  # (T-1, N)
        Y_data = test_data[1:]   # (T-1, N)
        if is_control:
            U_data = control_data[:-1]  # (T-1, control_dim)
    
    # Transform to observable space - handles 3D structure automatically
    H_X = model.observables.transform(X_data)
    H_Y = model.observables.transform(Y_data)
    
    # Flatten if 3D (combine trials and time)
    if H_X.ndim == 3:
        H_X = H_X.reshape(-1, H_X.shape[-1])
        H_Y = H_Y.reshape(-1, H_Y.shape[-1])
        if is_control:
            # Account for samples consumed by observable (e.g., TimeDelay)
            n_consumed = getattr(model.observables, 'n_consumed_samples', 0)
            U_data = U_data[:, n_consumed:]  # (K, T-1-n_consumed, control_dim)
            U_data = U_data.reshape(-1, U_data.shape[-1])
    elif is_control:
        n_consumed = getattr(model.observables, 'n_consumed_samples', 0)
        U_data = U_data[n_consumed:]
    
    # Get state matrix and eigendecomposition
    # Note: PyKoopman uses Y ≈ X @ A^T convention, so for the residual
    # ||Yg - λXg||^2 to be small, we need g to be an eigenvector of A^T
    A = model.A
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    
    # Get projection matrix and project into reduced space
    ur = model.ur
    rank = ur.shape[1]
    
    X = H_X @ ur  # (total_samples, rank)
    Y = H_Y @ ur  # (total_samples, rank)
    
    # Handle control if present - subtract control effects
    if is_control:
        B = model.B
        # Subtract control effects: Y_corrected = Y - U @ B.T
        Y = _subtract_control_effects(Y, B, U_data)
    
    # Truncate eigenvalues/vectors to rank
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]
    
    # Use shared core computation
    residuals, normalized_residuals = _compute_residuals_from_matrices(
        X, Y, eigenvalues, eigenvectors
    )
    
    return eigenvalues, eigenvectors, residuals, normalized_residuals


def compute_residuals_subspace_dmdc(
    model: "SubspaceDMDc",
    test_data: np.ndarray = None,
    control_data: np.ndarray = None,
    use_training_latents: bool = False,
    projection_method: str = 'smooth',
):
    """
    Compute residuals for a fitted SubspaceDMDc model.
    
    Parameters
    ----------
    model : SubspaceDMDc
        Fitted SubspaceDMDc model.
    test_data : np.ndarray, optional
        Test data (observations). Shape (T, N) or (K, T, N) or list of arrays.
        If None and use_training_latents=True, uses training data.
    control_data : np.ndarray, optional
        Control input data. Must have the same temporal structure as test_data.
        If None and use_training_latents=True, uses training control data.
    use_training_latents : bool, default=False
        If True, uses the exact latent states from training (stored in model.info['X_hat']).
        This should give near-zero residuals for training data since A was fit to these states.
        If False, projects test_data to latent space using Kalman filtering/smoothing.
    projection_method : str, default='smooth'
        Method for projecting observations to latent states (only used if use_training_latents=False):
        - 'smooth': Kalman smoothing (uses all observations, better estimates)
        - 'filter': Kalman filtering (causal, uses only past observations)
    
    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues of the state matrix A.
    eigenvectors : np.ndarray
        Eigenvectors of the state matrix A.
    residuals : np.ndarray
        Residuals for each eigenpair.
    normalized_residuals : np.ndarray
        Normalized residuals.
    
    Notes
    -----
    For SubspaceDMDc, the system operates in latent state space:
    
        x_{t+1} = A @ x_t + B @ u_t
        y_t = C @ x_t
    
    Residuals are computed in the latent space by:
    1. Projecting observations to latent states via Kalman smoothing/filtering
    2. Subtracting control effects: X_next_corrected = X_next - B @ U
    3. Computing residuals on (X, X_next_corrected) using the A matrix eigenstructure
    """
    # Get model parameters
    rank = model.info['rank_used']
    A = model.A_v
    B = model.B_v
    
    if use_training_latents:
        # Use the exact latent states from training - should give near-zero residuals
        X, X_next, U = model.get_training_latent_states(return_aligned=True)
    else:
        if control_data is None:
            raise ValueError("control_data is required for SubspaceDMDc residual computation.")
        
        # Convert to numpy if needed
        if isinstance(test_data, torch.Tensor):
            test_data = test_data.cpu().numpy()
        if isinstance(control_data, torch.Tensor):
            control_data = control_data.cpu().numpy()
        
        # Project test data to latent space using Kalman filtering/smoothing
        # Returns time-aligned (X, X_next, U) in row-major format
        X, X_next, U = model.project_to_latent(
            test_data, control_data, return_aligned=True, method=projection_method
        )
    
    # Subtract control effects: X_next_corrected = X_next - B @ U.T
    Y_corrected = _subtract_control_effects(X_next, B, U)
    
    # Compute eigendecomposition of A
    # Note: SubspaceDMDc uses X_next = A @ X + B @ U in column format,
    # but after transposing to row format, we have Y ≈ X @ A^T.
    # So for ||Yg - λXg||^2 to be small, we need g to be an eigenvector of A^T
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    
    # Truncate to rank
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]
    
    # Use shared core computation
    residuals, normalized_residuals = _compute_residuals_from_matrices(
        X, Y_corrected, eigenvalues, eigenvectors
    )
    
    return eigenvalues, eigenvectors, residuals, normalized_residuals


# =============================================================================
# ResidualComputer Class
# =============================================================================

class ResidualComputer:
    """
    Computes and plots eigenvalue residuals for DMD-type models.
    
    Supports local DMD, local DMDc, SubspaceDMDc, and PyKoopman models
    (including those with control: DMDc, EDMDc, PyDMD DMDc).
    
    Parameters
    ----------
    model : DMD, DMDc, SubspaceDMDc, or pk.Koopman
        A fitted DMD-type model.
    test_data : np.ndarray
        Test data for residual computation. Shape (T, N) or (K, T, N).
    control_data : np.ndarray, optional
        Control input data. Required for models with control.
        Must have the same temporal structure as test_data.
    model_type : str, optional
        Model type: "auto", "local_dmd", "subspace_dmdc", or "pykoopman". 
        Default "auto".
    
    Attributes
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of the state matrix.
    eigenvectors : np.ndarray
        Eigenvectors of the state matrix.
    residuals : np.ndarray
        Residuals for each eigenpair.
    normalized_residuals : np.ndarray
        Normalized residuals.
    
    Example
    -------
    >>> # For autonomous DMD
    >>> rc = ResidualComputer(model, test_data)
    >>> eigenvalues, eigenvectors, residuals, norm_resid = rc.compute()
    >>> rc.plot(title="Eigenvalue Residuals")
    
    >>> # For DMDc with control
    >>> rc = ResidualComputer(dmdc_model, test_data, control_data=control_test)
    >>> eigenvalues, eigenvectors, residuals, norm_resid = rc.compute()
    """
    
    def __init__(
        self, 
        model, 
        test_data: np.ndarray,
        control_data: np.ndarray = None,
        model_type: str = "auto"
    ):
        self.model = model
        self.test_data = test_data
        self.control_data = control_data
        self.model_type = self._detect_model_type(model) if model_type == "auto" else model_type
        
        # Validate control input
        _validate_control_input(model, control_data, test_data)
        
        # Results (computed lazily)
        self.eigenvalues = None
        self.eigenvectors = None
        self.residuals = None
        self.normalized_residuals = None
        self._computed = False
    
    def _detect_model_type(self, model) -> str:
        """Detect the model type for dispatch."""
        # Check for SubspaceDMDc first (it has A_v but also has 'info' with Gamma_hat)
        if hasattr(model, 'A_v') and hasattr(model, 'info') and model.info is not None:
            if 'Gamma_hat' in model.info:
                return "subspace_dmdc"
        
        # Local DMD or DMDc
        if hasattr(model, 'A_v') and hasattr(model, 'n_delays'):
            return "local_dmd"
        
        # PyKoopman
        if hasattr(model, '_pipeline') and hasattr(model, 'A'):
            return "pykoopman"
        
        raise ValueError(f"Cannot detect model type for {type(model)}. "
                        "Please specify model_type explicitly.")
    
    def compute(self):
        """
        Compute residuals for the model.
        
        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues.
        eigenvectors : np.ndarray  
            Eigenvectors.
        residuals : np.ndarray
            Residuals for each eigenpair.
        normalized_residuals : np.ndarray
            Normalized residuals.
        """
        if self.model_type == "local_dmd":
            result = compute_residuals(
                self.model, 
                self.test_data, 
                control_data=self.control_data
            )
        elif self.model_type == "subspace_dmdc":
            result = compute_residuals_subspace_dmdc(
                self.model, 
                self.test_data, 
                self.control_data
            )
        elif self.model_type == "pykoopman":
            result = compute_residuals_pykoopman(
                self.model, 
                self.test_data, 
                control_data=self.control_data
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.eigenvalues, self.eigenvectors, self.residuals, self.normalized_residuals = result
        self._computed = True
        return self.eigenvalues, self.eigenvectors, self.residuals, self.normalized_residuals
    
    def plot(
        self, 
        cmin: float = None, 
        cmax: float = None,
        ax=None,
        figsize: tuple = (6, 6),
        title: str = None,
    ):
        """
        Plot eigenvalues on the complex plane, colored by residuals.
        
        Parameters
        ----------
        cmin : float, optional
            Minimum value for color scale.
        cmax : float, optional
            Maximum value for color scale.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : tuple
            Figure size if creating new figure.
        title : str, optional
            Plot title.
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib import colors as mcolors

        if not self._computed:
            self.compute()
        
        residuals_real = np.real(self.residuals)
        
        if cmin is None:
            cmin = np.min(residuals_real)
        if cmax is None:
            cmax = np.max(residuals_real)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()
        
        cmap = cm.viridis
        norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
        
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        sc = ax.scatter(
            np.real(self.eigenvalues), 
            np.imag(self.eigenvalues), 
            c=residuals_real, 
            cmap=cmap, 
            norm=norm
        )
        cbar = plt.colorbar(sc, ax=ax, orientation="vertical")
        cbar.set_label("Residual")
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        return fig, ax
    
    def get_average_residual(self) -> float:
        """Return mean absolute residual."""
        if not self._computed:
            self.compute()
        return float(np.mean(np.abs(self.residuals)))


# =============================================================================
# Utility Functions
# =============================================================================

def plot_residuals(eigenvalues, residuals, cmin=None, cmax=None):
    """
    Plot eigenvalues on the complex plane, colored by residuals.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues.
    residuals : np.ndarray
        Residuals for each eigenpair.
    cmin : float, optional
        Minimum value for color scale.
    cmax : float, optional
        Maximum value for color scale.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import colors as mcolors

    residuals_real = np.abs(residuals)
    if cmin is None:
        cmin = np.min(residuals_real)
    if cmax is None:
        cmax = np.max(residuals_real)
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    sc = plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), c=residuals_real, cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, orientation="vertical")
    cbar.set_label("Residual")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")


def compute_inverse_participation_ratio(residuals):
    """Compute inverse participation ratio from residuals."""
    if isinstance(residuals, list):
        residuals = np.array(residuals)
    residuals = np.abs(residuals)
    inv_resid = 1 / (residuals + 1e-10)
    num = np.sum(inv_resid) ** 2
    denom = np.sum(inv_resid**2)
    return num / denom


def clean_spectrum(eigenvalues, eigenvectors, residuals, epsilon):
    """
    Remove eigenvalues with residual greater than epsilon.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues.
    eigenvectors : np.ndarray
        Eigenvectors.
    residuals : np.ndarray
        Residuals.
    epsilon : float
        Threshold.
    
    Returns
    -------
    eigenvalues, eigenvectors, residuals : filtered arrays
    """
    residuals_real = np.abs(residuals)
    mask = residuals_real < epsilon
    return eigenvalues[mask], eigenvectors[:, mask], residuals[mask]


def thresh_topn(eigenvalues, eigenvectors, residuals, n):
    """
    Keep the top n eigenvalues with smallest residuals.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues.
    eigenvectors : np.ndarray
        Eigenvectors.
    residuals : np.ndarray
        Residuals.
    n : int
        Number to keep.
    
    Returns
    -------
    eigenvalues, eigenvectors, residuals : filtered arrays
    """
    residuals_real = np.abs(residuals)
    sorted_resid = np.sort(residuals_real)
    if n > len(sorted_resid):
        n = -1
    topn = sorted_resid[n]
    mask = residuals_real <= topn
    return eigenvalues[mask], eigenvectors[:, mask], residuals[mask]


def format_eigs(eigenvalues):
    """Format eigenvalues as 2D array sorted by real magnitude."""
    if isinstance(eigenvalues, list):
        eigenvalues = np.array(eigenvalues)
    # Sort by real magnitude
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues.real))]
    return np.vstack([eigenvalues.real, eigenvalues.imag]).T


def compute_ot_distance(a, b):
    """
    Compute optimal transport distance between two sets of eigenvalues.
    
    Parameters
    ----------
    a, b : np.ndarray
        Eigenvalue arrays (can be complex).
    
    Returns
    -------
    score : float
        OT distance.
    C : np.ndarray
        Transport plan.
    """
    import ot

    # Convert complex to 2D if needed
    if np.iscomplexobj(a):
        a = np.vstack([a.real, a.imag]).T
    if np.iscomplexobj(b):
        b = np.vstack([b.real, b.imag]).T
    M = ot.dist(a, b)
    a_weights = np.ones(a.shape[0]) / a.shape[0]
    b_weights = np.ones(b.shape[0]) / b.shape[0]
    score = ot.emd2(a_weights, b_weights, M)
    C = ot.emd(a_weights, b_weights, M)
    return score, C


# Module exports
__all__ = [
    "ResidualComputer",
    "compute_residuals",
    "compute_residuals_pykoopman",
    "compute_residuals_subspace_dmdc",
    "plot_residuals",
    "compute_inverse_participation_ratio",
    "clean_spectrum",
    "thresh_topn",
    "format_eigs",
    "compute_ot_distance",
]
