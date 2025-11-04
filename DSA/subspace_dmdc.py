"""This module computes the subspace DMD with control (DMDc) model for a given dataset."""
import numpy as np
import warnings
import torch
from .base_dmd import BaseDMD

class SubspaceDMDc(BaseDMD):
    """Subspace DMDc class for computing and predicting with DMD with control models.
    
    Inherits from BaseDMD for common functionality like device management and data processing.
    """
    def __init__(
            self,
            data,
            control_data=None,
            n_delays=1,
            rank=None,
            lamb=1e-8,
            device='cpu',
            verbose=False,
            send_to_cpu=False,
            backend='n4sid',
    ):
        """
        Initialize SubspaceDMDc.
        
        Parameters
        ----------
        data : array-like
            Output/observation data
        control_data : array-like
            Control input data
        n_delays : int
            Number of time delays (past window)
        rank : int, optional
            Rank for system identification
        lamb : float
            Regularization parameter for ridge regression
        device : str or torch.device
            Device for computation:
            - 'cpu': Use NumPy on CPU (fastest for CPU)
            - 'cuda' or 'cuda:X': Use PyTorch on GPU (auto-falls back to CPU if unavailable)
        verbose : bool
            If True, print progress information
        send_to_cpu : bool
            If True, move results to CPU after fitting (useful for batch GPU processing)
        backend : str
            'n4sid' or 'custom' for subspace identification algorithm
        """
        # Initialize base class
        super().__init__(device=device, verbose=verbose, send_to_cpu=send_to_cpu, lamb=lamb)
        
        self._init_data(data, control_data)
        self._check_same_shape()

        # Smart device setup with graceful fallback
        self.device, self.use_torch = self._setup_device(device, True)
        
        # SubspaceDMDc specific attributes
        self.data = data
        self.control_data = control_data
        if self.control_data is None:
            raise ValueError("no control data detected, use DMD or SubspaceDMD instead")
        self.A_v = None
        self.B_v = None
        self.C_v = None
        self.info = None
        self.n_delays = n_delays
        self.rank = rank
        self.backend = backend

    def fit(self):
        """Fit the SubspaceDMDc model."""
        self.A_v, self.B_v, self.C_v, self.info = self.subspace_dmdc_multitrial_flexible(
                                                            y=self.data,
                                                            u=self.control_data,
                                                            p=self.n_delays,
                                                            f=self.n_delays,
                                                            n=self.rank, 
                                                            backend=self.backend,
                                                            lamb=self.lamb)
        
        # Send to CPU if requested (inherited from BaseDMD)
        if self.send_to_cpu:
            self.all_to_device(device='cpu')


    def _init_data(self, data, control_data=None):
        # Process main data
        self.data, data_is_ragged = self._process_single_dataset(data)

        # Process control data
        if control_data is not None:
            self.control_data, control_is_ragged = self._process_single_dataset(
                control_data
            )
        else:
            raise ValueError("control data should be present, otherwise use DMD")
            # self.control_data = torch.zeros_like(self.data)
            # control_is_ragged = False

        # Check consistency between data and control_data
        if data_is_ragged != control_is_ragged:
            raise ValueError(
                "Data and control data have different structure (type or dimensionality)."
            )

        if data_is_ragged:
            # Additional validation for ragged data
            if not all(d.shape[-1] == control_data[0].shape[-1] for d in control_data):
                raise ValueError(
                    "All control tensors in the list must have the same number of features (last dimension)."
                )
            if not all(
                d.shape[0] == control_d.shape[0]
                for d, control_d in zip(data, control_data)
            ):
                raise ValueError(
                    "Data and control_data tensors must have the same number of time steps."
                )

            # Set attributes for ragged data
            n_features = self.data[0].shape[-1]
            self.n = n_features
            self.ntrials = sum(d.shape[0] if d.ndim == 3 else 1 for d in self.data)
            self.trial_counts = [d.shape[0] if d.ndim == 3 else 1 for d in self.data]
            self.is_list_data = True
        else:
            # Set attributes for non-ragged data
            if self.data.ndim == 3:
                self.ntrials = self.data.shape[0]
                self.n = self.data.shape[2]
            else:
                self.n = self.data.shape[1]
                self.ntrials = 1
            self.is_list_data = False
            

    def _check_same_shape(self):
        if isinstance(self.data,(np.ndarray,torch.Tensor)):
            assert self.data.shape[:-1] == self.control_data.shape[:-1]
        elif isinstance(self.data,list):
            assert len(self.data) == len(self.control_data)

            for d,c in zip(self.data,self.control_data):
                assert d.shape[:-1] == c.shape[:-1]


    def _collect_data(self, y_list, u_list, p, f):
        """Helper function to validate dimensions and collect data from trials."""
        p_out = y_list[0].shape[-1]
        m = u_list[0].shape[-1]
        
        U_p_all = []
        Y_p_all = []
        U_f_all = []
        Y_f_all = []
        valid_trials = []
        T_per_trial = []
        
        def hankel_stack(X, start, L):
            # X is now (n_timepoints, n_features), so we transpose for slicing
            # then stack along axis=0 to get (L * n_features, 1)
            return np.concatenate([X[start + i:start + i + 1, :].T for i in range(L)], axis=0)
        
        for trial_idx, (Y_trial, U_trial) in enumerate(zip(y_list, u_list)):
            N_trial = Y_trial.shape[0]
            T_trial = N_trial - (p + f)
            
            if T_trial <= 0:
                continue

            valid_trials.append(trial_idx)
            T_per_trial.append(T_trial)

            # Build Hankel matrices for this trial
            U_p_trial = np.concatenate([hankel_stack(U_trial, j, p) for j in range(T_trial)], axis=1)
            Y_p_trial = np.concatenate([hankel_stack(Y_trial, j, p) for j in range(T_trial)], axis=1)
            U_f_trial = np.concatenate([hankel_stack(U_trial, j + p, f) for j in range(T_trial)], axis=1)
            Y_f_trial = np.concatenate([hankel_stack(Y_trial, j + p, f) for j in range(T_trial)], axis=1)
            
            U_p_all.append(U_p_trial)
            Y_p_all.append(Y_p_trial)
            U_f_all.append(U_f_trial)
            Y_f_all.append(Y_f_trial)
        
        if not valid_trials:
            raise ValueError("No trials have sufficient time points for given number of delays")
        elif len(valid_trials) < len(y_list) // 2:
            warnings.warn(
                f"Only {len(valid_trials)} out of {len(y_list)} trials have sufficient time points "
                f"relative to the number of delays. This may affect model quality. Consider reducing the number of delays.")
        else:
            # print(f"Using {len(valid_trials)} out of {len(y_list)} trials with sufficient time points.")
            pass
        
        U_p = np.concatenate(U_p_all, axis=1)
        Y_p = np.concatenate(Y_p_all, axis=1)
        U_f = np.concatenate(U_f_all, axis=1)
        Y_f = np.concatenate(Y_f_all, axis=1)
        
        T_total = sum(T_per_trial)
        Z_p = np.vstack([U_p, Y_p])
        
        return U_p, Y_p, U_f, Y_f, Z_p, valid_trials, T_per_trial, T_total, p_out, m

    def subspace_dmdc_multitrial_QR_decomposition(self, y_list, u_list, p, f, n=None, lamb=1e-8, energy=0.999):
        """
        Subspace-DMDc for multi-trial data with variable trial lengths using QR decomposition.
        """
        U_p, Y_p, U_f, Y_f, Z_p, valid_trials, T_per_trial, T_total, p_out, m = \
            self._collect_data(y_list, u_list, p, f)

        H = np.vstack([U_f, Z_p, Y_f])
        
        dim_uf = f * m
        dim_zp = p * (m + p_out)

        def calculate_projection_and_svd(H, Z_p):
            # Check if inputs are torch tensors
            is_torch = isinstance(H, torch.Tensor) or isinstance(Z_p, torch.Tensor)
            
            if is_torch:
                Q, R_upper = torch.linalg.qr(H.T)
                L = R_upper.T
                
                R22 = L[dim_uf:dim_uf + dim_zp, dim_uf:dim_uf + dim_zp]
                R32 = L[dim_uf + dim_zp:, dim_uf:dim_uf + dim_zp]
                
                O = R32 @ torch.linalg.pinv(R22) @ Z_p
                Uo, s, Vt = torch.linalg.svd(O, full_matrices=False)
            else:
                Q, R_upper = np.linalg.qr(H.T, mode='reduced')
                L = R_upper.T
                
                R22 = L[dim_uf:dim_uf + dim_zp, dim_uf:dim_uf + dim_zp]
                R32 = L[dim_uf + dim_zp:, dim_uf:dim_uf + dim_zp]
                
                O = R32 @ np.linalg.pinv(R22) @ Z_p
                Uo, s, Vt = np.linalg.svd(O, full_matrices=False)
            
            return Uo, s, Vt

        if self.use_torch and H.shape[1] > 100:
            H_torch = self._to_torch(H)
            Z_p_torch = self._to_torch(Z_p)
            Uo, s, Vt = calculate_projection_and_svd(H_torch, Z_p_torch)
            
            Uo = self._to_numpy(Uo)
            s = self._to_numpy(s)
            Vt = self._to_numpy(Vt)
        else:
            Uo, s, Vt = calculate_projection_and_svd(H, Z_p)

        if n is None:
            cs = np.cumsum(s**2) / (s**2).sum()
            n = int(np.searchsorted(cs, energy) + 1)
            n = max(1, min(n, min(Uo.shape[1], Vt.shape[0])))

        U_n = Uo[:, :n]
        S_n = np.diag(s[:n])
        V_n = Vt[:n, :]
        S_half = np.sqrt(S_n)
        Gamma_hat = U_n @ S_half
        X_hat = S_half @ V_n

        X, X_next, U_mid, Y_curr = self._time_align_valid_trials(X_hat, u_list, y_list, valid_trials, T_per_trial, p)


        A_hat, B_hat = self._perform_ridge_regression(X, X_next, U_mid, n, lamb)

        C_hat = Gamma_hat[:p_out, :]
        noise_covariance, R_hat, Q_hat, S_hat = self._estimate_noise_covariance(X_next, A_hat, X, B_hat, U_mid, Y_curr, C_hat)

        info = {
            "singular_values_O": s, 
            "rank_used": n, 
            "Gamma_hat": Gamma_hat, 
            "f": f,
            "n_trials_total": len(y_list),
            "n_trials_used": len(valid_trials),
            "valid_trials": valid_trials,
            "T_per_trial": T_per_trial,
            "T_total": T_total,
            "trial_lengths": [y.shape[1] for y in y_list],
            "noise_covariance": noise_covariance,
            'R_hat': R_hat,
            'Q_hat': Q_hat,
            'S_hat': S_hat,
            'X_hat': X_hat
        }
        
        return A_hat, B_hat, C_hat, info

    def _time_align_valid_trials(self, X_hat, u_list, y_list, valid_trials, T_per_trial, p):
        """Helper function to time-align trials for regression."""
        # import pdb; pdb.set_trace()
        X_segments = []
        X_next_segments = []
        U_mid_segments = []
        Y_segments = []

        start_idx = 0
        for trial_idx, T_trial in enumerate(T_per_trial):
            X_trial = X_hat[:, start_idx:start_idx + T_trial]
            X_trial_curr = X_trial[:, :-1]
            X_trial_next = X_trial[:, 1:]
            
            original_trial_idx = valid_trials[trial_idx]
            U_trial = u_list[original_trial_idx]
            # U_trial is now (n_timepoints, n_features), slice rows then transpose
            U_mid_trial = U_trial[p:p + (T_trial - 1), :].T
            
            X_segments.append(X_trial_curr)
            X_next_segments.append(X_trial_next)
            U_mid_segments.append(U_mid_trial)

            Y_trial = y_list[original_trial_idx]
            # Y_trial is now (n_timepoints, n_features), slice rows then transpose
            Y_trial_curr = Y_trial[p:p+T_trial-1, :].T
            Y_segments.append(Y_trial_curr)
                        
            start_idx += T_trial
        
        X = np.concatenate(X_segments, axis=1)
        X_next = np.concatenate(X_next_segments, axis=1)
        U_mid = np.concatenate(U_mid_segments, axis=1)
        Y_curr = np.concatenate(Y_segments, axis=1)

        return X, X_next, U_mid, Y_curr

    def _perform_ridge_regression(self, X, X_next, U_mid, n, lamb):
        """Helper function to perform ridge regression."""
        Z = np.vstack([X, U_mid])
        if self.use_torch and Z.shape[1] > 100:
            Z_torch = self._to_torch(Z)
            X_next_torch = self._to_torch(X_next)
            
            ZTZ = Z_torch @ Z_torch.T
            ridge_term = lamb * torch.eye(ZTZ.shape[0], device=self.device, dtype=Z_torch.dtype)
            AB = torch.linalg.solve(ZTZ + ridge_term, Z_torch @ X_next_torch.T).T
            AB = self._to_numpy(AB)
        else:
            ZTZ = Z @ Z.T
            ridge_term = lamb * np.eye(ZTZ.shape[0])
            AB = np.linalg.solve(ZTZ + ridge_term, Z @ X_next.T).T

        A_hat = AB[:, :n]
        B_hat = AB[:, n:]

        return A_hat, B_hat

    def _estimate_noise_covariance(self, X_next, A_hat, X, B_hat, U_mid, Y_curr, C_hat):
        """Helper function to estimate the noise covariance matrix."""
        W_hat = X_next - (A_hat @ X + B_hat @ U_mid)
        V_hat = Y_curr - (C_hat @ X)

        V_hat = V_hat - V_hat.mean(axis=1, keepdims=True)
        W_hat = W_hat - W_hat.mean(axis=1, keepdims=True)
        N_res = V_hat.shape[1]
        denom = max(N_res - 1, 1)

        R_hat = (V_hat @ V_hat.T) / denom
        Q_hat = (W_hat @ W_hat.T) / denom
        S_hat = (W_hat @ V_hat.T) / denom

        eps = 1e-12
        R_hat = 0.5 * (R_hat + R_hat.T) + eps * np.eye(R_hat.shape[0])
        Q_hat = 0.5 * (Q_hat + Q_hat.T) + eps * np.eye(Q_hat.shape[0])

        noise_covariance = np.block([[R_hat, S_hat.T],
                                     [S_hat, Q_hat]])
        
        return noise_covariance, R_hat, Q_hat, S_hat

    def subspace_dmdc_multitrial_custom(self, y_list, u_list, p, f, n=None, lamb=1e-8, energy=0.999):
        """Subspace-DMDc using custom method."""
        U_p, Y_p, U_f, Y_f, Z_p, valid_trials, T_per_trial, T_total, p_out, m = \
            self._collect_data(y_list, u_list, p, f)
        
        UfUfT = U_f @ U_f.T
        Xsolve = np.linalg.solve(UfUfT + lamb*np.eye(UfUfT.shape[0]), U_f)
        Pi_perp = np.eye(T_total) - U_f.T @ Xsolve
        Yf_perp = Y_f @ Pi_perp
        Zp_perp = Z_p @ Pi_perp
        
        ZZT = Zp_perp @ Zp_perp.T
        Zp_pinv_left = np.linalg.solve(ZZT + lamb*np.eye(ZZT.shape[0]), Zp_perp)
        P = Zp_perp.T @ Zp_pinv_left
        O = Yf_perp @ P
        
        Uo, s, Vt = np.linalg.svd(O, full_matrices=False)
        if n is None:
            cs = np.cumsum(s**2) / (s**2).sum()
            n = int(np.searchsorted(cs, energy) + 1)
            n = max(1, min(n, min(Uo.shape[1], Vt.shape[0])))
        
        U_n = Uo[:, :n]
        S_n = np.diag(s[:n])
        V_n = Vt[:n, :]
        S_half = np.sqrt(S_n)
        Gamma_hat = U_n @ S_half
        X_hat = S_half @ V_n

        X, X_next, U_mid, Y_curr = self._time_align_valid_trials(X_hat, u_list, y_list, valid_trials, T_per_trial, p)
        if any([i == 0 for i in X.shape]):
            raise ValueError ("too many delays for dataset, reduce number")
        A_hat, B_hat = self._perform_ridge_regression(X, X_next, U_mid, n, lamb)
        
        C_hat = Gamma_hat[:p_out, :]
        noise_covariance, R_hat, Q_hat, S_hat = self._estimate_noise_covariance(X_next, A_hat, X, B_hat, U_mid, Y_curr, C_hat)

        info = {
            "singular_values_O": s, 
            "rank_used": n, 
            "Gamma_hat": Gamma_hat, 
            "f": f,
            "n_trials_total": len(y_list),
            "n_trials_used": len(valid_trials),
            "valid_trials": valid_trials,
            "T_per_trial": T_per_trial,
            "T_total": T_total,
            "trial_lengths": [y.shape[1] for y in y_list],
            "noise_covariance": noise_covariance,
            'R_hat': R_hat,
            'Q_hat': Q_hat,
            'S_hat': S_hat,
            'X_hat': X_hat
        }
        
        return A_hat, B_hat, C_hat, info


    def _convert_to_subspace_dmdc_data_format(self,y, u):
        """Convert the data and control data to the format required for SubspaceDMDc."""
        if isinstance(y, list) and isinstance(u, list):
            y_list = []
            u_list = []
            for y_trial, u_trial in zip(y, u):
                if y_trial.ndim == 3 and u_trial.ndim == 3:
                    for t in range(len(y_trial)):
                        y_list.append(y_trial[t])
                        u_list.append(u_trial[t])
                elif y_trial.ndim == 2 and u_trial.ndim == 2:
                    y_list.append(y_trial)
                    u_list.append(u_trial)
                else:
                    raise ValueError("Invalid dimension. Only list of (n_trials, n_timepoints, n_features) or (n_timepoints, n_features) arrays are supported.")
        else:
            if y.ndim == 2:
                y_list = [y]
                u_list = [u]
            else:
                y_list = [y[i] for i in range(y.shape[0])]
                u_list = [u[i] for i in range(u.shape[0])]
            y_list = [y_trial for y_trial in y_list]
            u_list = [u_trial for u_trial in u_list]
        return y_list, u_list


    def subspace_dmdc_multitrial_flexible(self, y, u, p, f, n=None, lamb=1e-8, energy=0.999, backend='n4sid'):
        """
        Flexible wrapper that handles both fixed-length and variable-length multi-trial data.
        
        Parameters:
        - y: either (n_trials, p_out, N) array, (p_out, N) array, or list of (p_out, N_i) arrays
        - u: either (n_trials, m, N) array, (m, N) array, or list of (m, N_i) arrays
        """
        y_list, u_list = self._convert_to_subspace_dmdc_data_format(y, u)
        if backend == 'n4sid':
            return self.subspace_dmdc_multitrial_QR_decomposition(y_list, u_list, p, f, n, lamb, energy)
        else:
            return self.subspace_dmdc_multitrial_custom(y_list, u_list, p, f, n, lamb, energy)


    def predict(self, test_data=None, control_data=None, reseed=None):
        """Predict using the Kalman filter."""
        if test_data is None:
            test_data = self.data
        if control_data is None:
            control_data = self.control_data

        if reseed is None:
            reseed = 1

        if isinstance(test_data, list):
            self.kalman = OnlineKalman(self)
            Y_pred = []
            for trial in range(len(test_data)):
                self.kalman.reset()
                trial_predictions = [
                    self.kalman.step(y=test_data[trial][t] if t % reseed == 0 else None, u=control_data[trial][t])[0]
                    for t in range(test_data[trial].shape[0])
                ]
                Y_pred.append(np.concatenate(trial_predictions, axis=1).T)
            return Y_pred

        self.kalman = OnlineKalman(self)
        if test_data.ndim == 2:
            return np.concatenate(
                [self.kalman.step(y=test_data[t] if t % reseed == 0 else None, u=control_data[t])[0] for t in range(test_data.shape[0])],
                axis=1
            ).T
        else:
            Y_pred = []
            for trial in range(test_data.shape[0]):
                self.kalman.reset()
                trial_predictions = [
                    self.kalman.step(y=test_data[trial, t] if t % reseed == 0 else None, u=control_data[trial, t])[0]
                    for t in range(test_data.shape[1])
                ]
                Y_pred.append(np.concatenate(trial_predictions, axis=1).T)
            return np.array(Y_pred)
 
 
    def compute_hankel(self, *args, **kwargs):
        """Compute Hankel matrices for SubspaceDMDc."""
        raise NotImplementedError(
            "Hankel matrix computation is integrated into the fit() method for SubspaceDMDc. "
            "Use fit() to compute the model."
        )
    
    def compute_svd(self, *args, **kwargs):
        """Compute SVD for SubspaceDMDc."""
        raise NotImplementedError(
            "SVD computation is integrated into the fit() method for SubspaceDMDc. "
            "Use fit() to compute the model."
        )


class OnlineKalman:
    """Online Kalman Filter class for real-time state estimation."""

    def __init__(self, dmdc):
        """Initialize the Online Kalman Filter with a fitted DMDc model."""
        self.A = dmdc.A_v
        self.B = dmdc.B_v  
        self.C = dmdc.C_v
        self.R = dmdc.info['R_hat']
        self.S = dmdc.info['S_hat']
        self.Q = dmdc.info['Q_hat']
        
        self.y_dim, self.x_dim = self.C.shape
        self.u_dim = self.B.shape[1]
        
        self.reset()

    def step(self, y=None, u=None, reg_coef=1e-6):
        """Perform one step of the Kalman filter."""
        x_pred, p_pred = self._predict()
        p_pred_reg = p_pred + reg_coef * np.eye(self.x_dim)

        u = self._ensure_column_vector(u, self.u_dim)
        y = self._ensure_column_vector(y, self.y_dim)

        S_innov = self.R + self.C @ p_pred_reg @ self.C.T
        K_filtered = p_pred_reg @ self.C.T @ np.linalg.pinv(S_innov)
        p_filtered = self._regularize_and_symmetrize(p_pred_reg - K_filtered @ self.C @ p_pred_reg, reg_coef)

        x_filtered = x_pred + K_filtered @ (y - self.C @ x_pred) if not np.isnan(y).any() else x_pred.copy()

        K_pred = (self.S + self.A @ p_pred_reg @ self.C.T) @ np.linalg.pinv(S_innov)
        p_predicted = self._regularize_and_symmetrize(self.A @ p_pred_reg @ self.A.T + self.Q - K_pred @ (self.S + self.A @ p_pred_reg @ self.C.T).T, reg_coef)

        x_predicted = self.A @ x_pred + self.B @ u + (K_pred @ (y - self.C @ x_pred) if not np.isnan(y).any() else 0)

        self._store_results(x_filtered, x_predicted, p_filtered, p_predicted, u, y, K_pred)

        return self.y_filtereds[-1], self.x_filtereds[-1]

    def _store_results(self, x_filtered, x_predicted, p_filtered, p_predicted, u, y, K_pred):
        """Helper function to store filter results."""
        self.p_filtereds.append(p_filtered)
        self.x_filtereds.append(x_filtered)
        self.p_predicteds.append(p_predicted)
        self.x_predicteds.append(x_predicted)
        self.us.append(u)
        self.ys.append(y)
        self.y_filtereds.append(self.C @ x_filtered)
        self.y_predicteds.append(self.C @ x_predicted)
        self.kalman_gains.append(K_pred)

    def _predict(self):
        """Helper function for prediction step."""
        x_pred = self.x_predicteds[-1] if self.x_predicteds else np.zeros((self.x_dim, 1))
        p_pred = self.p_predicteds[-1] if self.p_predicteds else np.eye(self.x_dim)
        return x_pred, p_pred

    def _ensure_column_vector(self, vector, dim):
        """Ensure the input is a column vector."""
        if vector is not None and vector.ndim == 1:
            vector = vector.reshape(-1, 1)
        if vector is None:
            vector = np.zeros((dim, 1))
        return vector

    def _regularize_and_symmetrize(self, matrix, reg_coef):
        """Regularize and ensure the matrix is symmetric."""
        matrix = (matrix + matrix.T) / 2
        return matrix + reg_coef * np.eye(matrix.shape[0])

    def reset(self):
        """Reset the filter to initial state."""
        self.p_filtereds = []
        self.x_filtereds = []
        self.p_predicteds = []
        self.x_predicteds = []
        self.us = []
        self.ys = []
        self.y_filtereds = []
        self.y_predicteds = []
        self.kalman_gains = []

    def get_history(self):
        """Return the complete history of filter states."""
        return {
            'p_filtereds': self.p_filtereds,
            'x_filtereds': self.x_filtereds,
            'p_predicteds': self.p_predicteds,
            'x_predicteds': self.x_predicteds,
            'us': self.us,
            'ys': self.ys,
            'y_filtereds': self.y_filtereds,
            'y_predicteds': self.y_predicteds,
            'kalman_gains': self.kalman_gains
        }