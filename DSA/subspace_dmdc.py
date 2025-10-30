"""This module computes the subspace DMD with control (DMDc) model for a given dataset."""
import numpy as np
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
        
        # Smart device setup with graceful fallback
        self.device, self.use_torch = self._setup_device(device, True)
        
        # SubspaceDMDc specific attributes
        self.data = data
        self.control_data = control_data
        self.A_v = None
        self.B_v = None
        self.C_v = None
        self.info = None
        self.n_delays = n_delays
        self.rank = rank
        self.backend = backend


    def _to_torch(self, x):
        """Convert numpy array to torch tensor on the appropriate device."""
        if not self.use_torch or x is None:
            return x
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.from_numpy(x).to(self.device)
    
    def _to_numpy(self, x):
        """Convert torch tensor to numpy array."""
        if not self.use_torch or x is None:
            return x
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x
    
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



    def subspace_dmdc_multitrial_QR_decomposition(self, y_list, u_list, p, f, n=None, lamb=1e-8, energy=0.999):
        """
        Subspace-DMDc for multi-trial data with variable trial lengths.
        Now use QR decomposition for computing the oblique projection as in N4SID implementations.
        
        Parameters:
        - y_list: list of arrays, each (p_out, N_i) - output data for trial i
        - u_list: list of arrays, each (m, N_i) - input data for trial i
        - p: past window length
        - f: future window length
        - n: state dimension (auto-determined if None)
        - ridge: regularization parameter (used only for rank selection/SVD; QR is exact)
        - energy: energy threshold for rank selection
        
        Returns:
        - A_hat, B_hat, C_hat: system matrices
        - info: dictionary with additional information
        """
        if len(y_list) != len(u_list):
            raise ValueError("y_list and u_list must have same number of trials")
        
        n_trials = len(y_list)
        p_out = y_list[0].shape[0]
        m = u_list[0].shape[0]
        
        # Validate dimensions across trials
        for i, (y_trial, u_trial) in enumerate(zip(y_list, u_list)):
            if y_trial.shape[0] != p_out:
                raise ValueError(f"Trial {i}: y has {y_trial.shape[0]} outputs, expected {p_out}")
            if u_trial.shape[0] != m:
                raise ValueError(f"Trial {i}: u has {u_trial.shape[0]} inputs, expected {m}")
            if y_trial.shape[1] != u_trial.shape[1]:
                raise ValueError(f"Trial {i}: y and u have different time lengths")
        
        def hankel_stack(X, start, L):
            return np.concatenate([X[:, start + i:start + i + 1] for i in range(L)], axis=0)
        
        # Collect data from all trials
        U_p_all = []
        Y_p_all = []
        U_f_all = []
        Y_f_all = []
        valid_trials = []
        T_per_trial = []
        
        for trial_idx, (Y_trial, U_trial) in enumerate(zip(y_list, u_list)):
            N_trial = Y_trial.shape[1]
            T_trial = N_trial - (p + f) + 1
            
            if T_trial <= 0:
                print(f"Warning: Trial {trial_idx} has insufficient data (T={T_trial}), skipping")
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
            raise ValueError("No trials have sufficient data for given (p,f)")
        
        # Concatenate across valid trials
        U_p = np.concatenate(U_p_all, axis=1)  # (p m, T_total)
        Y_p = np.concatenate(Y_p_all, axis=1)  # (p p_out, T_total)
        U_f = np.concatenate(U_f_all, axis=1)  # (f m, T_total)
        Y_f = np.concatenate(Y_f_all, axis=1)  # (f p_out, T_total)
        
        T_total = sum(T_per_trial)
        Z_p = np.vstack([U_p, Y_p])  # (p (m + p_out), T_total)
        
        H = np.vstack([U_f, Z_p, Y_f])
        
        # Dimensions for slicing
        dim_uf = f * m
        dim_zp = p * (m + p_out)
        dim_yf = f * p_out
        
        # Use torch for expensive linear algebra if available
        if self.use_torch and H.shape[1] > 100:  # Only worth it for larger problems
            H_torch = self._to_torch(H)
            Z_p_torch = self._to_torch(Z_p)
            
            # Perform QR on H.T to get equivalent LQ on H
            Q, R_upper = torch.linalg.qr(H_torch.T, mode='reduced')
            L = R_upper.T
            
            # Extract submatrices from L
            R22 = L[dim_uf:dim_uf + dim_zp, dim_uf:dim_uf + dim_zp]
            R32 = L[dim_uf + dim_zp:, dim_uf:dim_uf + dim_zp]
            
            # Compute oblique projection O = R32 @ pinv(R22) @ Z_p
            R22_pinv = torch.linalg.pinv(R22)
            O = R32 @ R22_pinv @ Z_p_torch
            
            # SVD on O
            Uo, s, Vt = torch.linalg.svd(O, full_matrices=False)
            
            # Convert back to numpy
            Uo = self._to_numpy(Uo)
            s = self._to_numpy(s)
            Vt = self._to_numpy(Vt)
        else:
            # Use numpy for smaller problems or when torch is disabled
            # Perform QR on H.T to get equivalent LQ on H
            Q, R_upper = np.linalg.qr(H.T, mode='reduced')
            L = R_upper.T
            
            # Extract submatrices from L (lower triangular)
            R22 = L[dim_uf:dim_uf + dim_zp, dim_uf:dim_uf + dim_zp]
            R32 = L[dim_uf + dim_zp:, dim_uf:dim_uf + dim_zp]
            
            # Compute oblique projection O = R32 @ pinv(R22) @ Z_p
            O = R32 @ np.linalg.pinv(R22) @ Z_p
            
            # The rest remains the same: SVD on O
            Uo, s, Vt = np.linalg.svd(O, full_matrices=False)
        if n is None:
            cs = np.cumsum(s**2) / (s**2).sum()
            n = int(np.searchsorted(cs, energy) + 1)
            n = max(1, min(n, min(Uo.shape[1], Vt.shape[0])))
        
        U_n = Uo[:, :n]
        S_n = np.diag(s[:n])
        V_n = Vt[:n, :]
        S_half = np.sqrt(S_n)
        Gamma_hat = U_n @ S_half          # (f p_out, n)
        X_hat = S_half @ V_n              # (n, T_total)

        # Time alignment for regression across all trials
        # Need to handle variable lengths carefully
        X_segments = []
        X_next_segments = []
        U_mid_segments = []
        Y_segments = []
        
        start_idx = 0
        for trial_idx, T_trial in enumerate(T_per_trial):
            # Extract states for this trial
            X_trial = X_hat[:, start_idx:start_idx + T_trial]
            
            # State transitions within this trial
            X_trial_curr = X_trial[:, :-1]
            X_trial_next = X_trial[:, 1:]
            
            # Corresponding control inputs
            original_trial_idx = valid_trials[trial_idx]
            U_trial = u_list[original_trial_idx]
            U_mid_trial = U_trial[:, p:p + (T_trial - 1)]
            
            X_segments.append(X_trial_curr)
            X_next_segments.append(X_trial_next)
            U_mid_segments.append(U_mid_trial)
            
            # TODO: check the time-alignment of Y and X here
            # Corresponding output data - align with X_trial time indices
            Y_trial = y_list[original_trial_idx]
            Y_trial_curr = Y_trial[:, p:p+T_trial-1]
            # Y_trial_curr = Y_trial[:, p+1:p+T_trial]
            Y_segments.append(Y_trial_curr)

            start_idx += T_trial
        
        # Concatenate all segments
        X = np.concatenate(X_segments, axis=1)
        X_next = np.concatenate(X_next_segments, axis=1)
        U_mid = np.concatenate(U_mid_segments, axis=1)
        
        # Regression for A and B
        Z = np.vstack([X, U_mid])
        
        # Use torch for ridge regression if available
        if self.use_torch and Z.shape[1] > 100:
            Z_torch = self._to_torch(Z)
            X_next_torch = self._to_torch(X_next)
            
            # Ridge regression: (Z^T Z + λI)^(-1) Z^T X_next^T
            ZTZ = Z_torch @ Z_torch.T
            ridge_term = lamb * torch.eye(ZTZ.shape[0], device=self.device, dtype=Z_torch.dtype)
            AB = torch.linalg.solve(ZTZ + ridge_term, Z_torch @ X_next_torch.T).T
            
            AB = self._to_numpy(AB)
            A_hat = AB[:, :n]
            B_hat = AB[:, n:]
        else:
            # Ridge regression: (Z^T Z + λI)^(-1) Z^T X_next^T
            ZTZ = Z @ Z.T
            ridge_term = lamb * np.eye(ZTZ.shape[0])
            AB = np.linalg.solve(ZTZ + ridge_term, Z @ X_next.T).T
            A_hat = AB[:, :n]
            B_hat = AB[:, n:]

        # Z = np.vstack([X, U_mid])
        # AB = X_next @ np.linalg.pinv(Z)
        # A_hat = AB[:, :n]
        # B_hat = AB[:, n:]
        
        C_hat = Gamma_hat[:p_out, :]

        # Estimate noise covariance matrix
        # 0) Outputs aligned to X and U_mid (same time indices/columns)
        Y_curr = np.concatenate(Y_segments, axis=1)   # shape: (p_out, N)

        # 1) Residuals at time t
        #    Process noise residual (state eq): w_t ≈ x_{t+1} - A x_t - B u_ts
        W_hat = X_next - (A_hat @ X + B_hat @ U_mid)        # (n, N)

        #    Measurement noise residual (output eq): v_t ≈ y_t - C x_t  (since D = 0)
        V_hat = Y_curr - (C_hat @ X)                         # (p_out, N)

        # 2) Mean-centering
        V_hat = V_hat - V_hat.mean(axis=1, keepdims=True)
        W_hat = W_hat - W_hat.mean(axis=1, keepdims=True)
        N_res = V_hat.shape[1]
        denom = max(N_res - 1, 1)  

        # 3) Covariances
        R_hat = (V_hat @ V_hat.T) / denom                  # (p_out, p_out)   measurement
        Q_hat = (W_hat @ W_hat.T) / denom                  # (n, n)           process
        S_hat = (W_hat @ V_hat.T) / denom                  # (n, p_out) - cross (w,v)

        # 4) Symmetrize
        eps = 1e-12
        R_hat = 0.5 * (R_hat + R_hat.T) + eps * np.eye(R_hat.shape[0])
        Q_hat = 0.5 * (Q_hat + Q_hat.T) + eps * np.eye(Q_hat.shape[0])

        noise_covariance = np.block([[R_hat, S_hat.T],
                                     [S_hat, Q_hat]])

        info = {
            "singular_values_O": s, 
            "rank_used": n, 
            "Gamma_hat": Gamma_hat, 
            "f": f,
            "n_trials_total": n_trials,
            "n_trials_used": len(valid_trials),
            "valid_trials": valid_trials,
            "T_per_trial": T_per_trial,
            "T_total": T_total,
            "trial_lengths": [y.shape[1] for y in y_list],
            "noise_covariance": noise_covariance,
            'R_hat': R_hat,
            'Q_hat': Q_hat,
            'S_hat': S_hat
        }
        
        return A_hat, B_hat, C_hat, info
        


    
    def subspace_dmdc_multitrial_custom(self, y_list, u_list, p, f, n=None, lamb=1e-8, energy=0.999):
        """
        Subspace-DMDc for multi-trial data with variable trial lengths.
        
        Parameters:
        - y_list: list of arrays, each (p_out, N_i) - output data for trial i
        - u_list: list of arrays, each (m, N_i) - input data for trial i
        - p: past window length
        - f: future window length
        - n: state dimension (auto-determined if None)
        - ridge: regularization parameter
        - energy: energy threshold for rank selection∏
        
        Returns:
        - A_hat, B_hat, C_hat: system matrices
        - info: dictionary with additional information
        """
        if len(y_list) != len(u_list):
            raise ValueError("y_list and u_list must have same number of trials")
        
        n_trials = len(y_list)
        p_out = y_list[0].shape[0]
        m = u_list[0].shape[0]
        
        # Validate dimensions across trials

        for i, (y_trial, u_trial) in enumerate(zip(y_list, u_list)):
            if y_trial.shape[0] != p_out:
                raise ValueError(f"Trial {i}: y has {y_trial.shape[0]} outputs, expected {p_out}")
            if u_trial.shape[0] != m:
                raise ValueError(f"Trial {i}: u has {u_trial.shape[0]} inputs, expected {m}")
            if y_trial.shape[1] != u_trial.shape[1]:
                raise ValueError(f"Trial {i}: y and u have different time lengths")
        
        def hankel_stack(X, start, L):
            return np.concatenate([X[:, start + i:start + i + 1] for i in range(L)], axis=0)
        
        # Collect data from all trials
        U_p_all = []
        Y_p_all = []
        U_f_all = []
        Y_f_all = []
        valid_trials = []
        T_per_trial = []
        
        for trial_idx, (Y_trial, U_trial) in enumerate(zip(y_list, u_list)):
            N_trial = Y_trial.shape[1]
            T_trial = N_trial - (p + f) + 1
            
            if T_trial <= 0:
                print(f"Warning: Trial {trial_idx} has insufficient data (T={T_trial}), skipping")
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


        print("="*40)
        print(f"Number of valid trials: {len(U_p_trial)}")
        
        if not valid_trials:
            raise ValueError("No trials have sufficient data for given (p,f)")
        
        # Concatenate across valid trials
        U_p = np.concatenate(U_p_all, axis=1)  # (pm, T_total)
        Y_p = np.concatenate(Y_p_all, axis=1)  # (p*p_out, T_total)
        U_f = np.concatenate(U_f_all, axis=1)  # (fm, T_total)
        Y_f = np.concatenate(Y_f_all, axis=1)  # (f*p_out, T_total)
        
        T_total = sum(T_per_trial)
        Z_p = np.vstack([U_p, Y_p])  # (p(m+p_out), T_total)
        
        # Oblique projection: remove row(U_f), project onto row(Z_p)
        UfUfT = U_f @ U_f.T
        Xsolve = np.linalg.solve(UfUfT + lamb*np.eye(UfUfT.shape[0]), U_f)
        Pi_perp = np.eye(T_total) - U_f.T @ Xsolve
        Yf_perp = Y_f @ Pi_perp
        Zp_perp = Z_p @ Pi_perp
        
        ZZT = Zp_perp @ Zp_perp.T
        Zp_pinv_left = np.linalg.solve(ZZT + lamb*np.eye(ZZT.shape[0]), Zp_perp)
        P = Zp_perp.T @ Zp_pinv_left
        O = Yf_perp @ P  # ≈ Γ_f X_p
        
        Uo, s, Vt = np.linalg.svd(O, full_matrices=False)
        if n is None:
            cs = np.cumsum(s**2) / (s**2).sum()
            n = int(np.searchsorted(cs, energy) + 1)
            n = max(1, min(n, min(Uo.shape[1], Vt.shape[0])))
        
        U_n = Uo[:, :n]
        S_n = np.diag(s[:n])
        V_n = Vt[:n, :]
        S_half = np.sqrt(S_n)
        Gamma_hat = U_n @ S_half          # (f*p_out, n)
        X_hat = S_half @ V_n              # (n, T_total)
        
        # Time alignment for regression across all trials
        # Need to handle variable lengths carefully
        X_segments = []
        X_next_segments = []
        U_mid_segments = []
        
        start_idx = 0
        for trial_idx, T_trial in enumerate(T_per_trial):
            # Extract states for this trial
            X_trial = X_hat[:, start_idx:start_idx + T_trial]
            
            # State transitions within this trial
            X_trial_curr = X_trial[:, :-1]
            X_trial_next = X_trial[:, 1:]
            
            # Corresponding control inputs
            original_trial_idx = valid_trials[trial_idx]
            U_trial = u_list[original_trial_idx]
            U_mid_trial = U_trial[:, p:p + (T_trial - 1)]
            
            X_segments.append(X_trial_curr)
            X_next_segments.append(X_trial_next)
            U_mid_segments.append(U_mid_trial)
            
            start_idx += T_trial
        
        # Concatenate all segments
        X = np.concatenate(X_segments, axis=1)
        X_next = np.concatenate(X_next_segments, axis=1)
        U_mid = np.concatenate(U_mid_segments, axis=1)
        
        # Regression for A and B
        Z = np.vstack([X, U_mid])
        # Ridge regression: (Z^T Z + λI)^(-1) Z^T X_next^T
        ZTZ = Z @ Z.T
        ridge_term = lamb * np.eye(ZTZ.shape[0])
        AB = np.linalg.solve(ZTZ + ridge_term, Z @ X_next.T).T
        A_hat = AB[:, :n]
        B_hat = AB[:, n:]
        
        C_hat = Gamma_hat[:p_out, :]
        
        info = {
            "singular_values_O": s, 
            "rank_used": n, 
            "Gamma_hat": Gamma_hat, 
            "f": f,
            "n_trials_total": n_trials,
            "n_trials_used": len(valid_trials),
            "valid_trials": valid_trials,
            "T_per_trial": T_per_trial,
            "T_total": T_total,
            "trial_lengths": [y.shape[1] for y in y_list],
            "X_hat": X_hat
        }
        
        return A_hat, B_hat, C_hat, info



    def subspace_dmdc_multitrial_flexible(self, y, u, p, f, n=None, lamb=1e-8, energy=0.999, backend='n4sid'):
        """
        Flexible wrapper that handles both fixed-length and variable-length multi-trial data.
        
        Parameters:
        - y: either (n_trials, p_out, N) array, (p_out, N) array, or list of (p_out, N_i) arrays
        - u: either (n_trials, m, N) array, (m, N) array, or list of (m, N_i) arrays
        """
        if isinstance(y, list) and isinstance(u, list):
            y_list = [y_trial.T for y_trial in y]
            u_list = [u_trial.T for u_trial in u]
            if backend == 'n4sid':
                return self.subspace_dmdc_multitrial_QR_decomposition(y_list, u_list, p, f, n, lamb, energy)
            else:
                return self.subspace_dmdc_multitrial_custom(y_list, u_list, p, f, n, lamb, energy)
        
        else:
            # Handle 2D arrays (single trial) by converting to list format
            if y.ndim == 2:
                y_list = [y]
                u_list = [u]
            else:
                # Convert 3D arrays to list format
                y_list = [y[i] for i in range(y.shape[0])]
                u_list = [u[i] for i in range(u.shape[0])]
            
            y_list = [y_trial.T for y_trial in y_list]
            u_list = [u_trial.T for u_trial in u_list]
            
            if backend == 'n4sid':
               return self.subspace_dmdc_multitrial_QR_decomposition(y_list, u_list, p, f, n, lamb, energy)
            else:
                return self.subspace_dmdc_multitrial_custom(y_list, u_list, p, f, n, lamb, energy)


    def predict(self, Y, U, reseed=None):
        # Y and U are (n_times, n_channels) or list of 2D arrays
        if reseed is None:
            reseed = 1

        # Handle list of 2D arrays
        if isinstance(Y, list):
            
            self.kalman = OnlineKalman(self)
            Y_pred = []
            for trial in range(len(Y)):
                self.kalman.reset()  # Reset filter for each trial
                trial_predictions = []
                for t in range(Y[trial].shape[0]):
                    y_filtered, _ = self.kalman.step(
                        y=Y[trial][t] if t%reseed == 0 else None, 
                        u=U[trial][t]
                    )
                    trial_predictions.append(y_filtered)
                Y_pred.append(np.concatenate(trial_predictions, axis=1).T)
            return Y_pred  # Return as list to match input format

            
        self.kalman = OnlineKalman(self)
        if Y.ndim == 2:
            Y_pred = []
            for t in range(Y.shape[0]):
                y_filtered, _ = self.kalman.step(y=Y[t] if t%reseed == 0 else None, u=U[t])
                Y_pred.append(y_filtered)
            return np.concatenate(Y_pred, axis=1).T
        else:
            # 3D data (n_trials, time, p_out)
            # print("Y.shape", Y.shape)
            # print("U.shape", U.shape)
            Y_pred = []
            for trial in range(Y.shape[0]):
                self.kalman.reset()  # Reset filter for each trial
                trial_predictions = []
                for t in range(Y.shape[1]):
                    y_filtered, _ = self.kalman.step(y=Y[trial, t] if t%reseed == 0 else None, u=U[trial, t])
                    trial_predictions.append(y_filtered)
                    # print("y_filtered.shape", y_filtered.shape)
                Y_pred.append(np.concatenate(trial_predictions, axis=1).T)
            return np.array(Y_pred)
 
    def compute_hankel(self, *args, **kwargs):
        """
        Compute Hankel matrices for SubspaceDMDc.
        
        This is handled internally within subspace_dmdc_multitrial_QR_decomposition
        and subspace_dmdc_multitrial_custom methods.
        """
        raise NotImplementedError(
            "Hankel matrix computation is integrated into the fit() method for SubspaceDMDc. "
            "Use fit() to compute the model."
        )
    
    def compute_svd(self, *args, **kwargs):
        """
        Compute SVD for SubspaceDMDc.
        
        This is handled internally within the subspace identification process.
        """
        raise NotImplementedError(
            "SVD computation is integrated into the fit() method for SubspaceDMDc. "
            "Use fit() to compute the model."
        )


class OnlineKalman:
    """
    Online Kalman Filter class for real-time state estimation.
    
    This class maintains the internal state of the Kalman filter and provides
    a step method for updating the filter with new observations and inputs.
    """
    
    def __init__(self, dmdc):
        """
        Initialize the Online Kalman Filter with a fitted DMDc model.
        
        Parameters
        ----------
        dmdc : object
            Fitted DMDc model containing A_v, B_v, C_v matrices and 
            noise covariance estimates (R_hat, S_hat, Q_hat)
        """
        self.A = dmdc.A_v
        self.B = dmdc.B_v  
        self.C = dmdc.C_v
        self.R = dmdc.info['R_hat']
        self.S = dmdc.info['S_hat']
        self.Q = dmdc.info['Q_hat']
        
        # Get dimensions
        # print("C_shape", self.C.shape)
        self.y_dim, self.x_dim = self.C.shape
        
        # Initialize state storage
        self.p_filtereds = []
        self.x_filtereds = []
        self.p_predicteds = []
        self.x_predicteds = []
        self.us = []
        self.ys = []
        self.y_filtereds = []
        self.y_predicteds = []
        self.kalman_gains = []

        
    # def step(self, y=None, u=None, lam=1e-8):
    #     """
    #     Perform one step of the Kalman filter.
        
    #     Parameters
    #     ----------
    #     y : np.ndarray, optional
    #         Observed output at current time step. If None, the filter
    #         will predict without observation update.
    #     u : np.ndarray, optional  
    #         Input at current time step. If None, no input is applied.
            
    #     Returns
    #     -------
    #     y_filtered : np.ndarray
    #         Filtered output estimate
    #     x_filtered : np.ndarray  
    #         Filtered state estimate
    #     """
    #     x_pred = self.x_predicteds[-1] if self.x_predicteds else np.zeros((self.x_dim, 1))
    #     p_pred = self.p_predicteds[-1] if self.p_predicteds else np.eye(self.x_dim)

    #     # Ensure inputs are column vectors
    #     if u is not None and u.ndim == 1:
    #         u = u.reshape(-1, 1)
    #     if y is not None and y.ndim == 1:
    #         y = y.reshape(-1, 1)
    #     if u is None:
    #         u = np.zeros((self.u_dim, 1))
    #     if y is None:
    #         y = np.zeros((self.y_dim, 1))

    #     S_innov = self.R + self.C @ p_pred @ self.C.T
    #     K_filtered = p_pred @ self.C.T @ np.linalg.pinv(S_innov)
    #     p_filtered = p_pred - K_filtered @ self.C @ p_pred
    #     if not np.isnan(y).any():
    #         x_filtered = x_pred + K_filtered @ (y - self.C @ x_pred)
    #     else:
    #         x_filtered = x_pred.copy()
            
    #     K_pred = (self.S + self.A @ p_pred @ self.C.T) @ np.linalg.pinv(S_innov)
    #     p_predicted = (self.A @ p_pred @ self.A.T + self.Q - 
    #                   K_pred @ (self.S + self.A @ p_pred @ self.C.T).T)
    #     x_predicted = self.A @ x_pred + self.B @ u
    #     if not np.isnan(y).any():
    #         x_predicted += K_pred @ (y - self.C @ x_pred)
            
    #     # Store results
    #     self.p_filtereds.append(p_filtered)
    #     self.x_filtereds.append(x_filtered)
    #     self.p_predicteds.append(p_predicted)
    #     self.x_predicteds.append(x_predicted)
    #     self.us.append(u)
    #     self.ys.append(y)
    #     self.y_filtereds.append(self.C @ x_filtered)
    #     self.y_predicteds.append(self.C @ x_predicted)
    #     self.kalman_gains.append(K_pred)
        
    #     return self.y_filtereds[-1], self.x_filtereds[-1]


    def step(self, y=None, u=None, reg_coef=1e-6):
        """
        Perform one step of the Kalman filter.
        
        Parameters
        ----------
        y : np.ndarray, optional
            Observed output at current time step. If None, the filter
            will predict without observation update.
        u : np.ndarray, optional  
            Input at current time step. If None, no input is applied.
        reg_coef : float, optional
            Regularization coefficient to add to diagonal of P matrices
            to maintain numerical stability. Default: 1e-6
            
        Returns
        -------
        y_filtered : np.ndarray
            Filtered output estimate
        x_filtered : np.ndarray  
            Filtered state estimate
        """
        x_pred = self.x_predicteds[-1] if self.x_predicteds else np.zeros((self.x_dim, 1))
        p_pred = self.p_predicteds[-1] if self.p_predicteds else np.eye(self.x_dim)
        
        # Add regularization to p_pred to prevent ill-conditioning
        p_pred_reg = p_pred + reg_coef * np.eye(self.x_dim)

        # Ensure inputs are column vectors
        if u is not None and u.ndim == 1:
            u = u.reshape(-1, 1)
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)
        if u is None:
            u = np.zeros((self.u_dim, 1))
        if y is None:
            y = np.zeros((self.y_dim, 1))

        # Use regularized p_pred in computations
        S_innov = self.R + self.C @ p_pred_reg @ self.C.T
        K_filtered = p_pred_reg @ self.C.T @ np.linalg.pinv(S_innov)
        p_filtered = p_pred_reg - K_filtered @ self.C @ p_pred_reg
        
        # Add regularization to p_filtered to maintain positive definiteness
        p_filtered = (p_filtered + p_filtered.T) / 2  # Ensure symmetry
        p_filtered = p_filtered + reg_coef * np.eye(self.x_dim)  # Add regularization
        
        if not np.isnan(y).any():
            x_filtered = x_pred + K_filtered @ (y - self.C @ x_pred)
        else:
            x_filtered = x_pred.copy()
            
        K_pred = (self.S + self.A @ p_pred_reg @ self.C.T) @ np.linalg.pinv(S_innov)
        p_predicted = (self.A @ p_pred_reg @ self.A.T + self.Q - 
                    K_pred @ (self.S + self.A @ p_pred_reg @ self.C.T).T)
        
        # Add regularization to p_predicted and ensure symmetry
        p_predicted = (p_predicted + p_predicted.T) / 2  # Ensure symmetry
        p_predicted = p_predicted + reg_coef * np.eye(self.x_dim)  # Add regularization
        
        x_predicted = self.A @ x_pred + self.B @ u
        if not np.isnan(y).any():
            x_predicted += K_pred @ (y - self.C @ x_pred)
            
        # Store results
        self.p_filtereds.append(p_filtered)
        self.x_filtereds.append(x_filtered)
        self.p_predicteds.append(p_predicted)
        self.x_predicteds.append(x_predicted)
        self.us.append(u)
        self.ys.append(y)
        self.y_filtereds.append(self.C @ x_filtered)
        self.y_predicteds.append(self.C @ x_predicted)
        self.kalman_gains.append(K_pred)
        
        return self.y_filtereds[-1], self.x_filtereds[-1]
        
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