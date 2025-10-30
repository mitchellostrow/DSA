"""This module computes the DMD with control (DMDc) model for a given dataset."""

import numpy as np
import torch

try:
    from .dmd import embed_signal_torch
    from .base_dmd import BaseDMD
except ImportError:
    from dmd import embed_signal_torch
    from base_dmd import BaseDMD


def embed_data_DMDc(
    data, n_delays=1, n_control_delays=1, delay_interval=1, control=False
):
    if control:
        if n_control_delays == 1:
            if data.ndim == 2:
                return data[(n_delays - 1) * delay_interval :, :]
            else:
                return data[:, (n_delays - 1) * delay_interval :, :]
        else:
            embedded_data = embed_signal_torch(data, n_control_delays, delay_interval)
            return embedded_data
    else:
        return embed_signal_torch(data, n_delays, delay_interval)


class DMDc(BaseDMD):
    """DMDc class for computing and predicting with DMD with control models."""

    def __init__(
        self,
        data,
        control_data=None,
        n_delays=1,
        n_control_delays=1,
        delay_interval=1,
        rank_input=None,
        rank_thresh_input=None,
        rank_explained_variance_input=None,
        rank_output=None,
        rank_thresh_output=None,
        rank_explained_variance_output=None,
        lamb=1e-8,
        device="cpu",
        verbose=False,
        send_to_cpu=False,
        svd_separate=True,
        steps_ahead=1,
    ):
        """
        Parameters
        ----------
        data : np.ndarray or torch.tensor
            The state data to fit the DMDc model to. Must be either: (1) a
            2-dimensional array/tensor of shape T x N where T is the number
            of time points and N is the number of observed dimensions
            at each time point, or (2) a 3-dimensional array/tensor of shape
            K x T x N where K is the number of "trials" and T and N are
            as defined above.

        control_data : np.ndarray or torch.tensor
            The control input data corresponding to the state data. Must have compatible dimensions
            with the state data.

        n_delays : int
            Parameter that controls the size of the delay embedding. Explicitly,
            the number of delays to include.

        delay_interval : int
            The number of time steps between each delay in the delay embedding. Defaults
            to 1 time step.

        rank : int
            The rank of V in fitting DMDc - i.e., the number of columns of V to
            use to fit the DMDc model. Defaults to None, in which case all columns of V
            will be used.

        rank_thresh : float
            Parameter that controls the rank of V in fitting DMDc by dictating a threshold
            of singular values to use. Explicitly, the rank of V will be the number of singular
            values greater than rank_thresh. Defaults to None.

        rank_explained_variance : float
            Parameter that controls the rank of V in fitting DMDc by indicating the percentage of
            cumulative explained variance that should be explained by the columns of V. Defaults to None.

        lamb : float
            Regularization parameter for ridge regression. Defaults to 0.

        device: string, int, or torch.device
            Device for computation. Options:
            - 'cpu': Use CPU with PyTorch
            - 'cuda' or 'cuda:X': Use GPU (auto-falls back to CPU if unavailable)

        verbose: bool
            If True, print statements will be provided about the progress of the fitting procedure.

        send_to_cpu: bool
            If True, will send all tensors in the object back to the cpu after everything is computed.
            This is implemented to prevent gpu memory overload when computing multiple DMDs.

        steps_ahead: int
            The number of time steps ahead to predict. Defaults to 1.
        """

        super().__init__(
            device=device, verbose=verbose, send_to_cpu=send_to_cpu, lamb=lamb
        )
        
        # Smart device setup with graceful CUDA fallback
        # DMDc always uses PyTorch, so use_torch=True
        self.device, self.use_torch = self._setup_device(device, use_torch=True)

        self._init_data(data, control_data)
        self._check_same_shape()

        self.n_delays = n_delays
        self.n_control_delays = n_control_delays
        self.delay_interval = delay_interval

        self.rank_input = rank_input
        self.rank_thresh_input = rank_thresh_input
        self.rank_explained_variance_input = rank_explained_variance_input
        self.rank_output = rank_output
        self.rank_thresh_output = rank_thresh_output
        self.rank_explained_variance_output = rank_explained_variance_output
        self.svd_separate = (
            svd_separate  # do svd on H and u separately as well as regression
        )
        self.steps_ahead = steps_ahead

        # Hankel matrix
        self.H = None

        # Control input Hankel matrix
        self.Hu = None

        # SVD attributes
        self.U = None
        self.S = None
        self.V = None
        self.S_mat = None
        self.S_mat_inv = None

        # Change of basis between the reduced-order subspace and the full space
        self.U_out = None
        self.S_out = None
        self.V_out = None

        # DMDc attributes
        self.A_tilde = None
        self.B_tilde = None
        self.A = None
        self.B = None
        self.A_havok_dmd = None
        self.B_havok_dmd = None

        # Check if the state and control data are list (for different trial lengths)
        if not np.all([isinstance(data, list), isinstance(control_data, list)]):
            if isinstance(data, list) or isinstance(control_data, list):
                raise TypeError(
                    "If you pass one of (data, control_data) as list, the other must also be a list."
                )

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

    def compute_hankel(
        self,
        data=None,
        control_data=None,
        n_delays=None,
        delay_interval=None,
    ):
        """
        Computes the Hankel matrix from the provided data and forms Omega.
        """
        if self.verbose:
            print("Computing Hankel matrices ...")

        # Overwrite parameters if provided
        self.data = self.data if data is None else self._init_data(data, control_data)
        self.n_delays = self.n_delays if n_delays is None else n_delays
        self.delay_interval = (
            self.delay_interval if delay_interval is None else delay_interval
        )

        if self.is_list_data:
            self.data = [d.to(self.device) for d in self.data]
            self.control_data = [d.to(self.device) for d in self.control_data]
            # Compute Hankel matrices for each trial separately
            self.H = [
                embed_data_DMDc(
                    d,
                    n_delays=self.n_delays,
                    n_control_delays=self.n_control_delays,
                    delay_interval=self.delay_interval,
                ).float()
                for d in self.data
            ]
            self.Hu = [
                embed_data_DMDc(
                    d,
                    n_delays=self.n_delays,
                    n_control_delays=self.n_control_delays,
                    delay_interval=self.delay_interval,
                    control=True,
                ).float()
                for d in self.control_data
            ]
            self.H_shapes = [h.shape for h in self.H]
        else:
            self.data = self.data.to(self.device)
            self.control_data = self.control_data.to(self.device)
            # Compute Hankel matrices
            self.H = embed_data_DMDc(
                self.data,
                n_delays=self.n_delays,
                n_control_delays=self.n_control_delays,
                delay_interval=self.delay_interval,
            ).float()
            self.Hu = embed_data_DMDc(
                self.control_data,
                n_delays=self.n_delays,
                n_control_delays=self.n_control_delays,
                delay_interval=self.delay_interval,
                control=True,
            ).float()

        if self.verbose:
            print("Hankel matrices computed!")

    def compute_svd(self):
        """
        Computes the SVD of the Omega and Y matrices.
        """
        if self.verbose:
            print("Computing SVD on H and U matrices ...")

        if self.is_list_data:
            self.H_shapes = [h.shape for h in self.H]
            H_list = []
            Hu_list = []
            for h_elem in self.H:
                if h_elem.ndim == 3:
                    H_list.append(
                        h_elem.reshape(
                            h_elem.shape[0] * h_elem.shape[1], h_elem.shape[2]
                        )
                    )
                else:
                    H_list.append(h_elem)

            for hu_elem in self.Hu:
                if hu_elem.ndim == 3:
                    Hu_list.append(
                        hu_elem.reshape(
                            hu_elem.shape[0] * hu_elem.shape[1], hu_elem.shape[2]
                        )
                    )
                else:
                    Hu_list.append(hu_elem)
            self.H = torch.cat(H_list, dim=0)
            self.Hu = torch.cat(Hu_list, dim=0)
            # H = torch.cat(H_list, dim=0)
            self.H_row_counts = [h.shape[0] for h in H_list]
            H = self.H
            Hu = self.Hu

        elif self.H.ndim == 3:  # flatten across trials for 3d
            H = self.H.reshape(self.H.shape[0] * self.H.shape[1], self.H.shape[2])
            Hu = self.Hu.reshape(self.Hu.shape[0] * self.Hu.shape[1], self.Hu.shape[2])
        else:
            H = self.H
            Hu = self.Hu
        self.Uh, self.Sh, self.Vh = torch.linalg.svd(H.T, full_matrices=False)
        self.Uu, self.Su, self.Vu = torch.linalg.svd(Hu.T, full_matrices=False)

        self.Vh = self.Vh.T
        self.Vu = self.Vu.T

        self.Sh_mat = torch.diag(self.Sh).to(self.device)
        self.Sh_mat_inv = torch.diag(1 / self.Sh).to(self.device)

        self.Su_mat = torch.diag(self.Su).to(self.device)
        self.Su_mat_inv = torch.diag(1 / self.Su).to(self.device)

        self.cumulative_explained_variance_input = self._compute_explained_variance(
            self.Su
        )
        self.cumulative_explained_variance_output = self._compute_explained_variance(
            self.Sh
        )

        self.Vht_minus, self.Vht_plus = self.get_plus_minus(self.Vh, self.H)
        self.Vut_minus, _ = self.get_plus_minus(self.Vu, self.Hu)

        if self.verbose:
            print("SVDs computed!")

    def get_plus_minus(self, V, H):
        if self.ntrials > 1:
            if self.is_list_data:
                V_split = torch.split(V, self.H_row_counts, dim=0)
                Vt_minus_list, Vt_plus_list = [], []
                for v_part, h_shape in zip(V_split, self.H_shapes):
                    if len(h_shape) == 3:  # Has trials
                        v_part_reshaped = v_part.reshape(h_shape)
                        newshape = (
                            h_shape[0] * (h_shape[1] - self.steps_ahead),
                            h_shape[2],
                        )
                        Vt_minus_list.append(
                            v_part_reshaped[:, : -self.steps_ahead].reshape(newshape)
                        )
                        Vt_plus_list.append(
                            v_part_reshaped[:, self.steps_ahead :].reshape(newshape)
                        )
                    else:  # No trials, just time and features
                        Vt_minus_list.append(v_part[: -self.steps_ahead])
                        Vt_plus_list.append(v_part[self.steps_ahead :])

                Vt_minus = torch.cat(Vt_minus_list, dim=0)
                Vt_plus = torch.cat(Vt_plus_list, dim=0)
            else:

                if V.numel() < H.numel():
                    raise ValueError(
                        "The dimension of the SVD of the Hankel matrix is smaller than the dimension of the Hankel matrix itself. \n \
                                    This is likely due to the number of time points being smaller than the number of dimensions. \n \
                                    Please reduce the number of delays."
                    )

                V = V.reshape(H.shape)

                # first reshape back into Hankel shape, separated by trials
                newshape = (
                    H.shape[0] * (H.shape[1] - self.steps_ahead),
                    H.shape[2],
                )
                Vt_minus = V[:, : -self.steps_ahead].reshape(newshape)
                Vt_plus = V[:, self.steps_ahead :].reshape(newshape)
        else:
            Vt_minus = V[: -self.steps_ahead]
            Vt_plus = V[self.steps_ahead :]

        return Vt_minus, Vt_plus

    def recalc_rank(
        self,
        rank_input=None,
        rank_thresh_input=None,
        rank_explained_variance_input=None,
        rank_output=None,
        rank_thresh_output=None,
        rank_explained_variance_output=None,
    ):
        """
        Recalculates the rank for input and output based on provided parameters.
        """
        # Recalculate ranks for input
        self.rank_input = self._compute_rank_from_params(
            S=self.Su,
            cumulative_explained_variance=self.cumulative_explained_variance_input,
            max_rank=self.Hu.shape[-1],
            rank=rank_input,
            rank_thresh=rank_thresh_input,
            rank_explained_variance=rank_explained_variance_input,
        )
        # Recalculate ranks for output
        self.rank_output = self._compute_rank_from_params(
            S=self.Sh,
            cumulative_explained_variance=self.cumulative_explained_variance_output,
            max_rank=self.H.shape[-1],
            rank=rank_output,
            rank_thresh=rank_thresh_output,
            rank_explained_variance=rank_explained_variance_output,
        )

    def compute_dmdc(self, lamb=None):
        if self.verbose:
            print("Computing DMDc matrices ...")

        self.lamb = self.lamb if lamb is None else lamb

        V_minus_tot = torch.cat(
            [
                self.Vht_minus[:, : self.rank_output],
                self.Vut_minus[:, : self.rank_input],
            ],
            dim=1,
        )

        A_v_tot = (
            torch.linalg.inv(
                V_minus_tot.T @ V_minus_tot
                + self.lamb * torch.eye(V_minus_tot.shape[1]).to(self.device)
            )
            @ V_minus_tot.T
            @ self.Vht_plus[:, : self.rank_output]
        ).T
        # split A_v_tot into A_v and B_v
        self.A_v = A_v_tot[:, : self.rank_output]
        self.B_v = A_v_tot[:, self.rank_output :]
        self.A_havok_dmd = (
            self.Uh
            @ self.Sh_mat[: self.Uh.shape[1], : self.rank_output]
            @ self.A_v
            @ self.Sh_mat_inv[: self.rank_output, : self.Uh.shape[1]]
            @ self.Uh.T
        )

        self.B_havok_dmd = (
            self.Uh
            @ self.Sh_mat[: self.Uh.shape[1], : self.rank_output]
            @ self.B_v
            @ self.Su_mat_inv[: self.rank_input, : self.Uu.shape[1]]
            @ self.Uu.T
        )

        # Set the A and B properties for backward compatibility and easier access
        self.A = self.A_havok_dmd
        self.B = self.B_havok_dmd

        if self.verbose:
            print("DMDc matrices computed!")

    def fit(
        self,
        data=None,
        control_data=None,
        n_delays=None,
        delay_interval=None,
        lamb=None,
        device=None,
        verbose=None,
    ):
        """
        Fits the DMDc model to the provided data.
        """
        # Overwrite parameters if provided
        self.device = self.device if device is None else device
        self.verbose = self.verbose if verbose is None else verbose

        self.compute_hankel(data, control_data, n_delays, delay_interval)
        self.compute_svd()
        self.recalc_rank(
            self.rank_input,
            self.rank_thresh_input,
            self.rank_explained_variance_input,
            self.rank_output,
            self.rank_thresh_output,
            self.rank_explained_variance_output,
        )
        self.compute_dmdc(lamb)
        if self.send_to_cpu:
            self.all_to_device("cpu")  # send back to the cpu to save memory

    def predict(
        self, test_data=None, control_data=None, reseed=None, full_return=False
    ):
        """
        Parameters
        ----------
        test_data : np.ndarray or torch.tensor
            The state data to make predictions on.

        control_data : np.ndarray or torch.tensor
            The control input data corresponding to the test_data.

        reseed : int
            Frequency of reseeding the prediction with true data.

        full_return : bool
            If True, returns additional matrices used in prediction.

        Returns
        -------
        pred_data : torch.tensor
            The predictions generated by the DMDc model. Of the same shape as test_data.

        H_test_dmdc : torch.tensor (Optional)
            Returned if full_return=True. The predicted Hankel matrix generated by the DMDc model.

        H_test : torch.tensor (Optional)
            Returned if full_return=True. The true Hankel matrix.
        """
        # Initialize test_data
        if test_data is None:
            test_data = self.data
        if control_data is None:
            control_data = self.control_data

        if isinstance(test_data, list):
            predictions = [
                self.predict(
                    test_data=d,
                    control_data=d_control,
                    reseed=reseed,
                    full_return=full_return,
                )
                for d, d_control in zip(test_data, control_data)
            ]
            if full_return:
                pred_data = [pred[0] for pred in predictions]
                H_test_dmdc = [pred[1] for pred in predictions]
                H_test = [pred[2] for pred in predictions]
                return pred_data, H_test_dmdc, H_test
            else:
                return predictions

        if isinstance(test_data, np.ndarray):
            test_data = torch.from_numpy(test_data).to(self.device)
        if isinstance(control_data, np.ndarray):
            control_data = torch.from_numpy(control_data).to(self.device)

        ndim = test_data.ndim
        if ndim == 2:
            test_data = test_data.unsqueeze(0)
            control_data = control_data.unsqueeze(0)
        # H_test = embed_data_DMDc(test_data, n_delays=self.n_delays, delay_interval=self.delay_interval, control=False)
        # H_control = embed_data_DMDc(control_data, n_delays=self.n_control_delays, delay_interval=self.delay_interval, control=True)
        H_test = embed_signal_torch(
            test_data, self.n_delays, self.delay_interval
        ).float()
        H_control = embed_signal_torch(
            control_data, self.n_control_delays, self.delay_interval
        ).float()
        if reseed is None:
            reseed = 1

        H_test_dmdc = torch.zeros_like(H_test).to(self.device)
        H_test_dmdc[:, 0] = H_test[:, 0]
        A = self.A_havok_dmd
        B = self.B_havok_dmd

        for t in range(1, H_test.shape[1]):
            u_t = H_control[:, t - 1]
            # print(A.shape)
            # print(H_test[:, t - 1].shape)
            # print(B.shape)
            # print(u_t.shape)
            if t % reseed == 0:
                H_test_dmdc[:, t] = (A @ H_test[:, t - 1].transpose(-2, -1)).transpose(
                    -2, -1
                ) + (B @ u_t.transpose(-2, -1)).transpose(-2, -1)
            else:
                H_test_dmdc[:, t] = (
                    A @ H_test_dmdc[:, t - 1].transpose(-2, -1)
                ).transpose(-2, -1) + (B @ u_t.transpose(-2, -1)).transpose(-2, -1)
        pred_data = torch.hstack(
            [
                test_data[
                    :, : (self.n_delays - 1) * self.delay_interval + self.steps_ahead
                ],
                H_test_dmdc[:, self.steps_ahead :, : self.n],
            ]
        )

        if ndim == 2:
            pred_data = pred_data[0]

        if full_return:
            return pred_data, H_test_dmdc, H_test
        else:
            return pred_data
