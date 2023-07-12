"""This module computes the Havok DMD model for a given dataset."""
import numpy as np
import torch
import scipy
import tqdm

def embed_signal_torch(data, n_delays, delay_interval=1):
    """
    Create a delay embedding from the provided tensor x.

    Parameters
    ----------
    data : torch.tensor
        The data from which to create the delay embedding. Must be either: (1) a
        2-dimensional array/tensor of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    n_delays : int
        Parameter that controls the size of the delay embedding. Explicitly,
        the number of delays to include.

    delay_interval : int
        The number of time steps between each delay in the delay embedding. Defaults
        to 1 time step.
    """
    ndims = len(data.shape)
    device = data.device

    # initialize the embedding
    if ndims == 3:
        embedding = torch.zeros((data.shape[0], data.shape[1] - (n_delays - 1)*delay_interval, data.shape[2]*n_delays)).to(device)
    else:
        embedding = torch.zeros((data.shape[0] - (n_delays - 1)*delay_interval, data.shape[1]*n_delays)).to(device)
    for d in range(n_delays):
        if ndims == 3:
            embedding[:,:, d*data.shape[2]:(d + 1)*data.shape[2]] = data[:,(n_delays - 1 - d)*delay_interval:data.shape[1] - d*delay_interval]
        else:
            embedding[:, d*data.shape[1]:(d + 1)*data.shape[1]] = data[(n_delays - 1 - d)*delay_interval:data.shape[0] - d*delay_interval]
    
    return embedding

class DMD:
    """DMD class for computing and predicting with DMD models.
    """
    def __init__(
            self,
            data,
            n_delays,
            delay_interval=1,
            rank=None,
            rank_thresh=None,
            lamb=0,
            device='cpu',
            verbose=False,
        ):
        """
        Parameters
        ----------
        data : np.ndarray or torch.tensor
            The data to fit the DMD model to. Must be either: (1) a
            2-dimensional array/tensor of shape T x N where T is the number
            of time points and N is the number of observed dimensions
            at each time point, or (2) a 3-dimensional array/tensor of shape
            K x T x N where K is the number of "trials" and T and N are
            as defined above.

        n_delays : int
            Parameter that controls the size of the delay embedding. Explicitly,
            the number of delays to include.

        delay_interval : int
            The number of time steps between each delay in the delay embedding. Defaults
            to 1 time step.

        rank : int
            The rank of V in fitting HAVOK DMD - i.e., the number of columns of V to 
            use to fit the DMD model. Defaults to None, in which case all columns of V
            will be used.

        rank_thresh : int
            Parameter that controls the rank of V in fitting HAVOK DMD by dictating a threshold
            of singular values to use. Explicitly, the rank of V will be the number of singular
            values greater than rank_thresh. Defaults to None.

        lamb : float
            Regularization parameter for ridge regression. Defaults to 0.

        device: string, int, or torch.device
            A string, int or torch.device object to indicate the device to torch.

        verbose: bool
            If True, print statements will be provided about the progress of the fitting procedure.
        """

        self.device = device
        self._init_data(data)

        self.n_delays = n_delays
        self.delay_interval = delay_interval
        self.rank = rank
        self.rank_thresh = rank_thresh
        self.lamb = lamb
        self.verbose = verbose
        
        # Hankel matrix
        self.H = None

        # SVD attributes
        self.U = None
        self.S = None
        self.V = None
        self.S_mat = None
        self.S_mat_inv = None
        
        # DMD attributes
        self.A = None
        self.A_v = None
        self.A_havok_dmd = None

    def _init_data(self, data):
        # check if the data is an np.ndarry - if so, convert it to Torch
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        # move the data to the appropriate device
        self.data = data.to(self.device)

        # create attributes for the data dimensions
        if len(self.data.shape) == 3:
            self.ntrials = self.data.shape[0]
            self.window = self.data.shape[1]
            self.n = self.data.shape[2]
        else:
            self.window = self.data.shape[0]
            self.n = self.data.shape[1]
            self.ntrials = 1
        
    def compute_hankel(
            self,
            data=None,
            n_delays=None,
            delay_interval=None,
        ):
        """
        Computes the Hankel matrix from the provided data.

        Parameters
        ----------
        data : np.ndarray or torch.tensor
            The data to fit the DMD model to. Must be either: (1) a
            2-dimensional array/tensor of shape T x N where T is the number
            of time points and N is the number of observed dimensions
            at each time point, or (2) a 3-dimensional array/tensor of shape
            K x T x N where K is the number of "trials" and T and N are
            as defined above.

        n_delays : int
            Parameter that controls the size of the delay embedding. Explicitly,
            the number of delays to include. Defaults to None - provide only if you want
            to override the value of n_delays from the init.

        delay_interval : int
            The number of time steps between each delay in the delay embedding. Defaults
            to 1 time step. Defaults to None - provide only if you want
            to override the value of n_delays from the init.
        """
        if self.verbose:
            print("Computing Hankel matrix ...")

        # if parameters are provided, overwrite them from the init
        self.data = self.data if data is None else self._init_data(data)
        self.n_delays = self.n_delays if n_delays is None else n_delays
        self.delay_interval = self.delay_interval if delay_interval is None else delay_interval

        self.H = embed_signal_torch(self.data, self.n_delays, self.delay_interval)

        if self.verbose:
            print("Hankel matrix computed!")
    
    def compute_svd(self):
        """
        Computes the SVD of the Hankel matrix.
        """
        if self.verbose:
            print("Computing SVD on Hankel matrix ...")
        if self.ntrials > 1: #flatten across trials for 3d
            H = self.H.reshape(self.H.shape[0] * self.H.shape[1], self.H.shape[2])
        else:
            H = self.H
        
        # compute the SVD
        U, S, Vh = torch.linalg.svd(H.T, full_matrices=False)
        
        # update attributes
        V = Vh.T
        self.U = U
        self.S = S
        self.V = V

        # construct the singuar value matrix and its inverse
        dim = self.n_delays * self.n
        s = len(S)
        self.S_mat = torch.zeros(dim, dim).to(self.device)
        self.S_mat_inv = torch.zeros(dim, dim).to(self.device)
        self.S_mat[np.arange(s), np.arange(s)] = S
        self.S_mat_inv[np.arange(s), np.arange(s)] = 1/S
        
        if self.verbose:
            print("SVD complete!")
    
    def compute_havok_dmd(
            self,
            rank=None,
            rank_thresh=None,
            lamb=0,
        ):
        """
        Computes the Havok DMD matrix.

        Parameters
        ----------
        rank : int
            The rank of V in fitting HAVOK DMD - i.e., the number of columns of V to 
            use to fit the DMD model. Defaults to None, in which case all columns of V
            will be used. Provide only if you want to override the value of n_delays 
            from the init.

        rank_thresh : int
            Parameter that controls the rank of V in fitting HAVOK DMD by dictating a threshold
            of singular values to use. Explicitly, the rank of V will be the number of singular
            values greater than rank_thresh. Defaults to None - provide only if you want
            to override the value of n_delays from the init.

        lamb : float
            Regularization parameter for ridge regression. Defaults to 0 - provide only if you want
            to override the value of n_delays from the init.

        """
        if self.verbose:
            print("Computing least squares fits to HAVOK DMD ...")
        
        self.rank = self.rank if rank is None else rank
        self.rank_thresh = self.rank_thresh if rank_thresh is None else rank_thresh
        self.lamb = self.lamb if lamb is None else lamb

        if self.rank is not None and self.rank_thresh is not None:
            raise ValueError("Cannot provide both rank and rank_thresh - pick one!")
        if self.rank is None and self.rank_thresh is None:
            self.rank = len(self.S)

        if self.rank > self.H.shape[-1]:
            raise ValueError(f"Provided rank {self.rank} cannot be larger than the number of columns in H {self.H.shape[-1]} ")
        
        # reshape for leastsquares
        if self.ntrials > 1:
            V = self.V.reshape(self.H.shape)
            #first reshape back into Hankel shape, separated by trials
            newshape = (self.H.shape[0]*(self.H.shape[1]-1),self.H.shape[2])
            Vt_minus = V[:,:-1].reshape(newshape)
            Vt_plus = V[:,1:].reshape(newshape)
        else:
            Vt_minus = self.V[:-1]
            Vt_plus = self.V[1:]

        if self.rank is None:
            if self.S[-1] > self.rank_thresh:
                self.rank = len(self.S)
            else:
                self.rank = torch.argmax(torch.arange(len(self.S), 0, -1).to(self.device)*(self.S < self.rank_thresh))

        A_v = (torch.linalg.inv(Vt_minus[:, :self.rank].T @ Vt_minus[:, :self.rank] + self.lamb*torch.eye(self.rank).to(self.device))@ Vt_minus[:, :self.rank].T@ Vt_plus[:, :self.rank]).T
        self.A_v = A_v
        self.A = A_v # for consistency across other files
        self.A_havok_dmd = self.U @ self.S_mat[:self.U.shape[1], :self.rank] @ self.A_v @ self.S_mat_inv[:self.rank, :self.U.shape[1]] @ self.U.T

        if self.verbose:
            print("Least squares complete!")

    def fit(
            self,
            data=None,
            n_delays=None,
            delay_interval=None,
            rank=None,
            rank_thresh=None,
            lamb=None,
            device=None,
            verbose=None,
        ):
        """
        Parameters
        ----------
        data : np.ndarray or torch.tensor
            The data to fit the DMD model to. Must be either: (1) a
            2-dimensional array/tensor of shape T x N where T is the number
            of time points and N is the number of observed dimensions
            at each time point, or (2) a 3-dimensional array/tensor of shape
            K x T x N where K is the number of "trials" and T and N are
            as defined above. Defaults to None - provide only if you want to
            override the value from the init.

        n_delays : int
            Parameter that controls the size of the delay embedding. Explicitly,
            the number of delays to include. Defaults to None - provide only if you want to
            override the value from the init.

        delay_interval : int
            The number of time steps between each delay in the delay embedding. Defaults to None - 
            provide only if you want to override the value from the init.

        rank : int
            The rank of V in fitting HAVOK DMD - i.e., the number of columns of V to 
            use to fit the DMD model. Defaults to None, in which case all columns of V
            will be used - provide only if you want to
            override the value from the init.

        rank_thresh : int
            Parameter that controls the rank of V in fitting HAVOK DMD by dictating a threshold
            of singular values to use. Explicitly, the rank of V will be the number of singular
            values greater than rank_thresh. Defaults to None - provide only if you want to
            override the value from the init.

        lamb : float
            Regularization parameter for ridge regression. Defaults to None - provide only if you want to
            override the value from the init.

        device: string or int
            A string or int to indicate the device to torch. For example, can be 'cpu' or 'cuda',
            or alternatively 0 if the intenion is to use GPU device 0. Defaults to None - provide only 
            if you want to override the value from the init.

        verbose: bool
            If True, print statements will be provided about the progress of the fitting procedure. 
            Defaults to None - provide only if you want to override the value from the init.
        """
        # if parameters are provided, overwrite them from the init
        self.device = self.device if device is None else device
        self.verbose = self.verbose if verbose is None else verbose
    
        # compute hankel
        self.compute_hankel(data, n_delays, delay_interval)
        self.compute_svd()
        self.compute_havok_dmd(rank, rank_thresh, lamb)

    def predict(
        self,
        test_data=None,
        reseed=None,
        full_return=False
    ):
        # initialize test_data
        if test_data is None:
            test_data = self.data
        if isinstance(test_data, np.ndarray):
            test_data = torch.from_numpy(test_data).to(self.device)
        ndim = test_data.ndim
        if ndim == 2:
            test_data = test_data.unsqueeze(0)
        H_test = embed_signal_torch(test_data, self.n_delays, self.delay_interval)

        if reseed is None:
            reseed = 1

        H_test_havok_dmd = torch.zeros(H_test.shape).to(self.device)
        H_test_havok_dmd[:, 0] = H_test[:, 0]

        A = self.A_havok_dmd.unsqueeze(0)
        for t in range(1, H_test.shape[1]):
            if t % reseed == 0:
                H_test_havok_dmd[:, t] = (A @ H_test[:, t - 1].transpose(-2, -1)).transpose(-2, -1)
            else:
                H_test_havok_dmd[:, t] = (A @ H_test_havok_dmd[:, t - 1].transpose(-2, -1)).transpose(-2, -1)
        pred_data = torch.hstack([test_data[:, :(self.n_delays - 1)*self.delay_interval + 1], H_test_havok_dmd[:, 1:, :self.n]])

        if ndim == 2:
            pred_data = pred_data[0]

        if full_return:
            return pred_data, H_test_havok_dmd, H_test
        else:
            return pred_data

# # -----------------------------------
# # TESTING
# # -----------------------------------

# def numpify(tensor):
#     if isinstance(tensor, torch.Tensor):
#         return tensor.detach().cpu().numpy()
#     else:
#         return tensor

# # num_features x num_observations
# def pearsonr_torch(x, y):
#     xm = x - x.mean(axis=1).unsqueeze(-1)
#     ym = y - y.mean(axis=1).unsqueeze(-1)
#     num = (xm*ym).sum(axis=1)
#     denom = torch.norm(xm, dim=1, p=2)*torch.norm(ym, dim=1, p=2)
#     return num/denom

# def get_autocorrel_funcs_torch(signal_multi_dim, num_lags=500, verbose=False):
#     dev_num = signal_multi_dim.get_device()
#     device = 'cpu' if dev_num < 0 else dev_num
#     if num_lags >= signal_multi_dim.shape[0]:
#         num_lags = signal_multi_dim.shape[0] - 1
#     autocorrel_funcs = torch.zeros(signal_multi_dim.shape[1], num_lags).to(device)
#     for lag in range(num_lags):
#         autocorrel_funcs[:, lag] = pearsonr_torch(signal_multi_dim[lag:].T, signal_multi_dim[:signal_multi_dim.shape[0] - lag].T)
    
#     return autocorrel_funcs

# # num observation x num features
# def r2_score_torch(target, preds):
#     target_mean = torch.mean(target, axis=0).unsqueeze(0)
#     ss_tot = torch.sum((target - target_mean) ** 2, axis=0)
#     ss_res = torch.sum((target - preds) ** 2, axis=0)
#     r2 = 1 - ss_res / ss_tot
#     return r2.mean()

# # num observations x num features
# def corrcoef_torch(mat):
#     corrcoefs = torch.zeros(mat.shape[1], mat.shape[1])
#     for i in range(mat.shape[1]):
#         for j in range(i):
#             corrcoefs[i, j] = pearsonr_torch(mat[:, [i]].T, mat[:, [j]].T)
#             corrcoefs[j, i] = corrcoefs[i, j]
    
#     return corrcoefs

# def clean_from_outliers(prior, posterior):
#     nonzeros = (prior != 0)
#     if any(prior == 0):
#         prior = prior[nonzeros]
#         posterior = posterior[nonzeros]
#     outlier_ratio = (1 - nonzeros.float()).mean()
#     return prior, posterior, outlier_ratio

# def eval_likelihood_gmm_for_diagonal_cov(z, mu, std):
#     T = mu.shape[0]
#     mu = mu.reshape((1, T, -1))

#     vec = z - mu  # calculate difference for every time step
#     vec=vec.float()
#     precision = 1 / (std ** 2)
#     precision = torch.diag_embed(precision).float()

#     prec_vec = torch.einsum('zij,azj->azi', precision, vec)
#     exponent = torch.einsum('abc,abc->ab', vec, prec_vec)
#     sqrt_det_of_cov = torch.prod(std, dim=1)
#     likelihood = torch.exp(-0.5 * exponent) / sqrt_det_of_cov
#     return likelihood.sum(dim=1) / T

# ## KLX Statespace
# def calc_kl_mc(mu_inf, cov_inf, mu_gen, cov_gen, device='cpu', mc_n=1000):
#     t = torch.randint(0, mu_inf.shape[0], (mc_n,)).to(device)

#     std_inf = torch.sqrt(cov_inf)
#     std_gen = torch.sqrt(cov_gen)
    
#     #print(mu_inf.shape)
#     #print(std_inf.shape)

#     z_sample = (mu_inf[t] + std_inf[t] * torch.randn(mu_inf[t].shape).to(device)).reshape((mc_n, 1, -1))

#     prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen)
#     posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_inf, std_inf)
#     prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)
#     kl_mc = torch.mean(torch.log(posterior) - torch.log(prior), dim=0)
#     return kl_mc, outlier_ratio

# def calc_kl_from_data(mu_gen, data_true, num_samples=1, mc_n=1000):
    
#     if data_true.device.type == 'cuda':
#         device = 'cuda'
#     else:
#         device = 'cpu'
    
#     time_steps = min(len(data_true), 10000)
#     mu_inf= data_true[:time_steps]
    
#     mu_gen=mu_gen[:time_steps]
    

#     scaling = 1.
#     cov_inf = torch.ones(data_true.shape[-1]).repeat(time_steps, 1).to(device)*scaling
#     cov_gen = torch.ones(data_true.shape[-1]).repeat(time_steps, 1).to(device)*scaling

#     kl_mc = 0
#     for num_sample in range(num_samples):
#         kl_mc1, _  = calc_kl_mc(mu_gen, cov_gen.detach(), mu_inf.detach(), cov_inf.detach(), device, mc_n)

#         kl_mc2, _  = calc_kl_mc(mu_inf.detach(), cov_inf.detach(), mu_gen, cov_gen.detach(), device, mc_n)

#         kl_mc += 1 / 2 * (kl_mc1 + kl_mc2)
#     kl_mc /= num_samples 

#     #scaling = 1
#    # mu_inf = get_posterior_mean(model.rec_model, x)
#     #cov_true = scaling * tc.ones_like(data_true)
#    # mu_gen = get_prior_mean(model.gen_model, time_steps)
#     #cov_gen = scaling * tc.ones_like(data_gen)

#     #kl_mc, _ = calc_kl_mc(data_true, cov_true, data_gen, cov_gen)
#     return kl_mc

# def compare_signal_statistics_torch(true_signal, pred_signal, num_lags=500, max_freq=100, fft_n=1000, dt=1, log_scale=False, autocorrel_true=None, return_data=False):
#     if isinstance(true_signal, np.ndarray):
#         true_signal = torch.from_numpy(true_signal).to('cpu')
#     if isinstance(pred_signal, np.ndarray):
#         pred_signal = torch.from_numpy(pred_signal).to('cpu')   
#     correl = pearsonr_torch(true_signal.T, pred_signal.T).mean()
#     mse = ((true_signal - pred_signal)**2).mean()
#     r2 = r2_score_torch(true_signal, pred_signal)
    
#     if autocorrel_true is None:
#         autocorrel_true = get_autocorrel_funcs_torch(true_signal, num_lags)
#     autocorrel_pred = get_autocorrel_funcs_torch(pred_signal, num_lags)

#     autocorrel_correl = pearsonr_torch(autocorrel_true, autocorrel_pred).mean()
#     autocorrel_mse = ((autocorrel_true - autocorrel_pred)**2).mean()
#     autocorrel_r2 = r2_score_torch(autocorrel_true.T, autocorrel_pred.T)
    
#     fft_true = torch.abs(torch.fft.rfft(true_signal.T, n=fft_n))
#     fft_pred = torch.abs(torch.fft.rfft(pred_signal.T, n=fft_n))
#     freqs = torch.fft.rfftfreq(fft_n, d=dt)
#     freq_inds = freqs <= max_freq
    
#     fft_true = fft_true[:, freq_inds]
#     fft_pred = fft_pred[:, freq_inds]
#     freqs = freqs[freq_inds]
    
#     fft_correl = pearsonr_torch(fft_true, fft_pred).mean()
#     fft_mse = ((fft_true - fft_pred)**2).mean()
#     fft_r2 = r2_score_torch(fft_true.T, fft_pred.T).mean()

#     log_fft_true = 10*torch.log10(fft_true)
#     log_fft_pred = 10*torch.log10(fft_pred)

#     log_fft_correl = pearsonr_torch(log_fft_true, log_fft_pred).mean()
#     log_fft_mse = ((log_fft_true - log_fft_pred)**2).mean()
#     log_fft_r2 = r2_score_torch(log_fft_true.T, log_fft_pred.T).mean()

#     true_correl_mat = corrcoef_torch(true_signal)
#     pred_correl_mat = corrcoef_torch(pred_signal)

#     correl_mat_correl = pearsonr_torch(true_correl_mat.flatten().unsqueeze(0), pred_correl_mat.flatten().unsqueeze(0))
#     correl_mat_mse = ((true_correl_mat - pred_correl_mat)**2).mean()
#     correl_mat_r2 = r2_score_torch(true_correl_mat.flatten().unsqueeze(-1), pred_correl_mat.flatten().unsqueeze(-1))

#     kl = calc_kl_from_data(pred_signal, true_signal)

#     stats = dict(
#         correl=correl,
#         mse=mse,
#         r2=r2,
#         autocorrel_correl=autocorrel_correl,
#         autocorrel_mse=autocorrel_mse,
#         autocorrel_r2=autocorrel_r2,
#         fft_correl=fft_correl,
#         fft_mse=fft_mse,
#         fft_r2=fft_r2,
#         log_fft_correl=log_fft_correl,
#         log_fft_mse=log_fft_mse,
#         log_fft_r2=log_fft_r2,
#         correl_mat_correl=correl_mat_correl,
#         correl_mat_mse=correl_mat_mse,
#         correl_mat_r2=correl_mat_r2,
#         kl=kl
#     )

#     if return_data:
#         data = dict(
#             autocorrel_true=autocorrel_true,
#             autocorrel_pred=autocorrel_pred,
#             fft_true=fft_true,
#             fft_pred=fft_pred,
#             freqs=freqs,
#         )
        
#         return stats | data
#     else:
        
#         return stats

# def compute_integrated_performance(dmd, signal, verbose=False, return_curve=False):
#     reseed_vals = np.array([1, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000])
#     performance_curve = np.zeros(len(reseed_vals))
#     if isinstance(signal, np.ndarray):
#         # x_true = torch.from_numpy(signal[window - p:window + T_pred]).to(dmd.device)
#         x_true = torch.from_numpy(signal).to(dmd.device)
#     else:

#         x_true = signal.to(dmd.device)
    
#     for i, reseed in tqdm(enumerate(reseed_vals), total=len(reseed_vals), disable=not verbose):
#         x_pred = dmd.predict_havok_dmd(x_true, tail_bite=True, reseed=reseed)[0]
#         stats = compare_signal_statistics_torch(x_true[dmd.p:], x_pred[dmd.p:, :x_true.shape[1]], dt=dt, return_data=False)
#         performance_curve[i] = (stats['autocorrel_correl'] + stats['fft_correl'] + stats['fft_r2'])/3
#     ip = scipy.integrate.trapezoid(x=reseed_vals/reseed_vals.max(), y=performance_curve)
#     if return_curve:
#         return ip, performance_curve, reseed_vals
#     else:
#         return ip