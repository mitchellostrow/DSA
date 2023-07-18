"""This module computes the Havok DMD model for a given dataset."""
import numpy as np
import torch

def embed_signal_torch(data, n_delays, delay_interval=1):
    """
    Create a delay embedding from the provided tensor data.

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
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    device = data.device

    # initialize the embedding
    if data.ndim == 3:
        embedding = torch.zeros((data.shape[0], data.shape[1] - (n_delays - 1)*delay_interval, data.shape[2]*n_delays)).to(device)
    else:
        embedding = torch.zeros((data.shape[0] - (n_delays - 1)*delay_interval, data.shape[1]*n_delays)).to(device)
    
    for d in range(n_delays):
        index = (n_delays - 1 - d)*delay_interval
        ddelay = d*delay_interval

        if data.ndim == 3:
            ddata = d*data.shape[2]
            embedding[:,:, ddata: ddata + data.shape[2]] = data[:,index:data.shape[1] - ddelay]
        else:
            ddata = d*data.shape[1]
            embedding[:, ddata:ddata + data.shape[1]] = data[index:data.shape[0] - ddelay]
    
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

        rank_thresh : float
            Parameter that controls the rank of V in fitting HAVOK DMD by dictating a threshold
            of singular values to use. Explicitly, the rank of V will be the number of singular
            values greater than rank_thresh. Defaults to None.

        lamb : float
            Regularization parameter for ridge regression. Defaults to 0.

        device: string, int, or torch.device
            A string, int or torch.device object to indicate the device to torch.

        verbose: bool
            If True, print statements will be provided about the progress of the fitting procedure.

        Returns
         -------
         pred_data : torch.tensor
             The predictions generated by the HAVOK model. Of the same shape as test_data. Note that the first
             (self.n_delays - 1)*self.delay_interval + 1 time steps of the generated predictions are by construction
             identical to the test_data.
         
         H_test_havok_dmd : torch.tensor (Optional)
             Returned if full_return=True. The predicted Hankel matrix generated by the HAVOK model.
         H_test : torch.tensor (Optional)
             Returned if full_return=True. The true Hankel matri
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
        self.A_v = None
        self.A_havok_dmd = None

    def _init_data(self, data):
        # check if the data is an np.ndarry - if so, convert it to Torch
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        # move the data to the appropriate device
        self.data = data.to(self.device)

        # create attributes for the data dimensions
        if self.data.ndim == 3:
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
            self.rank = self.H.shape[-1]
        
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
        self.A_havok_dmd = self.U @ self.S_mat[:self.U.shape[1], :self.rank] @ self.A_v @ self.S_mat_inv[:self.rank, :self.U.shape[1]] @ self.U.T

        if self.verbose:
            print("Least squares complete! \n")

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