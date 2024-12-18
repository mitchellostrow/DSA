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

    if data.shape[int(data.ndim==3)] - (n_delays - 1)*delay_interval < 1:
        raise ValueError("The number of delays is too large for the number of time points in the data!")

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
            rank_explained_variance=None,
            reduced_rank_reg=False,
            lamb=0,
            device='cpu',
            verbose=False,
            send_to_cpu=False,
            steps_ahead=1
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

        rank_explained_variance : float
            Parameter that controls the rank of V in fitting HAVOK DMD by indicating the percentage of
            cumulative explained variance that should be explained by the columns of V. Defaults to None.
        
        reduced_rank_reg : bool
            Determines whether to use reduced rank regression (True) or principal component regression (False)

        lamb : float
            Regularization parameter for ridge regression. Defaults to 0.

        device: string, int, or torch.device
            A string, int or torch.device object to indicate the device to torch.

        verbose: bool
            If True, print statements will be provided about the progress of the fitting procedure.

        send_to_cpu: bool
            If True, will send all tensors in the object back to the cpu after everything is computed.
            This is implemented to prevent gpu memory overload when computing multiple DMDs.

        steps_ahead: int
            The number of time steps ahead to predict. Defaults to 1.
        """

        self.device = device
        self._init_data(data)

        self.n_delays = n_delays
        self.delay_interval = delay_interval
        self.rank = rank
        self.rank_thresh = rank_thresh
        self.rank_explained_variance = rank_explained_variance
        self.reduced_rank_reg = reduced_rank_reg
        self.lamb = lamb
        self.verbose = verbose
        self.send_to_cpu = send_to_cpu
        self.steps_ahead = steps_ahead

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
        self.data = data
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
        self.data = self.data.to(self.device)

        self.H = embed_signal_torch(self.data, self.n_delays, self.delay_interval)

        if self.verbose:
            print("Hankel matrix computed!")
    
    def compute_svd(self):
        """
        Computes the SVD of the Hankel matrix.
        """

        if self.verbose:
            print("Computing SVD on Hankel matrix ...")
        if self.H.ndim == 3: #flatten across trials for 3d
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
        # dim = self.n_delays * self.n
        # s = len(S)
        # self.S_mat = torch.zeros(dim, dim,dtype=torch.float32).to(self.device)
        # self.S_mat_inv = torch.zeros(dim, dim,dtype=torch.float32).to(self.device)
        self.S_mat = torch.diag(S).to(self.device)
        self.S_mat_inv= torch.diag(1 / S).to(self.device)

        # compute explained variance
        exp_variance_inds = self.S**2 / ((self.S**2).sum())
        cumulative_explained = torch.cumsum(exp_variance_inds, 0)
        self.cumulative_explained_variance = cumulative_explained
        
        #make the X and Y components of the regression by staggering the hankel eigen-time delay coordinates by time
        if self.reduced_rank_reg:
            V = self.V
        else:
            V = self.V

        if self.ntrials > 1:
            if V.numel() < self.H.numel():
                raise ValueError("The dimension of the SVD of the Hankel matrix is smaller than the dimension of the Hankel matrix itself. \n \
                                 This is likely due to the number of time points being smaller than the number of dimensions. \n \
                                 Please reduce the number of delays.")
                                 
            V = V.reshape(self.H.shape)

            #first reshape back into Hankel shape, separated by trials
            newshape = (self.H.shape[0]*(self.H.shape[1]-self.steps_ahead),self.H.shape[2])
            self.Vt_minus = V[:,:-self.steps_ahead].reshape(newshape)
            self.Vt_plus = V[:,self.steps_ahead:].reshape(newshape)
        else:
            self.Vt_minus = V[:-self.steps_ahead]
            self.Vt_plus = V[self.steps_ahead:]


        if self.verbose:
            print("SVD complete!")

    def recalc_rank(self,rank,rank_thresh,rank_explained_variance):
        '''
        Parameters
        ----------
        rank : int
            The rank of V in fitting HAVOK DMD - i.e., the number of columns of V to 
            use to fit the DMD model. Defaults to None, in which case all columns of V
            will be used. Provide only if you want to override the value from the init.

        rank_thresh : float
            Parameter that controls the rank of V in fitting HAVOK DMD by dictating a threshold
            of singular values to use. Explicitly, the rank of V will be the number of singular
            values greater than rank_thresh. Defaults to None - provide only if you want
            to override the value from the init.
        
        rank_explained_variance : float
            Parameter that controls the rank of V in fitting HAVOK DMD by indicating the percentage of
            cumulative explained variance that should be explained by the columns of V. Defaults to None -
            provide only if you want to overried the value from the init.        
            '''
        # if an argument was provided, overwrite the stored rank information
        none_vars = (rank is None) + (rank_thresh is None) + (rank_explained_variance is None)
        if none_vars != 3:
            self.rank = None
            self.rank_thresh = None
            self.rank_explained_variance = None
        
        self.rank = self.rank if rank is None else rank
        self.rank_thresh = self.rank_thresh if rank_thresh is None else rank_thresh
        self.rank_explained_variance = self.rank_explained_variance if rank_explained_variance is None else rank_explained_variance

        none_vars = (self.rank is None) + (self.rank_thresh is None) + (self.rank_explained_variance is None)
        if none_vars < 2:
            raise ValueError("More than one value was provided between rank, rank_thresh, and rank_explained_variance. Please provide only one of these, and ensure the others are None!")
        elif none_vars == 3:
           self.rank = len(self.S)

        if self.reduced_rank_reg:
            S = self.proj_mat_S
        else:
            S = self.S

        if rank_thresh is not None:
            if S[-1] > rank_thresh:
                self.rank = len(S)
            else:
                self.rank = torch.argmax(torch.arange(len(S), 0, -1).to(self.device)*(S < rank_thresh))

        if rank_explained_variance is not None:
            self.rank = int(torch.argmax((self.cumulative_explained_variance > rank_explained_variance).type(torch.int)).cpu().numpy())

        if self.rank > self.H.shape[-1]:
            self.rank = self.H.shape[-1]

        if self.rank is None:
            if S[-1] > self.rank_thresh:
                self.rank = len(S)
            else:
                self.rank = torch.argmax(torch.arange(len(S), 0, -1).to(self.device)*(S < self.rank_thresh))

    def compute_havok_dmd(self,lamb=None):
        """
        Computes the Havok DMD matrix (Principal Component Regression)

        Parameters
        ----------
        lamb : float
            Regularization parameter for ridge regression. Defaults to 0 - provide only if you want
            to override the value of n_delays from the init.

        """
        if self.verbose:
            print("Computing least squares fits to HAVOK DMD ...")
            
        self.lamb = self.lamb if lamb is None else lamb
    
        A_v = (torch.linalg.inv(self.Vt_minus[:, :self.rank].T @ self.Vt_minus[:, :self.rank] + self.lamb*torch.eye(self.rank).to(self.device)) \
               @ self.Vt_minus[:, :self.rank].T @ self.Vt_plus[:, :self.rank]).T
        self.A_v = A_v
        self.A_havok_dmd = self.U @ self.S_mat[:self.U.shape[1], :self.rank] @ self.A_v @ self.S_mat_inv[:self.rank, :self.U.shape[1]] @ self.U.T
        
        if self.verbose:
            print("Least squares complete! \n") 

    def compute_proj_mat(self,lamb=None):
        if self.verbose:
            print("Computing Projector Matrix for Reduced Rank Regression")

        self.lamb = self.lamb if lamb is None else lamb

        self.proj_mat = self.Vt_plus.T @ self.Vt_minus @ torch.linalg.inv(self.Vt_minus.T @ self.Vt_minus + 
                                                                          self.lamb*torch.eye(self.Vt_minus.shape[1]).to(self.device)) @ \
                                                                          self.Vt_minus.T @ self.Vt_plus
        
        self.proj_mat_S, self.proj_mat_V = torch.linalg.eigh(self.proj_mat)
        #todo: more efficient to flip ranks (negative index) in compute_reduced_rank_regression but also less interpretable
        self.proj_mat_S = torch.flip(self.proj_mat_S, dims=(0,))
        self.proj_mat_V = torch.flip(self.proj_mat_V, dims=(1,))

        if self.verbose:
            print("Projector Matrix computed! \n")

    def compute_reduced_rank_regression(self,lamb=None):
        if self.verbose:
            print("Computing Reduced Rank Regression ...")

        self.lamb = self.lamb if lamb is None else lamb
        proj_mat = self.proj_mat_V[:,:self.rank] @ self.proj_mat_V[:,:self.rank].T   
        B_ols = torch.linalg.inv(self.Vt_minus.T @ self.Vt_minus + self.lamb*torch.eye(self.Vt_minus.shape[1]).to(self.device)) @ self.Vt_minus.T @ self.Vt_plus

        self.A_v = B_ols @ proj_mat
        self.A_havok_dmd = self.U @ self.S_mat[:self.U.shape[1],:self.A_v.shape[1]] @ self.A_v.T @ self.S_mat_inv[:self.A_v.shape[0], :self.U.shape[1]] @ self.U.T


        if self.verbose:
            print("Reduced Rank Regression complete! \n")

    def fit(
            self,
            data=None,
            n_delays=None,
            delay_interval=None,
            rank=None,
            rank_thresh=None,
            rank_explained_variance=None,
            lamb=None,
            device=None,
            verbose=None,
            steps_ahead=None
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

        rank_explained_variance : float
            Parameter that controls the rank of V in fitting HAVOK DMD by indicating the percentage of
            cumulative explained variance that should be explained by the columns of V. Defaults to None -
            provide only if you want to overried the value from the init.

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

        steps_ahead: int
            The number of time steps ahead to predict. Defaults to 1.
        
        """
        # if parameters are provided, overwrite them from the init
        self.steps_ahead = self.steps_ahead if steps_ahead is None else steps_ahead
        self.device = self.device if device is None else device
        self.verbose = self.verbose if verbose is None else verbose
    
        self.compute_hankel(data, n_delays, delay_interval)
        self.compute_svd()

        if self.reduced_rank_reg:
            self.compute_proj_mat(lamb)
            self.recalc_rank(rank,rank_thresh,rank_explained_variance)
            self.compute_reduced_rank_regression(lamb)
        else:
            self.recalc_rank(rank,rank_thresh,rank_explained_variance)
            self.compute_havok_dmd(lamb)        
        
        if self.send_to_cpu:
            self.all_to_device('cpu') #send back to the cpu to save memory

    def predict(
        self,
        test_data=None,
        reseed=None,
        full_return=False
        ):
        """
        Returns
         -------
         pred_data : torch.tensor
             The predictions generated by the HAVOK model. Of the same shape as test_data. Note that the first
             (self.n_delays - 1)*self.delay_interval + 1 time steps of the generated predictions are by construction
             identical to the test_data.
         
         H_test_havok_dmd : torch.tensor (Optional)
             Returned if full_return=True. The predicted Hankel matrix generated by the HAVOK model.
         H_test : torch.tensor (Optional)
             Returned if full_return=True. The true Hankel matrix
        """
        # initialize test_data
        if test_data is None:
            test_data = self.data
        if isinstance(test_data, np.ndarray):
            test_data = torch.from_numpy(test_data).to(self.device)
        ndim = test_data.ndim
        if ndim == 2:
            test_data = test_data.unsqueeze(0)
        H_test = embed_signal_torch(test_data, self.n_delays, self.delay_interval)
        steps_ahead = self.steps_ahead if self.steps_ahead is not None else 1

        if reseed is None:
            reseed = 1

        H_test_havok_dmd = torch.zeros(H_test.shape).to(self.device)
        H_test_havok_dmd[:, :steps_ahead] = H_test[:, :steps_ahead]

        A = self.A_havok_dmd.unsqueeze(0)
        for t in range(steps_ahead, H_test.shape[1]):
            if t % reseed == 0:
                H_test_havok_dmd[:, t] = (A @ H_test[:, t - steps_ahead].transpose(-2, -1)).transpose(-2, -1)
            else:
                H_test_havok_dmd[:, t] = (A @ H_test_havok_dmd[:, t - steps_ahead].transpose(-2, -1)).transpose(-2, -1)
        pred_data = torch.hstack([test_data[:, :(self.n_delays - 1)*self.delay_interval + steps_ahead], H_test_havok_dmd[:, steps_ahead:, :self.n]])

        if ndim == 2:
            pred_data = pred_data[0]

        if full_return:
            return pred_data, H_test_havok_dmd, H_test
        else:
            return pred_data
    
    def all_to_device(self,device='cpu'):
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)

