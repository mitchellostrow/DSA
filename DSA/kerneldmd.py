from sklearn.gaussian_process.kernels import DotProduct, RBF
from kooplearn.data import traj_to_contexts
from kooplearn.models import NystroemKernel
import numpy as np
import torch

class KernelDMD(NystroemKernel):
    def __init__(
            self,
            data,
            n_delays,
            kernel = RBF(),
            num_centers=0.1,
            delay_interval=1,
            rank=10,
            reduced_rank_reg=True,
            lamb=None,
            verbose=False,
            svd_solver='full',
        ):
        """
        Subclass of kooplearn that uses a kernel to compute the DMD model.
        This will also use Reduced Rank Regresion as opposed to Principal Component Regression (above)
        """
        super().__init__(kernel,reduced_rank_reg,rank,lamb,svd_solver,num_centers)
        self.n_delays = n_delays
        self.delay_interval = delay_interval
        self.verbose = verbose
        self.rank = rank
        self.lamb = 0 if lamb is None else lamb
        
        self.data = data
    
    def fit(
            self,
            data=None,
            lamb=None,
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

        lamb : float
            Regularization parameter for ridge regression. Defaults to None - provide only if you want to
            override the value from the init.
        """
        data = self.data if data is None else data
        lamb = self.lamb if lamb is None else lamb

        self.compute_hankel(data)
        self.compute_kernel_dmd(lamb)

    def compute_hankel(self,trajs):
        '''
        Given a numpy array or list of trajectories, returns a numpy array of delay embeddings
        in the format required by kooplearn. 
        Parameters
        ----------
        trajs : np.ndarray or list, with each array having shape 
            (num_samples, timesteps, dimension) or shape (timesteps, dimension).
            Note that trajectories can have different numbers of timesteps but must have the same dimension
        n_delays : int
            The number of delays to include in the delay embedding
        delay_interval : int
            The number of time steps between each delay in the delay embedding
        '''
        if isinstance(trajs, torch.Tensor):
            #convert trajs to a np array
            trajs = trajs.numpy()
        if isinstance(trajs,np.ndarray) and trajs.ndim == 2:
            return traj_to_contexts(trajs,context_window_len=self.n_delays,time_lag=self.delay_interval)
        
        data = [] #TODO: preallocate
        for i in range(len(trajs)):
            data.extend(traj_to_contexts(trajs[i],context_window_len=self.n_delays,time_lag=self.delay_interval))
       
        self.data = np.array(data)

        if self.verbose:
            print("Hankel matrix computed")

    def compute_kernel_dmd(self,lamb = None):
        '''
        Computes the kernel DMD model. 
        '''
        self.tikhonov_reg = self.lamb if lamb is None else lamb
        #we need to use the inherited .fit method from NystroemKernel
        super().fit(self.data)

        self.A_v = self.V.T @ self.kernel_YX @ self.U / len(self.kernel_YX)

        if self.verbose:
            print("kernel regression complete")

    def predict(
        self,
        test_data=None,
        reseed=None,
        ):

        if test_data is None:
            test_data = self.data

        test_data = self.compute_hankel(test_data, self.n_delays, self.delay_interval)
        #get the test data into the format of the hankel matrix
        #apply the nystroem predict function
        if reseed is None:
            reseed = 1
        
        if test_data.ndim == 2:
            test_data = test_data[np.newaxis, :, :]

        pred_data = np.zeros(test_data.shape)

        for t in range(1, test_data.shape[1]):
            if t % reseed == 0:
                pred_data[:,t] = super().predict(test_data[:,t])
            else:
                pred_data[:,t] = super().predict(pred_data[:,t-1])

        if test_data.ndim == 2:
            pred_data = pred_data[0]

        return pred_data
