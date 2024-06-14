import numpy as np
import torch
from .dmd import DMD
from .simdist import SimilarityTransformDist
from .dsa import DSA
import warnings

def torch_convert(x):
    """
    Check if the array x is np.ndarray. If it is, convert to torch.Tensor.

    Parameters
    ----------
    x : np.ndarray or torch.tensor
        The array to check and convert if necessary.

    Returns
    -------
    x : torch.tensor
        If x was an np.ndarray, x is returns as a torch.tensor. Otherwise, x is returned as-is.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return x

def mae(x, y):
    """
    Compute the mean absolute error between the provided arrays.

    Parameters
    ----------
    x : np.ndarray or torch.tensor
        A multi-dimensional array.
    y : np.ndarray or torch.tensor
        A multi-dimensional array - must be the same size as x.

    Returns
    -------
    mae_val : float
        The mean absolute error between the provided arrays.
    """
    x = torch_convert(x)
    y = torch_convert(y)

    return torch.abs(x - y).mean().item()

def mase(true_vals, pred_vals):
    """
    Compute the mean absolute scaled error between the provided data. Explicitly, this
    is the mean absolute error on the predictions scaled by the mean absolut error achieved
    by taking the naive persistence baseline prediction, which is simply the value at the 
    previous time step.
    
    true_vals : np.ndarray or torch.tensor
        The ground truth time series. Must be either: (1) a
        2-dimensional array/tensor of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    pred_vals : np.ndarray or torch.tensor
        The predicted time series. Must be of the same shape as true_vals.

    Returns
    -------
    mase_val : float
        The mean absolute scaled error between the provided arrays.
    """
    true_vals = torch_convert(true_vals)
    pred_vals = torch_convert(pred_vals)

    if true_vals.ndim == 2:
        persistence_baseline = mae(true_vals[:-1], true_vals[1:])
    else: # true_vals.ndim == 3
        persistence_baseline = mae(true_vals[:, :-1], true_vals[:, 1:])
    
    return mae(true_vals, pred_vals) / persistence_baseline

def mse(x, y):
    """
    Compute the mean squared error between the provided arrays.

    Parameters
    ----------
    x : np.ndarray or torch.tensor
        A multi-dimensional array.
    y : np.ndarray or torch.tensor
        A multi-dimensional array - must be the same size as x.

    Returns
    -------
    mse_val : float
        The mean squared error between the provided arrays.
    """
    x = torch_convert(x)
    y = torch_convert(y)

    return ((x - y)**2).mean().item()

def r2(true_vals, pred_vals):
    """
    Compute the R-squared value between two sets of data. For arrays with multiple observed dimensions,
    the R-squared is computed separately for each dimension, and then averaged.
    
    true_vals : np.ndarray or torch.tensor
        The ground truth time series. Must be either: (1) a
        2-dimensional array/tensor of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    pred_vals : np.ndarray or torch.tensor
        The predicted time series. Must be of the same shape as true_vals.

    Returns
    -------
    r2_val : float
        The mean R-squared value for the provided sets of data.
    """
    true_vals = torch_convert(true_vals)
    pred_vals = torch_convert(pred_vals)

    if true_vals.ndim == 3:
        true_vals = true_vals.reshape(-1, true_vals.shape[-1])
        pred_vals = pred_vals.reshape(-1, pred_vals.shape[-1])
    
    SS_res = torch.sum((true_vals - pred_vals)**2, axis=0)
    SS_tot = torch.sum((true_vals - torch.mean(true_vals, axis=0))**2, axis=0)

    r2_vals = 1 - SS_res / SS_tot
    return torch.mean(r2_vals).item()

def correl(x, y):
    """
    Compute the correlation between two sets of data. For arrays with multiple observed dimensions,
    the correlation is computed separately for each dimension, and then averaged.
    
    x : np.ndarray or torch.tensor
        A multi-dimensional array. Must be either: (1) a
        2-dimensional array/tensor of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    y : np.ndarray or torch.tensor
        A multi-dimensional array. Must be of the same shape as x.

    Returns
    -------
    correl_val : float
        The mean correlation value for the provided sets of data.
    """
    x = torch_convert(x)
    y = torch_convert(y)

    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        
    correls = torch.zeros(x.shape[-1])
    for dim in range(x.shape[-1]):
        correls[dim] = torch.corrcoef(torch.vstack((x[:, dim], y[:, dim])))[0, 1]

    return correls.mean().item()

def aic(x, y, rank, norm=True):
    """
    Compute the Akaike information criterion (AIC) for the provided arrays. AIC attempts to
    balance a models prediction quality with the number of parameters used in the model.
    
    x : np.ndarray or torch.tensor
        A multi-dimensional array.

    y : np.ndarray or torch.tensor
        A multi-dimensional array. Must be of the same shape as x.

    rank : int
        The rank of the HAVOK model used for prediction.
    
    norm : bool
        If True, normalize the AIC by the number of data points in the arrays. Defaults
        to True.

    Returns
    -------
    aic_val : float
        The AIC value for the provided arrays.
    """
    x = torch_convert(x)
    y = torch_convert(y)

    N = np.prod(x.shape)

    AIC = (N*torch.log(((x - y)**2).sum()/N) + 2*(rank*rank + 1)).item()

    if norm:
        AIC /= N

    return AIC.item()
    
def log_mse(x,y,norm=True):
    x = torch_convert(x)
    y = torch_convert(y)

    N = np.prod(x.shape)
    logmse = (N*torch.log(((x - y)**2).sum()/N)).item()
    
    if norm:
        logmse /= N
    return logmse.item()

def compute_all_stats(true_vals, pred_vals, rank, norm=True):
    """
    Compute all statistics and put them in a dictionary.
    
    true_vals : np.ndarray or torch.tensor
        The ground truth time series. Must be either: (1) a
        2-dimensional array/tensor of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    pred_vals : np.ndarray or torch.tensor
        The predicted time series. Must be of the same shape as true_vals.

    rank : int
        The rank of the HAVOK model used for prediction.
    
    norm : bool
        If True, normalize the AIC by the number of data points in the arrays. Defaults
        to True.

    Returns
    -------
    stat_dict : dict
        All the computed statistics collected into a dictionary.
    """
    return {
        "MAE": mae(true_vals, pred_vals),
        "MASE": mase(true_vals, pred_vals),
        "MSE": mse(true_vals, pred_vals),
        "R2": r2(true_vals, pred_vals),
        "Correl": correl(true_vals, pred_vals),
        "AIC": aic(true_vals, pred_vals, rank, norm=norm),
        "logMSE": log_mse(true_vals,pred_vals,norm=norm)
    }

def dsa_to_id(data,rank,n_delays,delay_interval,iters=1000,lr=1e-2,
              score_method='angular',device='cpu'):
    """
    Compute the DSA between the provided data and the ID matrix of the specified rank

    Parameters
    ----------
    data : np.ndarray or torch.tensor
        The data to compute the DSA model for. Must be of shape T x N or C x T x N 
        where C is the number of conditions, T is the number
        of time points, and N is the number of observed dimensions at each time point.
    rank : int
        The rank of the dmd model.
    n_delays : int
        The number of delays to use for the dmd model.
    delay_interval : int
        The number of time steps between each delay.
    iters : int
        The number of iterations to use for the similarity transform distance.
    lr : float
        The learning rate to use for the similarity transform distance .
    score_method : str
        Whic metric value to use for the similarity transform distance. Defaults to 'angular'.
    device : str
        The device to use. Defaults to 'cpu'.

    Returns
    -------
    score: float


    """
    data = torch_convert(data)
    dmd = DMD(data,n_delays,delay_interval,rank,device=device)
    dmd.fit()

    if dmd.rank < rank:
        warnings.warn(f"The rank of the DMD model {dmd.rank} is less than the specified rank {rank}. Will revert to comparison at that rank.") 

    simdist = SimilarityTransformDist(iters,score_method=score_method,lr=lr,device=device)
    return simdist.fit_score(dmd.A_v,torch.eye(dmd.rank,device=device))


def dsa_bw_data_splits(data,rank,n_delays,delay_interval,nsplits=2,iters=1000,lr=1e-2,
              score_method='angular',device='cpu',avg=True):
    """
    Compute the DSA between splits of the provided data 

    Parameters
    ----------
    data : list, np.ndarray or torch.tensor
        The data to compute the DSA model over. 
        If data is a list, will default to computing the DSA between elements in the list.
        If not, will split the data into nsplits and compute the DSA between the splits.  
        Will always split on the 0th axis, so if data is a 3D array, will split along the context axis,
        and if data is 2d, will split along time.  
    rank : int
        The rank of the dmd model.
    n_delays : int
        The number of delays to use for the dmd model.
    delay_interval : int
        The number of time steps between each delay.
    iters : int
        The number of iterations to use for the similarity transform distance.
    lr : float
        The learning rate to use for the similarity transform distance .
    score_method : str
        Whic metric value to use for the similarity transform distance. Defaults to 'angular'.
    device : str
        The device to use. Defaults to 'cpu'.
    avg : bool
        Whether to average the DSA scores over the lower triangular component, not including the diagonal.
        Defaults to True.

    Returns
    -------
    score: np.ndarray (nsplits,nsplits)
    """
    if isinstance(data,list):
        data = np.array(data)

    if data.shape[0] % 2:
        data = data[:-1]
        
    data = np.split(data,nsplits,axis=0)
    
    
    dsa = DSA(data,n_delays=n_delays,rank=rank,delay_interval=delay_interval,iters=iters,lr=lr,score_method=score_method,device=device)
    score = dsa.fit_score()
    if avg:
        #average over the lower triangular component, not including the diagonal
        score = score[np.tril_indices(score.shape[0],k=-1)].mean()
        
    return score


