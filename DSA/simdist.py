import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal
import numpy as np
import torch.nn.utils.parametrize as parametrize
from scipy.stats import special_ortho_group

def pad_zeros(A,B,device):

    with torch.no_grad():
        dim = max(A.shape[0],B.shape[0])
        A1 = torch.zeros((dim,dim)).float()
        A1[:A.shape[0],:A.shape[1]] += A
        A = A1.float().to(device)

        B1 = torch.zeros((dim,dim)).float()
        B1[:B.shape[0],:B.shape[1]] += B
        B = B1.float().to(device)

    return A,B

class LearnableOrthogonalSimilarityTransform(nn.Module):
    """
    Computes the similarity transform for a learnable orthonormal matrix C 
    """
    def __init__(self, n):
        """
        Parameters
        __________
        n : int
            dimension of the C matrix
        """
        super(LearnableOrthogonalSimilarityTransform, self).__init__()
        #initialize orthogonal matrix as identity
        self.C = nn.Parameter(torch.eye(n).float())
        
    def forward(self, B):
        return self.C @ B @ self.C.transpose(-1, -2)
    
# class Skew(nn.Module):
#     """
#     Computes the skew-symmetric component of a matrix X
#     """
#     def forward(self, X):
#         return X - X.transpose(-1, -2)

class Skew(nn.Module):
    def __init__(self,n,device):
        """
        Computes a skew-symmetric matrix X from some parameters (also called X)
        
        """
        super().__init__()
      
        self.L1 = nn.Linear(n,n,bias = False, device = device)
        self.L2 = nn.Linear(n,n,bias = False, device = device)
        self.L3 = nn.Linear(n,n,bias = False, device = device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X - X.transpose(-1, -2)

class CayleyMap(nn.Module):
    """
    Maps a skew-symmetric matrix to an orthogonal matrix in O(n)
    """
    def __init__(self, n, device):
        """
        Parameters
        __________

        n : int 
            dimension of the matrix we want to map
        
        device : {'cpu','cuda'} or int
            hardware device on which to send the matrix
        """
        super().__init__()
        self.register_buffer("Id", torch.eye(n,device = device))

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        return torch.linalg.solve(self.Id + X, self.Id - X)
    
class SimilarityTransformDist:
    """
    Computes the Procrustes Analysis over Vector Fields
    """
    def __init__(self,
                 iters = 200, 
                 score_method: Literal["angular", "euclidean"] = "angular",
                 lr = 0.01,
                 device = 'cpu',
                 verbose = False
                ):
        """
        Parameters
        _________
        iters : int
            number of iterations to perform gradient descent
        
        score_method : {"angular","euclidean"}
            specifies the type of metric to use 

        lr : float
            learning rate

        device : {'cpu','cuda'} or int

        verbose : bool
            prints when finished optimizing
        """

        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.C_star = None
        self.A = None
        self.B = None

    def fit(self, 
            A, 
            B, 
            iters = None, 
            lr = None, 
            zero_pad = True,
            ):
        """
        Computes the optimal orthonormal matrix C

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B : np.array or torch.tensor
            second data matrix
        iters : int or None
            number of optimization steps, if None then resorts to saved self.iters
        lr : float or None
            learning rate, if None then resorts to saved self.lr
        zero_pad : bool
            if True, then the smaller matrix will be zero padded so its the same size

        Returns
        _______
        None
        """
        assert A.shape[0] == A.shape[1]
        assert B.shape[0] == B.shape[1]
        assert A.shape[0] == B.shape[1] or zero_pad

        if isinstance(A,np.ndarray):
            A = torch.from_numpy(A).float().to(self.device)
        if isinstance(B,np.ndarray):
            B = torch.from_numpy(B).float().to(self.device)

        if zero_pad and A.shape != B.shape: #no point zero-padding if already equal
           A,B = pad_zeros(A,B,self.device)
        self.A,self.B = A,B
        n = A.shape[0]
        lr = self.lr if lr is None else lr
        iters = self.iters if iters is None else iters

        #parameterize mapping to be orthogonal
        ortho_sim_net = LearnableOrthogonalSimilarityTransform(n).to(self.device)
        parametrize.register_parametrization(ortho_sim_net, "C", Skew(n,self.device))
        parametrize.register_parametrization(ortho_sim_net, "C", CayleyMap(n,self.device))
        
        simdist_loss = nn.MSELoss(reduction = 'sum')

        optimizer = optim.Adam(ortho_sim_net.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        self.losses = []
        A /= np.linalg.norm(A)
        B /= np.linalg.norm(B)
        for _ in range(iters):
            # Zero the gradients of the optimizer.
            optimizer.zero_grad()      
            # Compute the Frobenius norm between A and the product.
            loss = simdist_loss(A, ortho_sim_net(B))

            loss.backward()

            optimizer.step()
            # if _ % 99:
            #     scheduler.step()
            self.losses.append(loss.item())

        if self.verbose:
            print("Finished optimizing C")

        self.C_star = ortho_sim_net.C.detach()
    
    def score(self,A=None,B=None,score_method=None):
        """
        Given an optimal C already computed, calculate the metric

        Parameters
        __________
        A : np.array or torch.tensor or None
            first data matrix, if None defaults to the saved matrix in fit
        B : np.array or torch.tensor or None
            second data matrix if None, defaults to the savec matrix in fit
        score_method : None or {'angular','euclidean'}
            overwrites the score method in the object for this application
        Returns
        _______

        score : float
            similarity of the data under the similarity transform w.r.t C
        """
        assert self.C_star is not None
        A = self.A if A is None else A
        B = self.B if B is None else B 
        assert A is not None
        assert B is not None
        assert A.shape == self.C_star.shape
        assert B.shape == self.C_star.shape
        score_method = self.score_method if score_method is None else score_method
        
        with torch.no_grad():
            if not isinstance(A,torch.Tensor):
                A = torch.from_numpy(A).float().to(self.device)
            if not isinstance(B,torch.Tensor):
                B = torch.from_numpy(B).float().to(self.device)
            C = self.C_star.to(self.device)

        if score_method == 'angular':    
            num = torch.trace(A @ C @ B.T @ C.T) 
            den = torch.norm(A,p = 'fro')*torch.norm(B,p = 'fro')
            score = torch.arccos(num/den).cpu().numpy()
        else:
            score = torch.norm(A - C @ B @ C.T,p='fro').cpu().numpy() #/ A.numpy().size
    
        return score
    
    def fit_score(self,
                A,
                B,
                iters = None, 
                lr = None,
                score_method = None,
                zero_pad = True):
        """
        for efficiency, computes the optimal matrix and returns the score 

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B : np.array or torch.tensor
            second data matrix        
        iters : int or None
            number of optimization steps, if None then resorts to saved self.iters
        lr : float or None
            learning rate, if None then resorts to saved self.lr
        score_method : {'angular','euclidean'} or None
            overwrites parameter in the class
        zero_pad : bool
            if True, then the smaller matrix will be zero padded so its the same size
        Returns
        _______

        score : float
            similarity of the data under the similarity transform w.r.t C
            
        """
        score_method = self.score_method if score_method is None else score_method
        
        if zero_pad and A.shape != B.shape: 
           A,B = pad_zeros(A,B,self.device)
       
        self.fit(A, B,iters,lr,zero_pad)
        score_star = self.score(self.A,self.B,score_method=score_method)

        return score_star.item()

