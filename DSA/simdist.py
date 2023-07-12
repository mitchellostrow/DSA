import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal
import torch.nn.utils.parametrize as parametrize

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
        self.C = nn.Parameter(torch.eye(n))
        
    def forward(self, B):
        return self.C @ B @ self.C.transpose(-1, -2)
    
class Skew(nn.Module):
    """
    Computes the skew-symmetric component of a matrix X
    """
    def forward(self, X):
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
                 device = 'cpu'
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
        """

        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.device = device
        self.C_star = None

    def fit(self, A, B):
        """
        Computes the optimal orthonormal matrix C

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B ; np.array or torch.tensor
            second data matrix

        Returns
        _______
        None
        """
        
        A = torch.from_numpy(A).float().to(self.device)
        B = torch.from_numpy(B).float().to(self.device)
        n = A.shape[0]

        #parameterize mapping to be orthogonal
        ortho_sim_net = LearnableOrthogonalSimilarityTransform(n).to(self.device)
        parametrize.register_parametrization(ortho_sim_net, "C", Skew())
        parametrize.register_parametrization(ortho_sim_net, "C", CayleyMap(n,self.device))
        
        simdist_loss = nn.MSELoss(reduction = 'mean')

        optimizer = optim.Adam(ortho_sim_net.parameters(), lr=self.lr)

        self.losses = []
        for _ in range(self.iters):
            # Zero the gradients of the optimizer.
            optimizer.zero_grad()      
            # Compute the Frobenius norm between A and the product.
            loss = simdist_loss(A, ortho_sim_net(B))

            loss.backward()

            optimizer.step()

            self.losses.append(loss.item())

        self.C_star = ortho_sim_net.C.detach()
    
    def score(self,A,B):
        """
        Given an optimal C already computed, calculate the metric

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B : np.array or torch.tensor
            second data matrix        

        Returns
        _______

        score : float
            similarity of the data under the similarity transform w.r.t C
        """
        assert self.C_star is not None 
        with torch.no_grad():
            A = torch.from_numpy(A).float().to(self.device)
            B = torch.from_numpy(B).float().to(self.device)
            C = self.C_star.to(self.device)

        if self.score_method == 'angular':    
            num = torch.trace(A @ C @ B.T @ C.T)
            den = torch.norm(A,p = 'fro')*torch.norm(B,p = 'fro')
            score = torch.arccos(num/den).cpu().numpy()
        else:
            score = torch.norm(A - C @ B @ C.T,p='fro').cpu().numpy()
    
        return score
    
    def fit_score(self,A,B):
        """
        for efficiency, computes the optimal matrix and returns the score 

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B : np.array or torch.tensor
            second data matrix        

        Returns
        _______

        score : float
            similarity of the data under the similarity transform w.r.t C
        """

        C_star, losses = self.fit(A, B)
        score_star = self.score(A,B)

        return C_star, score_star.item(), losses