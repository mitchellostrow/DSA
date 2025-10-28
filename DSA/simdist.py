import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal
import numpy as np
import torch.nn.utils.parametrize as parametrize
from scipy.stats import wasserstein_distance
import ot  # optimal transport for multidimensional l2 wasserstein

try:
    from .dmd import DMD
except ImportError:
    from dmd import DMD


def pad_zeros(A, B, device):

    with torch.no_grad():
        dim = max(A.shape[0], B.shape[0])
        A1 = torch.zeros((dim, dim)).float()
        A1[: A.shape[0], : A.shape[1]] += A
        A = A1.float().to(device)

        B1 = torch.zeros((dim, dim)).float()
        B1[: B.shape[0], : B.shape[1]] += B
        B = B1.float().to(device)

    return A, B


def compute_angle(evec):
    """
    computes the angle between multiple complex eigenvectors
    """
    if isinstance(evec, np.ndarray):
        evec = torch.from_numpy(evec).float()
    # evec /= torch.linalg.norm(evec, dim=1, keepdim=True)
    ang = torch.real(evec.H @ evec)
    ang = torch.arccos(ang)
    ang[torch.isnan(ang)] = 0
    return ang


class LearnableSimilarityTransform(nn.Module):
    """
    Computes the similarity transform for a learnable orthonormal matrix C
    """

    def __init__(self, n, orthog=True):
        """
        Parameters
        __________
        n : int
            dimension of the C matrix
        """
        super(LearnableSimilarityTransform, self).__init__()
        # initialize orthogonal matrix as identity
        self.C = nn.Parameter(torch.eye(n).float())
        self.orthog = orthog

    def forward(self, B):
        if self.orthog:
            return self.C @ B @ self.C.transpose(-1, -2)
        else:
            return self.C @ B @ torch.linalg.inv(self.C)


class Skew(nn.Module):
    def __init__(self, n, device):
        """
        Computes a skew-symmetric matrix X from some parameters (also called X)

        """
        super().__init__()

        self.L1 = nn.Linear(n, n, bias=False, device=device)
        self.L2 = nn.Linear(n, n, bias=False, device=device)
        self.L3 = nn.Linear(n, n, bias=False, device=device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X - X.transpose(-1, -2)


class Matrix(nn.Module):
    def __init__(self, n, device):
        """
        Computes a matrix X from some parameters (also called X)

        """
        super().__init__()

        self.L1 = nn.Linear(n, n, bias=False, device=device)
        self.L2 = nn.Linear(n, n, bias=False, device=device)
        self.L3 = nn.Linear(n, n, bias=False, device=device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X


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
        self.register_buffer("Id", torch.eye(n, device=device))

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        return torch.linalg.solve(self.Id + X, self.Id - X)


class SimilarityTransformDist:
    """
    Computes the Procrustes Analysis over Vector Fields
    """

    def __init__(
        self,
        iters=200,
        score_method: Literal["angular", "euclidean", "wasserstein"] = "angular",
        lr=0.01,
        device: Literal["cpu", "cuda"] = "cpu",
        verbose=False,
        eps=1e-5,
        rescale_wasserstein=False,
    ):
        """
        Parameters
        _________
        iters : int
            number of iterations to perform gradient descent

        score_method : {"angular","euclidean","wasserstein"}
            specifies the type of metric to use
            "wasserstein" will compare the singular values or eigenvalues
            of the two matrices as in Redman et al., (2023)

        lr : float
            learning rate

        device : {'cpu','cuda'} or int

        verbose : bool
            prints when finished optimizing

        eps : float
            early stopping threshold
        """

        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.C_star = None
        self.A = None
        self.B = None
        self.eps = eps
        self.rescale_wasserstein = rescale_wasserstein

    def fit(
        self,
        A,
        B,
        iters=None,
        lr=None,
        score_method=None,
        wasserstein_weightings=None,
    ):
        """
        Computes the optimal matrix C over specified group

        Parameters
        __________
        A : np.array or torch.tensor or DMD object
            first data matrix
        B : np.array or torch.tensor or DMD object
            second data matrix
        iters : int or None
            number of optimization steps, if None then resorts to saved self.iters
        lr : float or None
            learning rate, if None then resorts to saved self.lr

        Returns
        _______
        None
        """
        if isinstance(A, DMD):
            A = A.A_v
        if isinstance(B, DMD):
            B = B.A_v

        assert A.shape[0] == A.shape[1]
        assert B.shape[0] == B.shape[1]

        A = A.to(self.device)
        B = B.to(self.device)
        self.A, self.B = A, B
        lr = self.lr if lr is None else lr
        iters = self.iters if iters is None else iters
        wasserstein_compare = (
            self.wasserstein_compare
            if wasserstein_compare is None
            else wasserstein_compare
        )
        score_method = self.score_method if score_method is None else score_method

        if score_method == "wasserstein":
            a, b = self._get_wasserstein_vars(A, B)
            device = a.device
            # a = a  # .cpu()
            # b = b  # .cpu()
            self.M = ot.dist(a, b)  # .numpy()
            if wasserstein_weightings is not None:
                a, b = wasserstein_weightings
                assert isinstance(a, (torch.Tensor, np.ndarray))
                assert isinstance(b, (torch.Tensor, np.ndarray))
                assert a.shape[0] == self.M.shape[0]
                assert b.shape[0] == self.M.shape[1]
                assert a.sum() == b.sum() == 1
            else:
                a, b = (
                    torch.ones(a.shape[0]) / a.shape[0],
                    torch.ones(b.shape[0]) / b.shape[0],
                )
            a, b = a.to(device), b.to(device)

            self.C_star = ot.emd(a, b, self.M)
            self.score_star = (
                ot.emd2(a, b, self.M) * a.shape[0]
            )  # add scaling factor due to random matrix theory
            # self.score_star = np.sum(self.C_star * self.M)
            self.C_star = self.C_star / torch.linalg.norm(
                self.C_star, dim=1, keepdim=True
            )
            # wasserstein_distance(A.cpu().numpy(),B.cpu().numpy())

        else:
            self.losses, self.C_star, self.sim_net = self.optimize_C(
                A, B, lr, iters, orthog=True, verbose=self.verbose
            )
            # permute the first row and column of B then rerun the optimization
            P = torch.eye(B.shape[0], device=self.device)
            if P.shape[0] > 1:
                P[[0, 1], :] = P[[1, 0], :]
            losses, C_star, sim_net = self.optimize_C(
                A, P @ B @ P.T, lr, iters, orthog=True, verbose=self.verbose
            )
            if losses[-1] < self.losses[-1]:
                self.losses = losses
                self.C_star = C_star @ P
                self.sim_net = sim_net

    def _get_wasserstein_vars(self, A, B):
        # assert self.wasserstein_compare in {"sv", "eig","evec_angle", 'evec'}
        assert self.wasserstein_compare in {"eig"}

        # deprecated: only do wasserstein comparison on eigenvalues (for now, until others are theoretically validated)
        # if self.wasserstein_compare == "sv":
        # a = torch.svd(A).S.view(-1, 1)
        # b = torch.svd(B).S.view(-1, 1)
        # if self.wasserstein_compare == "eig":
        a = torch.linalg.eig(A).eigenvalues
        a = torch.vstack([a.real, a.imag]).T

        b = torch.linalg.eig(B).eigenvalues
        b = torch.vstack([b.real, b.imag]).T
        # elif self.wasserstein_compare in {'evec_angle', 'evec'}:
        #     #this will compute the interior angles between eigenvectors
        #     aevec = torch.linalg.eig(A).eigenvectors
        #     bevec = torch.linalg.eig(B).eigenvectors

        #     a = compute_angle(aevec)
        #     b = compute_angle(bevec)
        # else:
        # raise AssertionError("wasserstein_compare must be 'sv', 'eig', 'evec_angle', or 'evec'")

        # if the number of elements in the sets are different, then we need to pad the smaller set with zeros
        if a.shape[0] != b.shape[0]:
            # if self.wasserstein_compare in {'evec_angle', 'evec'}:
            # raise AssertionError("Wasserstein comparison of eigenvectors is not supported when \
            #  the number of elements in the sets are different")
            if self.verbose:
                print(f"Padding the smaller set with zeros")
            if a.shape[0] < b.shape[0]:
                a = torch.cat(
                    [a, torch.zeros(b.shape[0] - a.shape[0], a.shape[1])], dim=0
                )
            else:
                b = torch.cat(
                    [b, torch.zeros(a.shape[0] - b.shape[0], b.shape[1])], dim=0
                )
        return a, b

    def optimize_C(self, A, B, lr, iters, orthog, verbose):
        # parameterize mapping to be orthogonal
        n = A.shape[0]
        sim_net = LearnableSimilarityTransform(n, orthog=orthog).to(self.device)
        if orthog:
            parametrize.register_parametrization(sim_net, "C", Skew(n, self.device))
            parametrize.register_parametrization(
                sim_net, "C", CayleyMap(n, self.device)
            )
        else:
            parametrize.register_parametrization(sim_net, "C", Matrix(n, self.device))

        simdist_loss = nn.MSELoss(reduction="sum")

        optimizer = optim.Adam(sim_net.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        losses = []
        A /= torch.linalg.norm(A)
        B /= torch.linalg.norm(B)
        for _ in range(iters):
            # Zero the gradients of the optimizer.
            optimizer.zero_grad()
            # Compute the Frobenius norm between A and the product.
            loss = simdist_loss(A, sim_net(B))

            loss.backward()

            optimizer.step()
            # if _ % 99:
            #     scheduler.step()
            losses.append(loss.item())
            # TODO: add a flag for this
            # if _ > 2 and abs(losses[-1] - losses[-2]) < self.eps: #early stopping
            # break

        if verbose:
            print("Finished optimizing C")

        C_star = sim_net.C.detach()
        return losses, C_star, sim_net

    def score(self, A=None, B=None, score_method=None):
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
        assert A.shape == self.C_star.shape or score_method == "wasserstein"
        assert B.shape == self.C_star.shape or score_method == "wasserstein"
        score_method = self.score_method if score_method is None else score_method
        with torch.no_grad():
            if not isinstance(A, torch.Tensor):
                A = torch.from_numpy(A).float().to(self.device)
            if not isinstance(B, torch.Tensor):
                B = torch.from_numpy(B).float().to(self.device)
            C = self.C_star.to(self.device)

        if score_method == "angular":
            num = torch.trace(A.T @ C @ B @ C.T)
            den = torch.norm(A, p="fro") * torch.norm(B, p="fro")
            score = torch.arccos(num / den).cpu().numpy()
            if np.isnan(score):  # around -1 and 1, we sometimes get NaNs due to arccos
                if num / den < 0:
                    score = np.pi
                else:
                    score = 0
        elif score_method == "euclidean":
            score = (
                torch.norm(A - C @ B @ C.T, p="fro").cpu().numpy().item()
            )  # / A.numpy().size
        elif score_method == "wasserstein":
            # use the current C_star to compute the score
            assert hasattr(self, "score_star")
            # if wasserstein_compare == self.wasserstein_compare:
            score = self.score_star.item()
            # non-eig wasserstein comparisons are deprecated until theoretically validated
            # else:
            #     #apply the current transport plan to the new data
            #     a,b = self._get_wasserstein_vars(A, B)
            #     # a_transported =  self.C_star @ A #shouldn't this be a?

            #     M = ot.dist(a, b, metric='sqeuclidean')
            #     score = torch.sum(self.C_star * M).item()
            #     #TODO: validate this
            #     # a_transported = self.C_star @ a
            #     # row_wise_sq_distances = torch.sum(torch.square(a_transported - b), axis=1)
            #     # transported_score = torch.sum(a * row_wise_sq_distances)
            #     # score = transported_score.item()
            #     if self.rescale_wasserstein:
            #         score = score * A.shape[0] #add scaling factor due to random matrix theory

        return score

    def fit_score(
        self, A, B, iters=None, lr=None, score_method=None, wasserstein_weightings=None
    ):
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
        score_method : {'angular','euclidean', 'wasserstein} or None
            overwrites parameter in the class
        zero_pad : bool
            if True, then the smaller matrix will be zero padded so its the same size
        Returns
        _______

        score : float
            similarity of the data under the similarity transform w.r.t C

        """
        score_method = self.score_method if score_method is None else score_method

        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B).float()

        assert A.shape[0] == B.shape[1] or self.wasserstein_compare is not None
        if A.shape[0] != B.shape[0]:
            # if self.wasserstein_compare is None:
            # raise AssertionError(
            # "Matrices must be the same size unless using wasserstein distance"
            # )
            if (
                score_method != "wasserstein"
            ):  # otherwise resort to L2 Wasserstein over singular or eigenvalues
                print(
                    f"resorting to wasserstein distance over {self.wasserstein_compare}"
                )
                score_method = "wasserstein"
            else:
                pass

        self.fit(
            A,
            B,
            iters,
            lr,
            wasserstein_weightings=wasserstein_weightings,
            score_method=score_method,
        )

        return self.score(self.A, self.B, score_method=score_method)


def compute_subspace_angles(A, B):
    """
    Computes the subspace angles between two DMD matrices.
    Matrices must be square and the same size.

    Parameters
    ----------
    A : DMD object or numpy array
        First DMD matrix
    B : DMD object or numpy array
        Second DMD matrix

    Returns
    -------
    angles : np.ndarray
        Principal angles between the subspaces
    """

    A_mat = val_matrix(A)
    B_mat = val_matrix(B)

    # Check matrices are same size
    if A_mat.shape != B_mat.shape:
        raise ValueError("Matrices must be the same size")

    # Get orthonormal bases via SVD
    U_A = np.linalg.svd(A_mat)[0]
    U_B = np.linalg.svd(B_mat)[0]

    # Compute principal angles
    S = np.linalg.svd(U_A.T @ U_B)[1]
    S = np.clip(S, -1.0, 1.0)  # Numerical stability
    angles = np.arccos(S)

    return angles


def val_matrix(matrix):
    if isinstance(matrix, DMD):
        mat = matrix.A_havok_dmd
    elif isinstance(matrix, torch.Tensor):
        mat = matrix.detach().numpy()
    elif isinstance(matrix, np.ndarray):
        mat = matrix
    else:
        raise AssertionError(f" must be tensor, numpy array, or DMD object")

    # Check matrix is square
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"matrix must be square")

    return mat
