import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from DSA.dmd import DMD
import torch
import ot
from typing import Literal

def compute_residuals(
    dmd: "DMD | np.ndarray | torch.Tensor",
    X: np.ndarray = None,
    Y: np.ndarray = None,
    rank: int = None,
    matrix: Literal["A_v", "A_havok_dmd"] = "A_v",
    return_num_denom = False,
    tol=1e-6
):
    """
    Compute DMD eigenvalues, eigenvectors, and residuals for each mode.

    Parameters
    ----------
    dmd : DMD object, np.ndarray, or torch.Tensor
        DMD object (with A_v, Vt_minus, Vt_plus, rank) or matrix.
    X : np.ndarray, optional
        Left-hand side data matrix.
    Y : np.ndarray, optional
        Right-hand side data matrix.
    rank : int, optional
        Rank of the DMD model.
    matrix : Literal["A_v", "A_havok_dmd"], optional
        Matrix to compute residuals on. Must be either "A_v" or "A_havok_dmd". Default is "A_v".
    return_num_denom : bool, optional
        Whether to return the numerator and denominator of the residual. Default is False.

    Returns
    -------
    L : np.ndarray
        Eigenvalues.
    G : np.ndarray
        Eigenvectors.
    residuals : np.ndarray
        Residuals for each eigenpair.
    normalized_residuals : np.ndarray or None
        Normalized residuals (if available, else None).
    """

    # Handle DMD object, numpy array, or torch tensor
    if hasattr(dmd, matrix):
        A = getattr(dmd, matrix).cpu().detach().numpy() if hasattr(getattr(dmd, matrix), "cpu") else getattr(dmd, matrix)
        L, G = np.linalg.eig(A)
        if matrix == "A_havok_dmd":
            X = dmd.Vt_minus.cpu().detach().numpy()[:, : dmd.rank] @ dmd.S_mat[:dmd.rank,:dmd.rank].cpu().detach().numpy() @ dmd.U.cpu().detach().numpy().T[:dmd.rank]
            Y = dmd.Vt_plus.cpu().detach().numpy()[:, : dmd.rank] @ dmd.S_mat[:dmd.rank,:dmd.rank].cpu().detach().numpy() @ dmd.U.cpu().detach().numpy().T[:dmd.rank]
            
        elif matrix == "A_v":
            X = (
                dmd.Vt_minus.cpu().detach().numpy()[:, : dmd.rank]
                if hasattr(dmd.Vt_minus, "cpu")
                else dmd.Vt_minus[:, : dmd.rank]
            )
            Y = (
                dmd.Vt_plus.cpu().detach().numpy()[:, : dmd.rank]
                if hasattr(dmd.Vt_plus, "cpu")
                else dmd.Vt_plus[:, : dmd.rank]
            )
        rank = dmd.rank
    elif isinstance(dmd, np.ndarray):
        A = dmd
        L, G = np.linalg.eig(A)
        if X is None or Y is None or rank is None:
            raise ValueError("If passing a raw matrix, must also provide X, Y, and rank.")
    elif hasattr(dmd, "numpy"):
        A = dmd.numpy()
        L, G = np.linalg.eig(A)
        if X is None or Y is None or rank is None:
            raise ValueError("If passing a raw matrix, must also provide X, Y, and rank.")
    else:
        raise ValueError("dmd must be a DMD object or a numpy array/torch tensor")

    # L = L[np.abs(L) > tol]
    # G = G[:, np.abs(L) > tol]
    # rank = len(L)
    if hasattr(dmd, "rank"):
        L = L[:rank]
        G = G[:, :rank]

    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, Y)
    YtY = np.dot(Y.T, Y)
    YtX = np.dot(Y.T, X)
    residuals = np.zeros(rank, dtype=np.complex64)
    persistence_residuals = np.zeros(rank, dtype=np.complex64)
    numerators = []
    denominators = []
    for i in range(rank):
        denominator = np.dot(G[:, i].conj().T, np.dot(XtX, G[:, i]))
        numerator = np.dot(
            G[:, i].conj().T,
            np.dot(
                YtY - np.conj(L[i]) * XtY - L[i] * YtX + np.abs(L[i]) ** 2 * XtX,
                G[:, i],
            ),
        )
        residuals[i] = numerator / denominator
        numerators.append(np.real(numerator))
        denominators.append(np.real(denominator))
        persistence_numerator = np.dot(
            G[:, i].conj().T, np.dot(YtY - XtY - YtX + XtX, G[:, i])
        )
        persistence_residuals[i] = persistence_numerator / denominator
    normalized_residuals = np.abs(residuals) / (np.abs(persistence_residuals) + 1e-10)
    if return_num_denom:
        return L, G, residuals, normalized_residuals, numerators, denominators
    else:
        return L, G, residuals, normalized_residuals


def plot_residuals(L, residuals, cmin=None, cmax=None):
    """
    Plot eigenvalues on the complex plane, colored by residuals.

    Parameters
    ----------
    L : np.ndarray
        Eigenvalues.
    residuals : np.ndarray
        Residuals for each eigenpair.
    cmin : float, optional
        Minimum value for color scale.
    cmax : float, optional
        Maximum value for color scale.
    """

    plt.scatter(np.real(L), np.imag(L), c=residuals)
    if cmin is None:
        cmin = np.min(residuals)
    if cmax is None:
        cmax = np.max(residuals)
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical")
    cbar.set_label("Residual")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    # plt.show()


def compute_inverse_participation_ratio(residuals):
    if isinstance(residuals, list):
        residuals = np.array(residuals)

    inv_resid = 1 / residuals
    num = np.sum(inv_resid) ** 2
    denom = np.sum(inv_resid**2)
    return num / denom


def clean_spectrum(L, G, residuals, epsilon):
    """
    remove the eigenvalues with value greater than epsilon
    """
    mask = residuals < epsilon
    return L[mask], G[:, mask], residuals[mask]


def thresh_topn(L, G, residuals, n):
    # pick the top n eigenvalues with the smallest residuals
    sorted_resid = np.sort(residuals)
    if n > len(sorted_resid):
        n = -1
    topn = sorted_resid[n]
    mask = residuals <= topn
    return L[mask], G[:, mask], residuals[mask]


def format_eigs(eig1):
    if isinstance(eig1, list):
        eig1 = np.array(eig1)
    # sort eigenvalues by real magnitude
    eig1 = eig1[np.argsort(np.abs(eig1.real))]

    eig1 = np.vstack([eig1.real, eig1.imag]).T
    return eig1


def compute_ot_distance(a, b):
    # check if a has imaginary compnents, if so convet to 2d array
    if np.iscomplexobj(a):
        a = np.vstack([a.real, a.imag]).T
    if np.iscomplexobj(b):
        b = np.vstack([b.real, b.imag]).T
    M = ot.dist(a, b)
    a, b = np.ones(a.shape[0]) / a.shape[0], np.ones(b.shape[0]) / b.shape[0]
    score = ot.emd2(a, b, M)
    C = ot.emd(a, b, M)
    return score, C
