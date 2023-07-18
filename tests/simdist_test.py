import pytest
import numpy as np
from DSA.simdist import SimilarityTransformDist,pad_zeros
from scipy.stats import special_ortho_group
import torch

TOL = 1e-6
SIMTOL = 0.1

@pytest.mark.parametrize('seed', [5])
@pytest.mark.parametrize('n', [10,50,100])
@pytest.mark.parametrize('score_method',['angular','euclidean'])
@pytest.mark.parametrize('dtype',['numpy','torch'])
@pytest.mark.parametrize('device',['cpu'])
def test_simdist_convergent(seed,n,score_method,dtype,device):
    rng = np.random.default_rng(seed) 
    X = rng.random(size=(n,n)) * 2 - 1
    Q = special_ortho_group(seed=rng,dim=n).rvs()
    Y = Q @ X @ Q.T
    #excessive but we just want to see that it converges
    sim = SimilarityTransformDist(lr=1e-1,iters=10000,score_method=score_method,device=device)
    if dtype == 'torch':
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).float()
    score = sim.fit_score(X,Y)
    assert score < SIMTOL

@pytest.mark.parametrize('n1', [50])
@pytest.mark.parametrize('n2', [10])
@pytest.mark.parametrize('seed', [5])
def test_zero_pad(seed,n1,n2):
    rng = np.random.default_rng(seed) 
    X = rng.random(size=(n1,n1))
    Y = rng.random(size=(n2,n2))
    m = max(n1,n2)
    sim = SimilarityTransformDist(iters=10) #don't care about fitting
    sim.fit_score(X,Y,zero_pad=True)
    assert sim.C_star.shape == (m,m)
    assert pad_zeros(X,Y,'cpu')[0].shape == (m,m)
    assert pad_zeros(X,Y,'cpu')[1].shape == (m,m)

@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('seed', [5])
def test_ortho_c(seed,n):
    rng = np.random.default_rng(seed) 
    X = rng.random(size=(n,n))
    Q = special_ortho_group(seed=rng,dim=n).rvs()
    Y = Q @ X @ Q.T
    sim = SimilarityTransformDist(iters=1000)
    score = sim.fit_score(X,Y)
    C = sim.C_star
    assert np.allclose(C.T @ C, np.eye(n),atol=TOL)
    assert np.allclose(C @ C.T, np.eye(n),atol=TOL)
