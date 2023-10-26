import pytest
import numpy as np
from DSA.simdist import SimilarityTransformDist,pad_zeros
from scipy.stats import special_ortho_group, ortho_group
import torch
from netrep.utils import whiten

TOL = 1e-3
SIMTOL = 2e-2

@pytest.mark.parametrize('device',['cpu'])
@pytest.mark.parametrize('preserve_var',[True,False])
@pytest.mark.parametrize('dtype',['numpy'])
@pytest.mark.parametrize('score_method',['angular','euclidean'])
@pytest.mark.parametrize('n', [10,50,100])
@pytest.mark.parametrize('group',['GL(n)','O(n)','SO(n)'])
@pytest.mark.parametrize('seed', [5])
def test_simdist_convergent(seed,n,score_method,dtype,preserve_var,group,device):
    rng = np.random.default_rng(seed) 
    X = rng.random(size=(n,n))
    if group == 'SO(n)':
        Q = special_ortho_group(seed=rng,dim=n).rvs()
        Y = Q @ X @ Q.T
        iters = 5000
    elif group == 'O(n)':
        Q = ortho_group(seed=rng,dim=n).rvs()
        while np.linalg.det(Q) > 0:
            Q = ortho_group(seed=rng,dim=n).rvs()
        Y = Q @ X @ Q.T
        iters = 5000
    elif group == 'GL(n)':
        #draw random invertible matrix 
        Q = rng.random(size=(n,n))
        Q /= np.linalg.norm(Q,axis=0)
        Y = Q @ X @ np.linalg.inv(Q)
        iters = 80_000

    X,_ = whiten(X,0,preserve_variance=preserve_var)
    Y,_ = whiten(Y,0,preserve_variance=preserve_var)
    #excessive but we just want to see that it converges
    sim = SimilarityTransformDist(lr=1e-2,iters=iters,score_method=score_method,device=device,group=group)
    if dtype == 'torch':
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).float()
    score = sim.fit_score(X,Y)
    print(score)
    assert score < SIMTOL

@pytest.mark.parametrize('device',['cpu'])
@pytest.mark.parametrize('dtype',['numpy'])
@pytest.mark.parametrize('score_method',['angular','euclidean'])
@pytest.mark.parametrize('n', [10,50,100])
@pytest.mark.parametrize('seed', [5])
def test_transposed_q_same(seed,n,score_method,dtype,device):
    rng = np.random.default_rng(seed) 
    X = rng.random(size=(n,n)) * 2 - 1
    Q = special_ortho_group(seed=rng,dim=n).rvs()
    Y1 = Q @ X @ Q.T
    Y2 = Q.T @ X @ Q
    #excessive but we just want to see that it converges
    sim = SimilarityTransformDist(lr=5e-3,iters=15000,score_method=score_method,device=device)
    if dtype == 'torch':
        X = torch.tensor(X).float()
        Y1 = torch.tensor(Y1).float()
        Y2 = torch.tensor(Y2).float()

    score1 = sim.fit_score(X,Y1)
    score2 = sim.fit_score(X,Y2)
    print(n,score_method,score1)
    print(n,score_method,score2)
    assert np.abs(score1 - score2) < SIMTOL

@pytest.mark.parametrize('n2', [10])
@pytest.mark.parametrize('n1', [50])
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
    sim = SimilarityTransformDist(lr=1e-2,iters=5000)
    score = sim.fit_score(X,Y)
    C = sim.C_star
    assert np.allclose(C.T @ C, np.eye(n),atol=TOL)
    assert np.allclose(C @ C.T, np.eye(n),atol=TOL)
