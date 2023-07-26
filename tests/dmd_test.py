import pytest
import numpy as np
from DSA.dmd import DMD, embed_signal_torch
from scipy.stats import ortho_group
import torch
TOL = 1e-2

@pytest.mark.parametrize('delay_interval', [1,2])
@pytest.mark.parametrize('n_delays', [20])
@pytest.mark.parametrize('c', [10])
@pytest.mark.parametrize('t', [100])
@pytest.mark.parametrize('seed', [21])
def test_embed_3dvs2d(seed,t,c,n_delays,delay_interval):
    #test to make sure 3d and 2d are effectively doing the same thing
    n = 3
    rng = np.random.default_rng(seed) 
    data = torch.from_numpy(rng.random((n,t,c)))
    embed1 = embed_signal_torch(data,n_delays,delay_interval)
    embed2s = [embed_signal_torch(data[i],n_delays,delay_interval) for i in range(n)]
    assert np.allclose(embed1[0],embed2s[0],atol=TOL)
    assert np.allclose(embed1[1],embed2s[1],atol=TOL)
    assert np.allclose(embed1[2],embed2s[2],atol=TOL)

@pytest.mark.parametrize('c', [10])
@pytest.mark.parametrize('t', [100])
@pytest.mark.parametrize('n', [50])
@pytest.mark.parametrize('seed', [21])
def test_embed_1delay(seed,n,t,c):
    rng = np.random.default_rng(seed) 
    data = torch.from_numpy(rng.random((n,t,c)))
    embed = embed_signal_torch(data,1)
    embed1 = embed_signal_torch(data[0],1)
    dmd = DMD(data,1)
    dmd.compute_hankel()
    assert np.allclose(embed,data,atol=TOL)
    assert np.allclose(dmd.H,data,atol=TOL)
    assert np.allclose(embed1,data[0],atol=TOL)

@pytest.mark.parametrize('rank', [10,50,250])
@pytest.mark.parametrize('n_delays', [1,20])
@pytest.mark.parametrize('c', [10])
@pytest.mark.parametrize('t', [500])
@pytest.mark.parametrize('n', [50])
@pytest.mark.parametrize('seed', [21])
def test_dmd_rank(seed,n,t,c,n_delays,rank):
    rng = np.random.default_rng(seed) 
    X = rng.random((n,t,c))
    dmd = DMD(X,n_delays,rank=rank)
    dmd.fit()
    rank = min(rank,n_delays*c)
    assert dmd.A_v.shape == (rank,rank)

@pytest.mark.parametrize('tau', [0.01])
@pytest.mark.parametrize('t', [1000])
@pytest.mark.parametrize('c', [5])
@pytest.mark.parametrize('seed', [21])
def test_dmd_2d(seed,c,t,tau): 
    rng = np.random.default_rng(seed)
    x0 = rng.random((c))
    data = np.zeros((t,c))
    data[0] = x0
    Q = ortho_group.rvs(c)
    A = np.eye(c) + tau*Q #\dot{x} = Qx -> x_t+1 ~= x + \tauQx
    for i in range(1,t):
        data[i] = A @ data[i-1]
    dmd = DMD(data,1)
    dmd.fit()
    assert np.linalg.norm(dmd.A_v.flatten() - A.flatten()) < 1e-1

@pytest.mark.parametrize('n', [500])
@pytest.mark.parametrize('t', [1000])
@pytest.mark.parametrize('c', [3])
@pytest.mark.parametrize('tau', [0.01])
@pytest.mark.parametrize('seed', [21])
def test_dmd_3d(seed,n,t,c,tau):
    rng = np.random.default_rng(seed)
    x0 = rng.random((n,c))
    data = np.zeros((n,t,c))
    data[:,0] = x0
    Q = ortho_group.rvs(c)
    A = np.eye(c) + tau*Q 
    #\dot{x} = Qx -> x_t+1 ~= x + \tauQx
    for i in range(1,t):
        data[:,i] = np.einsum('nn,cn->cn',A,data[:,i-1])
    dmd = DMD(data,1)
    dmd.fit()
    assert np.linalg.norm(dmd.A_v.flatten()-A.flatten()) < 1e-1


@pytest.mark.parametrize('c', [10])
@pytest.mark.parametrize('t', [100])
@pytest.mark.parametrize('n', [50])
@pytest.mark.parametrize('seed', [21])
def test_to_cpu(seed,n,t,c):
    rng = np.random.default_rng(seed) 
    X = rng.random((n,t,c))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dmd = DMD(X,1,device=device)
    dmd.fit(send_to_cpu=True)
    assert dmd.A_v.device.type == 'cpu'
    assert dmd.H.device.type == 'cpu'
