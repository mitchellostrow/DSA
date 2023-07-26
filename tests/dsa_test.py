import pytest
import numpy as np
from DSA import DSA
from scipy.stats import ortho_group

@pytest.mark.parametrize('seed', [5])
@pytest.mark.parametrize('n1', [2])
@pytest.mark.parametrize('n2', [10])
@pytest.mark.parametrize('t1', [1000])
@pytest.mark.parametrize('t2', [1000])
@pytest.mark.parametrize('c1', [5])
@pytest.mark.parametrize('c2', [10]) #only these really need to be different
@pytest.mark.parametrize('rank1', [5])
@pytest.mark.parametrize('rank2', [5,10])
def test_different_dims(seed,n1,n2,t1,t2,c1,c2,rank1,rank2):
    rng = np.random.default_rng(seed) 
    X = rng.random((n1,t1,c1))
    Y = rng.random((n2,t2,c2))
    dsa = DSA(X,Y,rank=(rank1,rank2),n_delays=10)
    sim = dsa.fit_score()
    assert dsa.dmds[0][0].A_v.shape == (rank1,rank1)
    assert dsa.dmds[1][0].A_v.shape == (rank2,rank2)

@pytest.mark.parametrize('seed', [5])
@pytest.mark.parametrize('n_delays', [10,[[10,20,30],[5,10,15]]]) #only need to test 1 param
def test_param_broadcasting_1(seed,n_delays):
    rng = np.random.default_rng(seed) 
    d1 = rng.random((100,50,10))
    dsa = DSA(d1,d1,n_delays=n_delays)
    if isinstance(n_delays,list):
        delay1 = n_delays[0][0]
        delay2 = n_delays[1][0]
    else:
        delay1,delay2 = n_delays,n_delays
    assert dsa.dmds[0][0].n_delays == delay1
    assert dsa.dmds[1][0].n_delays == delay2
    assert len(dsa.dmds) == 2
    assert len(dsa.dmds[0]) == 1

@pytest.mark.parametrize('n', [2,5])
@pytest.mark.parametrize('seed', [5])
@pytest.mark.parametrize('n_delays', [10,[10,20,30,40,50]])
def test_param_broadcasting_list(seed,n,n_delays):
    rng = np.random.default_rng(seed) 
    ds = [rng.random((100,50,10)) for i in range(n)]
    dsa = DSA(ds,n_delays=n_delays)
    for i in range(n):
        if isinstance(n_delays,int):
            assert dsa.dmds[0][i].n_delays == n_delays
        else:
            assert dsa.dmds[0][i].n_delays == n_delays[i]
    assert len(dsa.dmds[0]) == n

@pytest.mark.parametrize('n1', [2,5])
@pytest.mark.parametrize('n2', [3,4])
@pytest.mark.parametrize('seed', [5])
@pytest.mark.parametrize('n_delays1', [10,[10,20,30,40,50]])
@pytest.mark.parametrize('n_delays2', [11,[11,21,31,41,51]])
def test_param_broadcasting_2lists(seed,n1,n2,n_delays1,n_delays2):
    rng = np.random.default_rng(seed) 
    ds1 = [rng.random((100,50,10)) for i in range(n1)]
    ds2 = [rng.random((100,50,10)) for i in range(n2)]
    dsa = DSA(ds1,ds2,n_delays=(n_delays1,n_delays2))
    itrable = zip([n1,n2],[n_delays1,n_delays2])
    for j,(n,delays) in enumerate(itrable):
        for i in range(n):
            if isinstance(delays,int):
                assert dsa.dmds[j][i].n_delays == delays
            else:
                assert dsa.dmds[j][i].n_delays == delays[i]
    assert len(dsa.dmds[0]) == n1
    assert len(dsa.dmds[1]) == n2

# def test_multiple_param_variations(seed,n,n_delays,rank):
#     rng = np.random.default_rng(seed) 
#     ds = [rng.random((100,50,10)) for i in range(n)]
#     dsa = DSA(ds,n_delays=n_delays)
        
@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('c', [2])
@pytest.mark.parametrize('t', [100])
@pytest.mark.parametrize('seed', [5])
def test_dsa_1to1(n,t,c,seed):
    rng = np.random.default_rng(seed) 
    X = rng.random((n,t,c))
    Y = rng.random((n,t,c))
    dsa = DSA(X,Y)
    sim = dsa.fit_score()
    assert isinstance(sim,float)

@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('c', [2])
@pytest.mark.parametrize('t', [100])
@pytest.mark.parametrize('seed', [5])
@pytest.mark.parametrize('nmodels', [10])
def test_dsa_1tomany(n,t,c,seed,nmodels):
    rng = np.random.default_rng(seed) 
    X = [rng.random((n,t,c)) for i in range(nmodels)]
    Y = rng.random((n,t,c))
    dsa = DSA(X,Y)
    sim = dsa.fit_score()
    assert isinstance(sim,np.ndarray)
    assert sim.shape == (nmodels,1)

@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('c', [2])
@pytest.mark.parametrize('t', [100])
@pytest.mark.parametrize('seed', [5])
@pytest.mark.parametrize('nmodels', [10])
def test_dsa_manyto1(n,t,c,seed,nmodels):
    rng = np.random.default_rng(seed) 
    X = [rng.random((n,t,c)) for i in range(nmodels)]
    Y = rng.random((n,t,c))
    dsa = DSA(Y,X)
    sim = dsa.fit_score()
    assert isinstance(sim,np.ndarray)
    assert sim.shape == (1,nmodels)

@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('c', [2])
@pytest.mark.parametrize('t', [100])
@pytest.mark.parametrize('seed', [5])
@pytest.mark.parametrize('nmodels1', [2])
@pytest.mark.parametrize('nmodels2', [2])
def test_dsa_manytomany(n,t,c,seed,nmodels1,nmodels2):
    rng = np.random.default_rng(seed) 
    X = [rng.random((n,t,c)) for i in range(nmodels1)]
    Y = [rng.random((n,t,c)) for i in range(nmodels2)]
    dsa = DSA(X,Y)
    sim = dsa.fit_score()
    print(sim.shape)
    assert isinstance(sim,np.ndarray)
    assert sim.shape == (nmodels1,nmodels2)
