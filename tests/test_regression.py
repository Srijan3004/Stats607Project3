import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from dgps import generate_freedman_population, generate_nonlinear_population, randomize_treatment
from methods import estimate_effects as baseline
from optimized import estimate_effects as optimized
from runner import run

@pytest.mark.parametrize('dgp', [generate_freedman_population, generate_nonlinear_population])
@pytest.mark.parametrize('n', [100,500,1000,5000])
def test_paired_estimators(dgp, n):
    z,c,t = dgp(n, seed=1234)
    for seed in range(10):
        x,y = randomize_treatment(c,t,seed=seed)
        before=baseline(x,z,y)
        after,info=optimized(x,z,y,diagnostics=True)
        assert info['status']=='ok'
        np.testing.assert_allclose(after,before,atol=1e-7,rtol=1e-7,equal_nan=True)

def test_parallel_reproducibility():
    a=run(reps=5,ns=[100,500],workers=1)
    b=run(reps=5,ns=[100,500],workers=2)
    assert a.equals(b)
    assert a.equals(run(reps=5,ns=[100,500],workers=1))

@pytest.mark.parametrize('kind',['empty_arm','constant','rank','separated'])
def test_degenerate(kind):
    x=np.tile([0,1],50);z=np.linspace(-2,2,100);y=np.tile([0,1,1,0],25)
    if kind=='empty_arm': x[:]=1
    if kind=='constant': y[:]=0
    if kind=='rank': z[:]=0
    if kind=='separated': y=(z>0).astype(int)
    result,info=optimized(x,z,y,diagnostics=True)
    assert info['status']!='ok'
    assert np.isnan(result[0]) and np.isnan(result[1])

def test_covariate_rescaling():
    z,c,t=generate_nonlinear_population(1000,seed=55)
    x,y=randomize_treatment(c,t,seed=66)
    np.testing.assert_allclose(optimized(x,z,y),optimized(x,1e8+1e5*z,y),atol=1e-8)

def test_invalid_input():
    with pytest.raises(ValueError): optimized([0,1],[np.nan,0],[0,1])
