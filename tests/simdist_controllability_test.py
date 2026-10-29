"""Tests for ControllabilitySimilarityTransformDist (the InputDSA joint score).

Test for Issue #12
"""
import numpy as np
import pytest
from scipy.linalg import orthogonal_procrustes

from DSA.simdist_controllability import ControllabilitySimilarityTransformDist

TOL = 1e-6


def _random_system(rng, n, m):
    A = rng.normal(size=(n, n))
    A *= 0.9 / np.abs(np.linalg.eigvals(A)).max()
    B = rng.normal(size=(n, m))
    return A, B


def _random_rotation(rng, n):
    # A generic orthogonal matrix. The identity or a diagonal of sign flips
    # satisfies R @ R == I and would not expose a C / C.T mix-up.
    R, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return R


@pytest.mark.parametrize("score_method", ["euclidean", "angular"])
@pytest.mark.parametrize("return_distance_components", [False, True])
@pytest.mark.parametrize("n,m", [(5, 2), (6, 3)])
@pytest.mark.parametrize("seed", [0, 1])
def test_joint_distance_invariant_to_latent_rotation(
    seed, n, m, score_method, return_distance_components
):
    rng = np.random.default_rng(seed)
    A, B = _random_system(rng, n, m)
    R = _random_rotation(rng, n)

    dist = ControllabilitySimilarityTransformDist(
        score_method=score_method,
        compare="joint",
        return_distance_components=return_distance_components,
    )
    score = dist.fit_score(A, R @ A @ R.T, B, R @ B)

    assert np.allclose(np.asarray(score), 0.0, atol=TOL)


# noinspection PyPep8Naming
@pytest.mark.parametrize("n,m", [(6, 3)])
@pytest.mark.parametrize("seed", [1])
def test_returned_C_maps_system1_onto_system2(seed, n, m):
    rng = np.random.default_rng(seed)
    A, B = _random_system(rng, n, m)
    R = _random_rotation(rng, n)
    A2, B2 = R @ A @ R.T, R @ B

    dist = ControllabilitySimilarityTransformDist(compare="joint")
    C, _, _, _ = dist.compare_systems_procrustes(A1=A, B1=B, A2=A2, B2=B2)
    K1 = dist.get_controllability_matrix(A, B)
    K2 = dist.get_controllability_matrix(A2, B2)

    assert np.allclose(C.T @ C, np.eye(n), atol=TOL)
    assert np.allclose(C, R, atol=TOL)
    assert np.allclose(C @ K1, K2, atol=TOL)


@pytest.mark.parametrize("n,m", [(6, 3)])
@pytest.mark.parametrize("seed", [7])
def test_joint_residual_matches_scipy_procrustes(seed, n, m):
    rng = np.random.default_rng(seed)
    A1, B1 = _random_system(rng, n, m)
    A2, B2 = _random_system(rng, n, m)

    dist = ControllabilitySimilarityTransformDist(compare="joint")
    _, _, err, _ = dist.compare_systems_procrustes(A1=A1, B1=B1, A2=A2, B2=B2)
    K1 = dist.get_controllability_matrix(A1, B1)
    K2 = dist.get_controllability_matrix(A2, B2)

    # scipy solves min_R ||X R - Y||. With X = K1.T, Y = K2.T this is
    # min_R ||R.T K1 - K2||, so the optimal left transform is R.T.
    R_opt, _ = orthogonal_procrustes(K1.T, K2.T)
    expected = np.linalg.norm(R_opt.T @ K1 - K2, "fro")

    assert err > 0
    assert np.isclose(err, expected, atol=TOL)
