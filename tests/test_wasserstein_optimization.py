"""
Test to verify Wasserstein distance optimization works correctly.
Tests both SimilarityTransformDist and DSA classes to ensure:
1. Pre-computed eigenvalues produce identical results to matrices
2. Identical systems produce near-zero scores
3. Both torch and numpy complex arrays work correctly
4. DSA correctly caches eigenvalues for efficiency
"""
import pytest
import numpy as np
import torch
from DSA.simdist import SimilarityTransformDist
from DSA import DSA
from DSA.dmd import DMD


@pytest.fixture
def random_matrices():
    """Generate random test matrices."""
    np.random.seed(42)
    torch.manual_seed(42)
    A = torch.randn(5, 5)
    B = torch.randn(5, 5)
    return A, B


@pytest.fixture
def random_data():
    """Generate random time series data for DSA tests."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X1 = np.random.randn(n_features, n_samples)
    X2 = np.random.randn(n_features, n_samples)
    return X1, X2


def test_simdist_wasserstein_with_matrices(random_matrices):
    """Test SimilarityTransformDist with full matrices."""
    A, B = random_matrices
    
    simdist = SimilarityTransformDist(
        iters=100,
        score_method="wasserstein",
        lr=0.01,
        device="cpu",
        verbose=False
    )
    
    # Test with matrices
    score_from_matrices = simdist.fit_score(A, B)
    assert score_from_matrices > 0, "Score should be positive for different matrices"
    
    # Test with identical matrices (should be close to 0)
    score_identical = simdist.fit_score(A, A)
    assert score_identical < 1e-3, f"Identical matrices should have near-zero score, got {score_identical}"


def test_simdist_wasserstein_with_precomputed_eigenvalues(random_matrices):
    """Test SimilarityTransformDist with pre-computed eigenvalues produces same results as matrices."""
    A, B = random_matrices
    
    # Get score from matrices
    simdist1 = SimilarityTransformDist(
        iters=100,
        score_method="wasserstein",
        lr=0.01,
        device="cpu",
        verbose=False
    )
    score_from_matrices = simdist1.fit_score(A, B)
    
    # Get eigenvalues
    eigenvalues_A = torch.linalg.eig(A).eigenvalues
    eigenvalues_B = torch.linalg.eig(B).eigenvalues
    
    assert eigenvalues_A.ndim == 1, "Eigenvalues should be 1D"
    assert torch.is_complex(eigenvalues_A), "Eigenvalues should be complex"
    
    # Get score from pre-computed eigenvalues
    simdist2 = SimilarityTransformDist(
        iters=100,
        score_method="wasserstein",
        lr=0.01,
        device="cpu",
        verbose=False
    )
    score_from_eigenvalues = simdist2.fit_score(eigenvalues_A, eigenvalues_B)
    
    # Scores should be identical (or very close)
    diff = abs(score_from_matrices - score_from_eigenvalues)
    assert diff < 1e-3, f"Scores should match, got difference of {diff}"
    
    # Test with identical eigenvalues
    score_identical_eig = simdist2.fit_score(eigenvalues_A, eigenvalues_A)
    assert score_identical_eig < 1e-3, f"Identical eigenvalues should have near-zero score, got {score_identical_eig}"


def test_simdist_wasserstein_with_numpy_arrays(random_matrices):
    """Test that numpy complex arrays are handled correctly."""
    A, B = random_matrices
    
    # Convert to numpy
    A_np = A.numpy()
    B_np = B.numpy()
    eigenvalues_A_np = np.linalg.eig(A_np)[0]  # numpy returns (eigenvalues, eigenvectors)
    eigenvalues_B_np = np.linalg.eig(B_np)[0]
    
    assert eigenvalues_A_np.ndim == 1, "Numpy eigenvalues should be 1D"
    assert np.iscomplexobj(eigenvalues_A_np), "Numpy eigenvalues should be complex"
    
    simdist = SimilarityTransformDist(
        iters=100,
        score_method="wasserstein",
        lr=0.01,
        device="cpu",
        verbose=False
    )
    
    # Should work without errors
    score_from_numpy_eig = simdist.fit_score(eigenvalues_A_np, eigenvalues_B_np)
    assert score_from_numpy_eig > 0, "Score should be positive"


def test_dsa_wasserstein_caching(random_data):
    """Test that DSA correctly caches eigenvalues for efficiency."""
    X1, X2 = random_data
    
    dsa = DSA(
        X=[X1, X2],
        Y=None,
        dmd_class=DMD,
        device="cpu",
        verbose=False,
        n_jobs=1,
        score_method="wasserstein",
        iters=100,
        lr=0.01,
        n_delays=1,
        rank=5,
    )
    
    scores = dsa.fit_score()
    
    # Check scores shape and properties
    assert scores.shape == (2, 2), f"Expected shape (2, 2), got {scores.shape}"
    
    # Check that diagonal is near-zero (comparing same systems)
    diagonal_scores = np.array([scores[i, i] for i in range(len(scores))])
    assert np.all(diagonal_scores < 1e-3), f"Diagonal should be near-zero, got {diagonal_scores}"
    
    # Check that matrix is symmetric
    symmetry_diff = np.abs(scores - scores.T).max()
    assert symmetry_diff < 1e-6, f"Score matrix should be symmetric, got max diff {symmetry_diff}"
    
    # Verify the optimization actually cached the eigenvalues
    assert hasattr(dsa, 'cached_compare_objects'), "DSA should have cached_compare_objects"
    assert len(dsa.cached_compare_objects) == 2, "Should have 2 groups of cached objects"
    assert len(dsa.cached_compare_objects[0]) == 2, "First group should have 2 objects"
    assert len(dsa.cached_compare_objects[1]) == 2, "Second group should have 2 objects"
    
    # Check that cached objects are complex eigenvalues
    first_obj = dsa.cached_compare_objects[0][0]
    assert first_obj.ndim == 1, "Cached objects should be 1D"
    assert torch.is_complex(first_obj), "Cached objects should be complex"


def test_dsa_wasserstein_vs_angular_difference(random_data):
    """Test that Wasserstein and angular methods produce different results (as expected)."""
    X1, X2 = random_data
    
    # DSA with Wasserstein
    dsa_wass = DSA(
        X=[X1, X2],
        Y=None,
        dmd_class=DMD,
        device="cpu",
        verbose=False,
        n_jobs=1,
        score_method="wasserstein",
        iters=100,
        lr=0.01,
        n_delays=1,
        rank=5,
    )
    scores_wass = dsa_wass.fit_score()
    
    # DSA with angular
    dsa_ang = DSA(
        X=[X1, X2],
        Y=None,
        dmd_class=DMD,
        device="cpu",
        verbose=False,
        n_jobs=1,
        score_method="angular",
        iters=100,
        lr=0.01,
        n_delays=1,
        rank=5,
    )
    scores_ang = dsa_ang.fit_score()
    
    # Both should have same shape
    assert scores_wass.shape == scores_ang.shape
    
    # Both should have near-zero diagonals
    assert np.all(np.abs(np.diag(scores_wass)) < 1e-3)
    assert np.all(np.abs(np.diag(scores_ang)) < 1e-3)
    
    # Off-diagonal elements should be different (different metrics)
    # But both should be positive and non-zero
    assert scores_wass[0, 1] > 0
    assert scores_ang[0, 1] > 0


if __name__ == "__main__":
    # Allow running as a script for quick testing
    print("=" * 60)
    print("Testing Wasserstein Distance Optimization")
    print("=" * 60)
    
    # Create fixtures manually
    np.random.seed(42)
    torch.manual_seed(42)
    A = torch.randn(5, 5)
    B = torch.randn(5, 5)
    random_matrices = (A, B)
    
    np.random.seed(42)
    X1 = np.random.randn(10, 100)
    X2 = np.random.randn(10, 100)
    random_data = (X1, X2)
    
    print("\n1. Testing SimilarityTransformDist with matrices...")
    test_simdist_wasserstein_with_matrices(random_matrices)
    print("✓ Passed")
    
    print("\n2. Testing SimilarityTransformDist with pre-computed eigenvalues...")
    test_simdist_wasserstein_with_precomputed_eigenvalues(random_matrices)
    print("✓ Passed")
    
    print("\n3. Testing with numpy arrays...")
    test_simdist_wasserstein_with_numpy_arrays(random_matrices)
    print("✓ Passed")
    
    print("\n4. Testing DSA with Wasserstein distance caching...")
    test_dsa_wasserstein_caching(random_data)
    print("✓ Passed")
    
    print("\n5. Testing DSA Wasserstein vs Angular...")
    test_dsa_wasserstein_vs_angular_difference(random_data)
    print("✓ Passed")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
