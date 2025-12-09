"""
Test suite for PyKoopman with 3D data and various observables.
Tests the implementation that prevents trial boundary crossing.
"""
import pytest
import numpy as np
from DSA.pykoopman.observables import (
    TimeDelay, Identity, Polynomial,
    RadialBasisFunction, RandomFourierFeatures
)
from DSA.pykoopman import Koopman
from pydmd import DMD as pDMD

TOL = 1e-10  # Tolerance for Frobenius norm comparisons


@pytest.fixture
def basic_3d_data():
    """Create basic 3D test data with distinct values per trial."""
    np.random.seed(42)
    n_trials = 3
    n_timesteps = 10
    n_features = 2
    
    data = np.zeros((n_trials, n_timesteps, n_features))
    for trial_idx in range(n_trials):
        for t in range(n_timesteps):
            data[trial_idx, t, 0] = trial_idx * 100 + t
            data[trial_idx, t, 1] = trial_idx * 100 + t + 0.5
    
    return data


@pytest.fixture
def variable_length_data():
    """Create list of 3D arrays with different shapes."""
    np.random.seed(42)
    n_features = 2
    
    # First array: 2 trials, 12 timesteps each
    data_var1 = np.zeros((2, 12, n_features))
    for trial_idx in range(2):
        for t in range(12):
            data_var1[trial_idx, t, 0] = trial_idx * 100 + t
            data_var1[trial_idx, t, 1] = trial_idx * 100 + t + 0.5
    
    # Second array: 3 trials, 8 timesteps each
    data_var2 = np.zeros((3, 8, n_features))
    for trial_idx in range(3):
        for t in range(8):
            data_var2[trial_idx, t, 0] = (trial_idx + 10) * 100 + t
            data_var2[trial_idx, t, 1] = (trial_idx + 10) * 100 + t + 0.5
    
    return [data_var1, data_var2]


def compute_manual_embedding(data, n_delays, rank):
    """
    Manually embed data with TimeDelay to get ground truth.
    This is the reference implementation.
    """
    delay_computer = TimeDelay(n_delays=n_delays)
    X_list = []
    Y_list = []
    
    # Handle 3D array
    if isinstance(data, np.ndarray) and data.ndim == 3:
        for trial_idx in range(data.shape[0]):
            traj = data[trial_idx, :, :]
            embedded_traj = delay_computer.fit_transform(traj)
            X_list.append(embedded_traj[:-1])
            Y_list.append(embedded_traj[1:])
    
    # Handle list of arrays
    elif isinstance(data, list):
        for data_array in data:
            if data_array.ndim == 3:
                # List of 3D arrays
                for trial_idx in range(data_array.shape[0]):
                    traj = data_array[trial_idx, :, :]
                    embedded_traj = delay_computer.fit_transform(traj)
                    X_list.append(embedded_traj[:-1])
                    Y_list.append(embedded_traj[1:])
            elif data_array.ndim == 2:
                # List of 2D arrays
                embedded_traj = delay_computer.fit_transform(data_array)
                X_list.append(embedded_traj[:-1])
                Y_list.append(embedded_traj[1:])
    
    X_all = np.vstack(X_list)
    Y_all = np.vstack(Y_list)
    
    # Fit with Identity since embedding already done
    k_manual = Koopman(
        observables=Identity(),
        regressor=pDMD(svd_rank=rank)
    )
    k_manual.fit(X_all, Y_all)
    
    return k_manual.A


class TestKoopman3DData:
    """Test Koopman with 3D data structures."""
    
    def test_3d_timedelay_vs_manual(self, basic_3d_data):
        """Test that 3D input with TimeDelay matches manual approach."""
        n_delays = 3
        rank = 5
        
        # Manual approach (ground truth)
        A_manual = compute_manual_embedding(basic_3d_data, n_delays, rank)
        
        # New implementation
        k_new = Koopman(
            observables=TimeDelay(n_delays=n_delays),
            regressor=pDMD(svd_rank=rank)
        )
        k_new.fit(basic_3d_data)
        A_new = k_new.A
        
        # Compare
        diff = np.linalg.norm(A_manual - A_new, 'fro')
        assert diff < TOL, f"Matrices differ by {diff:.10e}"
        assert A_manual.shape == A_new.shape
    
    def test_2d_timedelay_backward_compatible(self):
        """Test that 2D input still works (backward compatibility)."""
        np.random.seed(42)
        data_2d = np.random.randn(10, 2)
        
        k = Koopman(
            observables=TimeDelay(n_delays=3),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(data_2d)
        
        assert k.A.shape == (5, 5)
    
    def test_list_of_2d_timedelay(self, basic_3d_data):
        """Test list of 2D arrays with TimeDelay."""
        n_delays = 3
        rank = 5
        
        # Convert 3D to list of 2D
        data_list = [basic_3d_data[i, :, :] for i in range(basic_3d_data.shape[0])]
        
        # Manual approach
        A_manual = compute_manual_embedding(basic_3d_data, n_delays, rank)
        
        # List approach
        k_list = Koopman(
            observables=TimeDelay(n_delays=n_delays),
            regressor=pDMD(svd_rank=rank)
        )
        k_list.fit(data_list)
        A_list = k_list.A
        
        # Compare
        diff = np.linalg.norm(A_manual - A_list, 'fro')
        assert diff < TOL, f"List approach differs from manual by {diff:.10e}"


class TestKoopmanObservables:
    """Test various observables with 3D data."""
    
    def test_identity_3d(self, basic_3d_data):
        """Test Identity observable with 3D data."""
        k = Koopman(
            observables=Identity(),
            regressor=pDMD(svd_rank=2)
        )
        k.fit(basic_3d_data)
        assert k.A.shape == (2, 2)
    
    def test_polynomial_3d(self, basic_3d_data):
        """Test Polynomial observable with 3D data."""
        k = Koopman(
            observables=Polynomial(degree=2),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(basic_3d_data)
        assert k.A.shape == (5, 5)
    
    def test_rbf_3d(self, basic_3d_data):
        """Test RadialBasisFunction observable with 3D data."""
        k = Koopman(
            observables=RadialBasisFunction(n_centers=5, rbf_type='gauss'),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(basic_3d_data)
        assert k.A.shape == (5, 5)
    
    def test_rff_3d(self, basic_3d_data):
        """Test RandomFourierFeatures observable with 3D data."""
        k = Koopman(
            observables=RandomFourierFeatures(D=10, random_state=42),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(basic_3d_data)
        assert k.A.shape == (5, 5)


class TestKoopmanVariableLength:
    """Test Koopman with variable-length trials."""
    
    def test_variable_length_timedelay(self, variable_length_data):
        """Test TimeDelay with list of different-shaped 3D arrays."""
        n_delays = 3
        rank = 5
        
        # Manual approach
        A_manual = compute_manual_embedding(variable_length_data, n_delays, rank)
        
        # New implementation
        k = Koopman(
            observables=TimeDelay(n_delays=n_delays),
            regressor=pDMD(svd_rank=rank)
        )
        k.fit(variable_length_data)
        A_new = k.A
        
        # Compare
        diff = np.linalg.norm(A_manual - A_new, 'fro')
        assert diff < TOL, f"Variable-length differs from manual by {diff:.10e}"
    
    def test_variable_length_sample_count(self, variable_length_data):
        """Verify correct number of samples (no boundary crossing)."""
        n_delays = 3
        
        k = Koopman(
            observables=TimeDelay(n_delays=n_delays),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(variable_length_data)
        
        # Array 1: 2 trials * (12 - n_delays - 1) = 2 * 8 = 16
        # Array 2: 3 trials * (8 - n_delays - 1) = 3 * 4 = 12
        # Total: 28 samples
        expected_samples = 2 * (12 - n_delays - 1) + 3 * (8 - n_delays - 1)
        actual_samples = k._regressor().n_samples_
        
        assert actual_samples == expected_samples, \
            f"Expected {expected_samples} samples, got {actual_samples}"
    
    def test_variable_length_identity(self, variable_length_data):
        """Test Identity with variable-length trials."""
        k = Koopman(
            observables=Identity(),
            regressor=pDMD(svd_rank=2)
        )
        k.fit(variable_length_data)
        assert k.A.shape == (2, 2)
    
    def test_variable_length_polynomial(self, variable_length_data):
        """Test Polynomial with variable-length trials."""
        k = Koopman(
            observables=Polynomial(degree=2),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(variable_length_data)
        assert k.A.shape == (5, 5)


class TestKoopmanPrediction:
    """Test prediction functionality."""
    
    def test_predict_timedelay(self, basic_3d_data):
        """Test prediction with TimeDelay observable."""
        n_delays = 3
        
        k = Koopman(
            observables=TimeDelay(n_delays=n_delays),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(basic_3d_data)
        
        # Need n_delays + 1 samples for prediction with TimeDelay
        test_point = basic_3d_data[0, 0:n_delays+1, :]
        pred = k.predict(test_point)
        
        assert pred.shape == (1, 2), f"Expected shape (1, 2), got {pred.shape}"
    
    def test_predict_identity(self, basic_3d_data):
        """Test prediction with Identity observable."""
        k = Koopman(
            observables=Identity(),
            regressor=pDMD(svd_rank=2)
        )
        k.fit(basic_3d_data)
        
        test_point = basic_3d_data[0, 0:1, :]
        pred = k.predict(test_point)
        
        assert pred.shape == (1, 2), f"Expected shape (1, 2), got {pred.shape}"
    
    def test_predict_polynomial(self, basic_3d_data):
        """Test prediction with Polynomial observable."""
        k = Koopman(
            observables=Polynomial(degree=2),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(basic_3d_data)
        
        test_point = basic_3d_data[0, 0:1, :]
        pred = k.predict(test_point)
        
        assert pred.shape == (1, 2), f"Expected shape (1, 2), got {pred.shape}"
    
    def test_predict_rbf(self, basic_3d_data):
        """Test prediction with RadialBasisFunction observable."""
        k = Koopman(
            observables=RadialBasisFunction(n_centers=5, rbf_type='gauss'),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(basic_3d_data)
        
        test_point = basic_3d_data[0, 0:1, :]
        pred = k.predict(test_point)
        
        assert pred.shape == (1, 2), f"Expected shape (1, 2), got {pred.shape}"
    
    def test_predict_rff(self, basic_3d_data):
        """Test prediction with RandomFourierFeatures observable."""
        k = Koopman(
            observables=RandomFourierFeatures(D=10, random_state=42),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(basic_3d_data)
        
        test_point = basic_3d_data[0, 0:1, :]
        pred = k.predict(test_point)
        
        assert pred.shape == (1, 2), f"Expected shape (1, 2), got {pred.shape}"
    
    def test_predict_multiple_steps(self, basic_3d_data):
        """Test prediction with multiple samples."""
        k = Koopman(
            observables=Identity(),
            regressor=pDMD(svd_rank=2)
        )
        k.fit(basic_3d_data)
        
        # Predict for multiple samples
        test_points = basic_3d_data[0, 0:3, :]
        pred = k.predict(test_points)
        
        assert pred.shape == (3, 2), f"Expected shape (3, 2), got {pred.shape}"
    
    def test_predict_after_variable_length_fit(self, variable_length_data):
        """Test prediction after fitting on variable-length trials."""
        k = Koopman(
            observables=Identity(),
            regressor=pDMD(svd_rank=2)
        )
        k.fit(variable_length_data)
        
        # Predict on new data
        test_point = variable_length_data[0][0, 0:1, :]
        pred = k.predict(test_point)
        
        assert pred.shape == (1, 2), f"Expected shape (1, 2), got {pred.shape}"


class TestKoopmanNoBoundaryCrossing:
    """Test that trial boundaries are never crossed."""
    
    def test_no_boundary_crossing_in_embedding(self, basic_3d_data):
        """
        Verify that time-delay windows never span across trials.
        We do this by checking that manually processing each trial
        independently gives the same result as the automatic 3D processing.
        """
        n_delays = 3
        rank = 5
        
        # Manual approach: explicitly process each trial
        A_manual = compute_manual_embedding(basic_3d_data, n_delays, rank)
        
        # Automatic 3D approach
        k = Koopman(
            observables=TimeDelay(n_delays=n_delays),
            regressor=pDMD(svd_rank=rank)
        )
        k.fit(basic_3d_data)
        A_auto = k.A
        
        # If boundaries were crossed, the matrices would differ
        diff = np.linalg.norm(A_manual - A_auto, 'fro')
        assert diff < TOL, \
            f"Boundary crossing detected! Manual and auto differ by {diff:.10e}"
    
    def test_sample_count_matches_expectation(self, basic_3d_data):
        """Verify that the number of samples matches expected value."""
        n_delays = 3
        n_trials = basic_3d_data.shape[0]
        n_timesteps = basic_3d_data.shape[1]
        
        k = Koopman(
            observables=TimeDelay(n_delays=n_delays),
            regressor=pDMD(svd_rank=5)
        )
        k.fit(basic_3d_data)
        
        # Each trial contributes (n_timesteps - n_delays - 1) samples
        expected_samples = n_trials * (n_timesteps - n_delays - 1)
        actual_samples = k._regressor().n_samples_
        
        assert actual_samples == expected_samples, \
            f"Expected {expected_samples} samples, got {actual_samples}"


@pytest.mark.parametrize("n_delays", [1, 2, 3, 5])
@pytest.mark.parametrize("rank", [2, 5])
def test_parametrized_timedelay(basic_3d_data, n_delays, rank):
    """Test various combinations of n_delays and rank."""
    k = Koopman(
        observables=TimeDelay(n_delays=n_delays),
        regressor=pDMD(svd_rank=rank)
    )
    k.fit(basic_3d_data)
    
    # Should successfully fit without errors
    # Note: actual rank may be less than requested if not enough features
    n_features = basic_3d_data.shape[2]
    n_output_features = n_features * (1 + n_delays)
    expected_rank = min(rank, n_output_features)
    assert k.A.shape == (expected_rank, expected_rank), \
        f"Expected shape ({expected_rank}, {expected_rank}), got {k.A.shape}"
    
    # Verify we can predict
    test_point = basic_3d_data[0, 0:n_delays+1, :]
    pred = k.predict(test_point)
    assert pred.shape[1] == basic_3d_data.shape[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

