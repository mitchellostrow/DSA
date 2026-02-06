"""
Tests for sweeps module: LocalDMDSweeper, PyKoopmanSweeper, and convenience functions.
"""

import pytest
import numpy as np
from DSA.sweeps import (
    LocalDMDSweeper,
    PyKoopmanSweeper,
    sweep_local_dmd,
    sweep_pykoopman,
    sweep_ranks_delays,
    split_train_test,
)
from DSA.dmd import DMD


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def data_2d():
    """2D data: (time, dim)"""
    np.random.seed(42)
    return np.random.randn(200, 5)


@pytest.fixture
def data_3d():
    """3D data: (trials, time, dim)"""
    np.random.seed(42)
    return np.random.randn(10, 100, 5)


@pytest.fixture
def data_list_2d():
    """List of 2D arrays"""
    np.random.seed(42)
    return [np.random.randn(100, 5) for _ in range(5)]


@pytest.fixture
def data_list_3d():
    """List of 3D arrays"""
    np.random.seed(42)
    return [np.random.randn(3, 80, 5) for _ in range(4)]


# =============================================================================
# Test split_train_test
# =============================================================================

class TestSplitTrainTest:
    
    def test_split_2d(self, data_2d):
        train, test, dim = split_train_test(data_2d, train_frac=0.8)
        assert train.shape[0] == 160
        assert test.shape[0] == 40
        assert dim == 5
    
    def test_split_3d(self, data_3d):
        train, test, dim = split_train_test(data_3d, train_frac=0.8)
        # For 3D with shape[0] > 1, splits along first dimension
        assert dim == 5
    
    def test_split_list(self, data_list_2d):
        train, test, dim = split_train_test(data_list_2d, train_frac=0.8)
        assert isinstance(train, list)
        assert isinstance(test, list)
        assert len(train) == 4  # 80% of 5
        assert len(test) == 1
        assert dim == 5


# =============================================================================
# Test LocalDMDSweeper
# =============================================================================

class TestLocalDMDSweeper:
    
    def test_sweeper_creation_2d(self, data_2d):
        """Test sweeper can be created with 2D data."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2, 3],
            param2_values=[5, 10],
        )
        assert sweeper.param1_name == "n_delays"
        assert sweeper.param2_name == "rank"
        assert len(sweeper.param1_values) == 2
        assert len(sweeper.param2_values) == 2
    
    def test_sweeper_creation_3d(self, data_3d):
        """Test sweeper can be created with 3D data."""
        sweeper = LocalDMDSweeper(
            data=data_3d,
            param1_values=[2, 3],
            param2_values=[5, 10],
        )
        assert sweeper.dim == 5
    
    def test_sweep_runs_2d(self, data_2d):
        """Test sweep completes on 2D data."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2, 3],
            param2_values=[5, 8],
            train_frac=0.8,
        )
        sweeper.sweep()
        
        assert sweeper._swept
        assert sweeper.aics.shape == (2, 2)
        assert sweeper.mases.shape == (2, 2)
        assert sweeper.mses.shape == (2, 2)
        assert sweeper.nnormals.shape == (2, 2)
    
    def test_sweep_runs_3d(self, data_3d):
        """Test sweep completes on 3D data."""
        sweeper = LocalDMDSweeper(
            data=data_3d,
            param1_values=[2, 3],
            param2_values=[5, 8],
        )
        sweeper.sweep()
        
        assert sweeper._swept
        assert sweeper.aics.shape == (2, 2)
    
    def test_sweep_with_residuals(self, data_2d):
        """Test sweep with residual computation."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2],
            param2_values=[5],
            compute_residuals_flag=True,
        )
        sweeper.sweep()
        
        assert sweeper.residuals is not None
        assert sweeper.residuals.shape == (1, 1)
        assert not np.isnan(sweeper.residuals[0, 0])
    
    def test_invalid_rank_skipped(self, data_2d):
        """Test that invalid rank combinations are skipped."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2],  # n_delays=2, dim=5 -> max_rank=10
            param2_values=[5, 15],  # 15 > 10, should be skipped
        )
        sweeper.sweep()
        
        assert not np.isnan(sweeper.aics[0, 0])  # rank=5 valid
        assert np.isnan(sweeper.aics[0, 1])  # rank=15 invalid
    
    def test_get_results(self, data_2d):
        """Test get_results returns correct structure."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2, 3],
            param2_values=[5, 8],
        )
        sweeper.sweep()
        
        results = sweeper.get_results()
        assert "param1_name" in results
        assert "param1_values" in results
        assert "aics" in results
        assert "mases" in results
        assert "mses" in results
        assert "nnormals" in results
        assert results["param1_name"] == "n_delays"
        assert results["param2_name"] == "rank"
    
    def test_fitted_models_stored(self, data_2d):
        """Test that fitted models are stored."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2],
            param2_values=[5],
        )
        sweeper.sweep()
        
        model = sweeper.fitted_models[0][0]
        assert model is not None
        assert hasattr(model, 'A_v')
    
    def test_plot_runs(self, data_2d):
        """Test that plot method runs without error."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2, 3],
            param2_values=[5, 8],
        )
        sweeper.sweep()
        
        fig, axes = sweeper.plot()
        assert fig is not None
        assert len(axes) >= 1
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_error_before_sweep(self, data_2d):
        """Test that accessing results before sweep raises error."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2],
            param2_values=[5],
        )
        
        with pytest.raises(RuntimeError):
            _ = sweeper.aics


# =============================================================================
# Test PyKoopmanSweeper
# =============================================================================

class TestPyKoopmanSweeper:
    
    def test_sweeper_creation(self, data_2d):
        """Test PyKoopman sweeper can be created."""
        sweeper = PyKoopmanSweeper(
            data=data_2d,
            param1_name="observables.n_delays",
            param1_values=[2, 3],
            param2_name="regressor.svd_rank",
            param2_values=[5, 10],
        )
        assert sweeper._param1_component == "observables"
        assert sweeper._param1_attr == "n_delays"
        assert sweeper._param2_component == "regressor"
        assert sweeper._param2_attr == "svd_rank"
    
    def test_sweep_runs_2d(self, data_2d):
        """Test PyKoopman sweep completes on 2D data."""
        sweeper = PyKoopmanSweeper(
            data=data_2d,
            param1_name="observables.n_delays",
            param1_values=[2, 3],
            param2_name="regressor.svd_rank",
            param2_values=[5, 8],
        )
        sweeper.sweep()
        
        assert sweeper._swept
        assert sweeper.aics.shape == (2, 2)
        assert sweeper.mases.shape == (2, 2)
    
    def test_sweep_runs_3d(self, data_3d):
        """Test PyKoopman sweep on 3D data (flattened internally)."""
        sweeper = PyKoopmanSweeper(
            data=data_3d,
            param1_name="observables.n_delays",
            param1_values=[2],
            param2_name="regressor.svd_rank",
            param2_values=[5],
        )
        sweeper.sweep()
        
        assert sweeper._swept
        assert sweeper.aics.shape == (1, 1)
    
    def test_with_extra_observables(self, data_2d):
        """Test PyKoopman sweeper with extra observables."""
        from DSA.pykoopman.observables import RandomFourierFeatures
        
        sweeper = PyKoopmanSweeper(
            data=data_2d,
            param1_name="observables.n_delays",
            param1_values=[2],
            param2_name="regressor.svd_rank",
            param2_values=[10],
            extra_observables=[RandomFourierFeatures(D=20, gamma=1.0)],
        )
        sweeper.sweep()
        
        assert sweeper._swept
        assert not np.isnan(sweeper.aics[0, 0])
    
    def test_get_results(self, data_2d):
        """Test get_results for PyKoopman sweeper."""
        sweeper = PyKoopmanSweeper(
            data=data_2d,
            param1_name="observables.n_delays",
            param1_values=[2],
            param2_name="regressor.svd_rank",
            param2_values=[5],
        )
        sweeper.sweep()
        
        results = sweeper.get_results()
        assert results["param1_name"] == "observables.n_delays"
        assert results["param2_name"] == "regressor.svd_rank"
    
    def test_plot_runs(self, data_2d):
        """Test that plot method runs without error."""
        sweeper = PyKoopmanSweeper(
            data=data_2d,
            param1_name="observables.n_delays",
            param1_values=[2, 3],
            param2_name="regressor.svd_rank",
            param2_values=[5, 8],
        )
        sweeper.sweep()
        
        fig, axes = sweeper.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    
    def test_sweep_local_dmd(self, data_2d):
        """Test sweep_local_dmd convenience function."""
        sweeper = sweep_local_dmd(
            data_2d,
            n_delays_values=[2, 3],
            rank_values=[5, 8],
        )
        
        assert sweeper._swept
        assert sweeper.aics.shape == (2, 2)
    
    def test_sweep_pykoopman(self, data_2d):
        """Test sweep_pykoopman convenience function."""
        sweeper = sweep_pykoopman(
            data_2d,
            param1_name="observables.n_delays",
            param1_values=[2, 3],
            param2_name="regressor.svd_rank",
            param2_values=[5, 8],
        )
        
        assert sweeper._swept
        assert sweeper.aics.shape == (2, 2)
    
    def test_sweep_ranks_delays_backward_compat(self, data_2d):
        """Test backward-compatible sweep_ranks_delays function."""
        result = sweep_ranks_delays(
            data_2d,
            n_delays=[2, 3],
            ranks=[5, 8],
            return_residuals=False,
        )
        
        aics, mases, nnormals = result
        assert aics.shape == (2, 2)
        assert mases.shape == (2, 2)
        assert nnormals.shape == (2, 2)
    
    def test_sweep_ranks_delays_with_residuals(self, data_2d):
        """Test sweep_ranks_delays with residuals."""
        result = sweep_ranks_delays(
            data_2d,
            n_delays=[2],
            ranks=[5],
            return_residuals=True,
        )
        
        aics, mases, nnormals, residuals = result
        assert aics.shape == (1, 1)
        assert residuals.shape == (1, 1)


# =============================================================================
# Test with List Data
# =============================================================================

class TestListData:
    
    def test_local_dmd_sweeper_list_2d(self, data_list_2d):
        """Test LocalDMDSweeper with list of 2D arrays."""
        sweeper = LocalDMDSweeper(
            data=data_list_2d,
            param1_values=[2],
            param2_values=[5],
        )
        sweeper.sweep()
        
        assert sweeper._swept
        assert sweeper.aics.shape == (1, 1)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    
    def test_single_param_value(self, data_2d):
        """Test with single parameter values."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[3],
            param2_values=[10],
        )
        sweeper.sweep()
        
        assert sweeper.aics.shape == (1, 1)
        assert not np.isnan(sweeper.aics[0, 0])
    
    def test_large_n_delays(self, data_2d):
        """Test with larger n_delays (more delay embedding)."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[5, 10],
            param2_values=[20, 30],
        )
        sweeper.sweep()
        
        # Should complete without error
        assert sweeper._swept
    
    def test_train_frac_1(self, data_2d):
        """Test with train_frac=1.0 (test on train data)."""
        sweeper = LocalDMDSweeper(
            data=data_2d,
            param1_values=[2],
            param2_values=[5],
            train_frac=1.0,
        )
        sweeper.sweep()
        
        assert sweeper._swept
        assert not np.isnan(sweeper.aics[0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
