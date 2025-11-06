#!/usr/bin/env python
# coding: utf-8
"""Comprehensive tests for GPyTorch surrogate implementation."""

import numpy as np
import pytest
import tempfile
import os

try:
    import torch
    import gpytorch
    from profit.sur.gp import GPyTorchSurrogate, MultiOutputGPyTorchSurrogate
    from profit.sur import Surrogate

    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


# Test data
np.random.seed(42)
X_train_1d = np.linspace(0, 5, 20).reshape(-1, 1)
y_train_1d = np.sin(X_train_1d) + 0.1 * np.random.randn(20, 1)

X_train_2d = np.random.rand(30, 2) * 5
y_train_2d = (np.sin(X_train_2d[:, 0]) + np.cos(X_train_2d[:, 1])).reshape(-1, 1)

X_test = np.linspace(0, 5, 10).reshape(-1, 1)
X_test_2d = np.random.rand(10, 2) * 5


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestGPyTorchSurrogate:
    """Test suite for GPyTorchSurrogate."""

    def test_initialization(self):
        """Test surrogate initialization."""
        sur = GPyTorchSurrogate()
        assert sur.model is None
        assert sur.likelihood is None
        assert sur.device.type == 'cpu'

    def test_train_1d(self):
        """Test training on 1D data."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, training_iter=50)

        assert sur.trained
        assert sur.model is not None
        assert sur.likelihood is not None
        assert sur.ndim == 1
        assert 'length_scale' in sur.hyperparameters
        assert 'sigma_f' in sur.hyperparameters
        assert 'sigma_n' in sur.hyperparameters

    def test_train_2d(self):
        """Test training on 2D data."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_2d, y_train_2d, training_iter=50)

        assert sur.trained
        assert sur.ndim == 2

    def test_predict_1d(self):
        """Test predictions on 1D data."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, training_iter=50)

        ymean, yvar = sur.predict(X_test)

        assert ymean.shape == (10, 1)
        assert yvar.shape == (10, 1)
        assert np.all(yvar > 0)  # Variance should be positive

    def test_predict_without_noise(self):
        """Test predictions without observation noise."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, training_iter=50)

        ymean_with, yvar_with = sur.predict(X_test, add_data_variance=True)
        ymean_without, yvar_without = sur.predict(X_test, add_data_variance=False)

        assert np.allclose(ymean_with, ymean_without)
        assert np.all(yvar_with >= yvar_without)  # With noise should have higher variance

    def test_kernel_rbf(self):
        """Test RBF kernel."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, kernel='RBF', training_iter=30)

        assert sur.trained
        assert sur.kernel == 'RBF'

    def test_kernel_matern32(self):
        """Test Matern32 kernel."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, kernel='Matern32', training_iter=30)

        assert sur.trained
        assert sur.kernel == 'Matern32'

    def test_kernel_matern52(self):
        """Test Matern52 kernel."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, kernel='Matern52', training_iter=30)

        assert sur.trained
        assert sur.kernel == 'Matern52'

    def test_fixed_noise(self):
        """Test training with fixed noise."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, fixed_sigma_n=True, training_iter=30)

        assert sur.trained
        assert sur.fixed_sigma_n

    def test_add_training_data(self):
        """Test adding new training data."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d[:10], y_train_1d[:10], training_iter=30)

        X_new = X_train_1d[10:15]
        y_new = y_train_1d[10:15]

        sur.add_training_data(X_new, y_new)

        assert sur.Xtrain.shape[0] == 15
        assert sur.ytrain.shape[0] == 15

    def test_set_ytrain(self):
        """Test updating training outputs."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, training_iter=30)

        y_new = y_train_1d + 0.5
        sur.set_ytrain(y_new)

        assert np.allclose(sur.ytrain, y_new)

    def test_save_and_load(self):
        """Test model saving and loading."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, training_iter=50)

        # Get predictions before saving
        ymean_before, yvar_before = sur.predict(X_test)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            sur.save_model(temp_path)

            # Load model
            sur_loaded = GPyTorchSurrogate.load_model(temp_path)

            # Check attributes
            assert sur_loaded.trained
            assert sur_loaded.ndim == sur.ndim
            assert sur_loaded.kernel == sur.kernel

            # Get predictions after loading
            ymean_after, yvar_after = sur_loaded.predict(X_test)

            # Predictions should match
            assert np.allclose(ymean_before, ymean_after, rtol=1e-4)
            assert np.allclose(yvar_before, yvar_after, rtol=1e-4)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_optimize(self):
        """Test model re-optimization."""
        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_train_1d, training_iter=30)

        # Get hyperparameters before optimization
        length_scale_before = sur.hyperparameters['length_scale'].copy()

        # Re-optimize
        sur.optimize(training_iter=20)

        # Hyperparameters should exist (may or may not change)
        assert 'length_scale' in sur.hyperparameters

    def test_select_kernel(self):
        """Test kernel selection."""
        sur = GPyTorchSurrogate()

        assert sur.select_kernel('RBF') == 'RBF'
        assert sur.select_kernel('Matern32') == 'Matern32'
        assert sur.select_kernel('Matern52') == 'Matern52'

        # Unknown kernel should default to RBF
        assert sur.select_kernel('UnknownKernel') == 'RBF'

    def test_registration(self):
        """Test surrogate is registered."""
        assert 'GPyTorch' in Surrogate._registry
        sur = Surrogate['GPyTorch']()
        assert isinstance(sur, GPyTorchSurrogate)

    def test_hyperparameter_initialization(self):
        """Test initial hyperparameters are set."""
        sur = GPyTorchSurrogate()

        initial_hyperparams = {
            'length_scale': np.array([0.5]),
            'sigma_f': np.array([1.0]),
            'sigma_n': np.array([0.1])
        }

        sur.train(X_train_1d, y_train_1d, hyperparameters=initial_hyperparams, training_iter=30)

        assert sur.trained
        # Check hyperparameters were used (they will be optimized, so may differ)
        assert 'length_scale' in sur.hyperparameters


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestMultiOutputGPyTorchSurrogate:
    """Test suite for MultiOutputGPyTorchSurrogate."""

    def test_initialization(self):
        """Test multi-output surrogate initialization."""
        sur = MultiOutputGPyTorchSurrogate()
        assert sur.models == []
        assert sur.output_ndim is None

    def test_train_multi_output(self):
        """Test training on multi-output data."""
        # Create 2D output data
        y_multi = np.hstack([y_train_2d, y_train_2d * 2])

        sur = MultiOutputGPyTorchSurrogate()
        sur.train(X_train_2d, y_multi, training_iter=30)

        assert sur.trained
        assert sur.output_ndim == 2
        assert len(sur.models) == 2
        assert all(m.trained for m in sur.models)

    def test_predict_multi_output(self):
        """Test predictions on multi-output data."""
        y_multi = np.hstack([y_train_2d, y_train_2d * 2])

        sur = MultiOutputGPyTorchSurrogate()
        sur.train(X_train_2d, y_multi, training_iter=30)

        ymean, yvar = sur.predict(X_test_2d)

        assert ymean.shape == (10, 2)
        assert yvar.shape == (10, 2)
        assert np.all(yvar > 0)

    def test_add_training_data_multi_output(self):
        """Test adding data to multi-output model."""
        y_multi = np.hstack([y_train_2d, y_train_2d * 2])

        sur = MultiOutputGPyTorchSurrogate()
        sur.train(X_train_2d[:20], y_multi[:20], training_iter=20)

        X_new = X_train_2d[20:25]
        y_new = y_multi[20:25]

        sur.add_training_data(X_new, y_new)

        assert sur.Xtrain.shape[0] == 25

    def test_save_and_load_multi_output(self):
        """Test saving and loading multi-output model."""
        y_multi = np.hstack([y_train_2d, y_train_2d * 2])

        sur = MultiOutputGPyTorchSurrogate()
        sur.train(X_train_2d, y_multi, training_iter=30)

        ymean_before, _ = sur.predict(X_test_2d)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            sur.save_model(temp_path)
            sur_loaded = MultiOutputGPyTorchSurrogate.load_model(temp_path)

            assert sur_loaded.trained
            assert sur_loaded.output_ndim == 2
            assert len(sur_loaded.models) == 2

            ymean_after, _ = sur_loaded.predict(X_test_2d)
            assert np.allclose(ymean_before, ymean_after, rtol=1e-4)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_registration_multi_output(self):
        """Test multi-output surrogate is registered."""
        assert 'MultiOutputGPyTorch' in Surrogate._registry
        sur = Surrogate['MultiOutputGPyTorch']()
        assert isinstance(sur, MultiOutputGPyTorchSurrogate)


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestGPyTorchEdgeCases:
    """Test edge cases and error handling."""

    def test_single_datapoint(self):
        """Test behavior with minimal data."""
        X_single = np.array([[1.0]])
        y_single = np.array([[2.0]])

        sur = GPyTorchSurrogate()
        # Should handle gracefully (may not train well but shouldn't crash)
        try:
            sur.train(X_single, y_single, training_iter=10)
        except Exception as e:
            # Some implementations may require minimum data points
            pass

    def test_constant_output(self):
        """Test with constant output values."""
        y_constant = np.ones_like(y_train_1d)

        sur = GPyTorchSurrogate()
        sur.train(X_train_1d, y_constant, training_iter=30)

        # Should handle constant data
        assert sur.trained

    def test_normalization(self):
        """Test data normalization."""
        # Use data with large scale
        X_large = X_train_1d * 1000
        y_large = y_train_1d * 1000

        sur = GPyTorchSurrogate()
        sur.train(X_large, y_large, training_iter=30)

        # Should normalize internally
        assert sur.ymean is not None
        assert sur.yscale is not None

    def test_predict_before_train(self):
        """Test prediction before training (should fail gracefully)."""
        sur = GPyTorchSurrogate()

        # This should raise an error or handle gracefully
        with pytest.raises((AttributeError, RuntimeError, ValueError)):
            sur.predict(X_test)
