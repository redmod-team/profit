#!/usr/bin/env python
# coding: utf-8
"""
CRITICAL ACCURACY & REGRESSION TESTS

Tests that GPyTorch actually performs GP regression correctly:
1. Prediction accuracy on known functions
2. Comparison with Custom and Sklearn surrogates
3. Numerical regression tests
4. Cross-validation of all surrogates
"""

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, r2_score

try:
    import torch
    import gpytorch
    from profit.sur.gp import GPyTorchSurrogate, MultiOutputGPyTorchSurrogate
    from profit.sur import Surrogate
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


# Known test function
def test_function_1d(x):
    """Simple 1D test function: sin(2*pi*x)"""
    return np.sin(2 * np.pi * x)


def test_function_2d(x):
    """2D test function: sin(x1) * cos(x2)"""
    return np.sin(x[:, 0]) * np.cos(x[:, 1])


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestGPRegressionAccuracy:
    """Test actual GP regression accuracy."""

    def test_gpytorch_fits_sine_function(self):
        """Test GPyTorch can fit a simple sine function accurately."""
        np.random.seed(123)

        # Generate training data
        X_train = np.linspace(0, 1, 30).reshape(-1, 1)
        y_train = test_function_1d(X_train) + 0.05 * np.random.randn(30, 1)

        # Train
        sur = Surrogate["GPyTorch"]()
        sur.train(X_train, y_train, training_iter=200)

        # Test on clean data
        X_test = np.linspace(0, 1, 50).reshape(-1, 1)
        y_true = test_function_1d(X_test)
        y_pred, _ = sur.predict(X_test)

        # Check accuracy
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        assert rmse < 0.15, f"RMSE too high: {rmse}"
        assert r2 > 0.95, f"R² too low: {r2}"

    def test_gpytorch_2d_accuracy(self):
        """Test GPyTorch on 2D function."""
        np.random.seed(456)

        X_train = np.random.rand(50, 2)
        y_train = test_function_2d(X_train).reshape(-1, 1) + 0.05 * np.random.randn(50, 1)

        sur = Surrogate["GPyTorch"]()
        sur.train(X_train, y_train, training_iter=200)

        X_test = np.random.rand(30, 2)
        y_true = test_function_2d(X_test).reshape(-1, 1)
        y_pred, _ = sur.predict(X_test)

        r2 = r2_score(y_true, y_pred)
        assert r2 > 0.90, f"2D regression R² too low: {r2}"

    def test_uncertainty_increases_away_from_data(self):
        """Test that prediction uncertainty increases away from training data."""
        np.random.seed(789)

        # Train on data in [0.2, 0.8]
        X_train = np.linspace(0.2, 0.8, 20).reshape(-1, 1)
        y_train = test_function_1d(X_train)

        sur = Surrogate["GPyTorch"]()
        sur.train(X_train, y_train, training_iter=100)

        # Predict inside and outside training range
        X_inside = np.array([[0.5]])  # Inside training data
        X_outside = np.array([[1.5]])  # Far outside training data

        _, var_inside = sur.predict(X_inside)
        _, var_outside = sur.predict(X_outside)

        # Variance should be higher outside training region
        assert var_outside[0, 0] > var_inside[0, 0], \
            "Uncertainty should increase away from training data"

    def test_interpolation_vs_extrapolation(self):
        """Test interpolation is more accurate than extrapolation."""
        np.random.seed(101)

        # Train on [0.3, 0.7]
        X_train = np.linspace(0.3, 0.7, 25).reshape(-1, 1)
        y_train = test_function_1d(X_train)

        sur = Surrogate["GPyTorch"]()
        sur.train(X_train, y_train, training_iter=150)

        # Test interpolation (inside range)
        X_interp = np.array([[0.5]])
        y_interp_true = test_function_1d(X_interp)
        y_interp_pred, _ = sur.predict(X_interp)
        error_interp = np.abs(y_interp_true - y_interp_pred)[0, 0]

        # Test extrapolation (outside range)
        X_extrap = np.array([[0.9]])
        y_extrap_true = test_function_1d(X_extrap)
        y_extrap_pred, _ = sur.predict(X_extrap)
        error_extrap = np.abs(y_extrap_true - y_extrap_pred)[0, 0]

        # Interpolation should be more accurate
        assert error_interp < error_extrap, \
            f"Interpolation error ({error_interp}) should be < extrapolation error ({error_extrap})"


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestSurrogateComparison:
    """Compare GPyTorch with Custom and Sklearn surrogates."""

    def test_all_surrogates_can_fit(self):
        """Test that GPyTorch, Custom, and Sklearn all work on same data."""
        np.random.seed(202)

        X_train = np.random.rand(25, 1)
        y_train = test_function_1d(X_train) + 0.05 * np.random.randn(25, 1)

        X_test = np.random.rand(15, 1)
        y_true = test_function_1d(X_test)

        surrogates_to_test = ["GPyTorch", "Custom", "Sklearn"]
        results = {}

        for sur_name in surrogates_to_test:
            sur = Surrogate[sur_name]()

            if sur_name == "GPyTorch":
                sur.train(X_train, y_train, training_iter=100)
            else:
                sur.train(X_train, y_train)

            y_pred, _ = sur.predict(X_test)
            r2 = r2_score(y_true, y_pred)
            results[sur_name] = r2

            # All should achieve reasonable fit
            assert r2 > 0.80, f"{sur_name} R² too low: {r2}"

        # GPyTorch should be competitive
        assert results["GPyTorch"] > 0.85, \
            f"GPyTorch R² ({results['GPyTorch']}) should be competitive"

    def test_gpytorch_matches_custom_on_simple_problem(self):
        """Test GPyTorch gives similar results to Custom on easy problem."""
        np.random.seed(303)

        # Simple linear-ish function
        X_train = np.linspace(0, 1, 20).reshape(-1, 1)
        y_train = 2 * X_train + 0.5 + 0.02 * np.random.randn(20, 1)

        X_test = np.array([[0.3], [0.5], [0.7]])

        # Train both
        sur_gpytorch = Surrogate["GPyTorch"]()
        sur_custom = Surrogate["Custom"]()

        sur_gpytorch.train(X_train, y_train, training_iter=150)
        sur_custom.train(X_train, y_train)

        y_pred_gpytorch, _ = sur_gpytorch.predict(X_test)
        y_pred_custom, _ = sur_custom.predict(X_test)

        # Predictions should be similar (both are GP regression)
        diff = np.abs(y_pred_gpytorch - y_pred_custom).mean()
        assert diff < 0.2, \
            f"GPyTorch and Custom predictions differ by {diff} (should be similar)"


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestMultiOutputAccuracy:
    """Test multi-output GP regression accuracy."""

    def test_multi_output_independent_prediction(self):
        """Test multi-output GP can predict independent outputs."""
        np.random.seed(404)

        X_train = np.random.rand(40, 2)
        y1 = np.sin(2 * np.pi * X_train[:, 0]).reshape(-1, 1)
        y2 = np.cos(2 * np.pi * X_train[:, 1]).reshape(-1, 1)
        y_train = np.hstack([y1, y2]) + 0.05 * np.random.randn(40, 2)

        sur = Surrogate["MultiOutputGPyTorch"]()
        sur.train(X_train, y_train, training_iter=100)

        X_test = np.random.rand(20, 2)
        y_true_1 = np.sin(2 * np.pi * X_test[:, 0])
        y_true_2 = np.cos(2 * np.pi * X_test[:, 1])

        y_pred, _ = sur.predict(X_test)

        # Check each output
        r2_1 = r2_score(y_true_1, y_pred[:, 0])
        r2_2 = r2_score(y_true_2, y_pred[:, 1])

        assert r2_1 > 0.85, f"Output 1 R² too low: {r2_1}"
        assert r2_2 > 0.85, f"Output 2 R² too low: {r2_2}"


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestSerializationPreservesAccuracy:
    """Test that save/load preserves prediction accuracy."""

    def test_predictions_identical_after_save_load(self):
        """Test save/load gives IDENTICAL predictions (numerical regression)."""
        np.random.seed(505)

        X_train = np.random.rand(30, 1)
        y_train = test_function_1d(X_train) + 0.05 * np.random.randn(30, 1)

        sur = Surrogate["GPyTorch"]()
        sur.train(X_train, y_train, training_iter=150)

        X_test = np.random.rand(10, 1)
        y_pred_before, var_before = sur.predict(X_test)

        # Save and load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_file = f.name

        try:
            sur.save_model(model_file)

            from profit.sur.gp import GPyTorchSurrogate
            sur_loaded = GPyTorchSurrogate.load_model(model_file)

            y_pred_after, var_after = sur_loaded.predict(X_test)

            # Should be IDENTICAL (not just close)
            np.testing.assert_allclose(y_pred_before, y_pred_after, rtol=1e-5, atol=1e-6,
                err_msg="Predictions changed after save/load!")
            np.testing.assert_allclose(var_before, var_after, rtol=1e-5, atol=1e-6,
                err_msg="Variance changed after save/load!")

        finally:
            import os
            if os.path.exists(model_file):
                os.remove(model_file)


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestKernelEffects:
    """Test that different kernels give different (but valid) results."""

    def test_different_kernels_all_work(self):
        """Test RBF, Matern32, Matern52 all produce valid fits."""
        np.random.seed(606)

        X_train = np.random.rand(25, 1)
        y_train = test_function_1d(X_train) + 0.05 * np.random.randn(25, 1)

        X_test = np.random.rand(15, 1)
        y_true = test_function_1d(X_test)

        kernels = ["RBF", "Matern32", "Matern52"]

        for kernel_name in kernels:
            sur = Surrogate["GPyTorch"]()
            sur.train(X_train, y_train, kernel=kernel_name, training_iter=150)

            y_pred, _ = sur.predict(X_test)
            r2 = r2_score(y_true, y_pred)

            # All kernels should work reasonably well
            assert r2 > 0.85, f"{kernel_name} kernel R² too low: {r2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
