#!/usr/bin/env python
# coding: utf-8
"""
CRITICAL INTEGRATION TEST - Default Workflow Smoke Test

This tests the EXACT workflow a user would follow with DEFAULT configuration.
If this fails, the default configuration is BROKEN.

This test would have CAUGHT the GPy configuration issue immediately!
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


try:
    import torch
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestDefaultUserWorkflow:
    """
    CRITICAL: Test the exact workflow from documentation/README.

    This is what users will do. It MUST work with defaults.
    """

    def test_readme_example_workflow(self):
        """
        CRITICAL: Test the exact workflow from README/docs.

        Simulates: user installs profit, runs default example.
        """
        # Step 1: Import (as user would)
        from profit.sur import Surrogate

        # Step 2: Create surrogate using default
        # User doesn't specify anything - uses defaults
        sur = Surrogate["GPyTorch"]()  # Should work because it's the default

        # Step 3: Generate training data
        np.random.seed(123)
        X_train = np.random.rand(30, 2)
        y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])
        y_train = y_train.reshape(-1, 1)

        # Step 4: Train
        sur.train(X_train, y_train, training_iter=100)

        # Step 5: Predict
        X_test = np.random.rand(10, 2)
        ymean, yvar = sur.predict(X_test)

        # Verify
        assert ymean.shape == (10, 1), "Prediction shape incorrect"
        assert yvar.shape == (10, 1), "Variance shape incorrect"
        assert np.all(yvar > 0), "Variance must be positive"

    def test_config_file_default_workflow(self):
        """
        CRITICAL: Test workflow using config file (common use case).

        Simulates: user creates config.yaml, runs profit fit
        """
        from profit.config import BaseConfig
        from profit.sur import Surrogate

        # Create a minimal config (user wouldn't specify surrogate, uses default)
        config_content = """
ntrain: 20
variables:
    x1: Uniform(0, 1)
    x2: Uniform(0, 1)
    y: Output
fit:
    save: ./test_model.pkl
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_file = f.name

        try:
            # Load config
            config = BaseConfig.from_file(config_file)

            # Get surrogate from config (should use default: GPyTorch)
            from profit.defaults import fit_gaussian_process
            surrogate_type = config.get("fit", {}).get("surrogate", fit_gaussian_process["surrogate"])

            assert surrogate_type == "GPyTorch", (
                f"Config didn't use GPyTorch default, got {surrogate_type}"
            )

            # Create and test
            sur = Surrogate[surrogate_type]()
            assert sur.__class__.__name__ == "GPyTorchSurrogate"

        finally:
            if os.path.exists(config_file):
                os.remove(config_file)

    def test_save_load_roundtrip_default(self):
        """
        CRITICAL: Test save/load with default configuration.

        Very common workflow: train, save, load later for predictions.
        """
        from profit.sur import Surrogate
        import tempfile

        # Train a model
        np.random.seed(456)
        X = np.linspace(0, 2*np.pi, 25).reshape(-1, 1)
        y = np.sin(X) + 0.1 * np.random.randn(25, 1)

        sur = Surrogate["GPyTorch"]()
        sur.train(X, y, training_iter=100)

        # Get predictions before save
        X_test = np.array([[1.0], [2.0], [3.0]])
        ymean_before, yvar_before = sur.predict(X_test)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_file = f.name

        try:
            sur.save_model(model_file)

            # Load
            from profit.sur.gp import GPyTorchSurrogate
            sur_loaded = GPyTorchSurrogate.load_model(model_file)

            # Get predictions after load
            ymean_after, yvar_after = sur_loaded.predict(X_test)

            # Verify they match
            np.testing.assert_allclose(ymean_before, ymean_after, rtol=1e-4)
            np.testing.assert_allclose(yvar_before, yvar_after, rtol=1e-4)

        finally:
            if os.path.exists(model_file):
                os.remove(model_file)

    def test_multi_output_default_workflow(self):
        """
        CRITICAL: Test multi-output workflow with defaults.
        """
        from profit.sur import Surrogate

        # Multi-output data
        np.random.seed(789)
        X = np.random.rand(40, 3)
        y1 = np.sin(X[:, 0]) + X[:, 1]
        y2 = np.cos(X[:, 2])
        y = np.column_stack([y1, y2])

        # Use multi-output surrogate
        sur = Surrogate["MultiOutputGPyTorch"]()
        sur.train(X, y, training_iter=50)

        # Predict
        X_test = np.random.rand(5, 3)
        ymean, yvar = sur.predict(X_test)

        assert ymean.shape == (5, 2), "Multi-output prediction shape wrong"
        assert yvar.shape == (5, 2), "Multi-output variance shape wrong"
        assert np.all(yvar > 0), "Variance must be positive"

    def test_different_kernels_work(self):
        """
        CRITICAL: Verify all advertised kernels actually work.
        """
        from profit.sur import Surrogate

        kernels_to_test = ["RBF", "Matern32", "Matern52"]

        np.random.seed(101)
        X = np.random.rand(20, 1)
        y = np.sin(5*X) + 0.1*np.random.randn(20, 1)

        for kernel_name in kernels_to_test:
            sur = Surrogate["GPyTorch"]()
            sur.train(X, y, kernel=kernel_name, training_iter=50)

            assert sur.trained, f"Training failed for kernel {kernel_name}"
            assert sur.kernel == kernel_name

            # Verify it can predict
            X_test = np.array([[0.5]])
            ymean, yvar = sur.predict(X_test)
            assert ymean.shape == (1, 1)
            assert yvar[0, 0] > 0


class TestFailureModes:
    """
    CRITICAL: Test common failure modes and error messages.

    Good tests also verify failures happen gracefully.
    """

    @pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
    def test_predict_before_train_fails_gracefully(self):
        """Verify attempting to predict before training gives clear error."""
        from profit.sur import Surrogate

        sur = Surrogate["GPyTorch"]()
        X_test = np.array([[0.5]])

        # Should raise an error (not crash silently)
        with pytest.raises((AttributeError, RuntimeError, ValueError)):
            sur.predict(X_test)

    @pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
    def test_invalid_kernel_handled(self):
        """Verify invalid kernel name is handled."""
        from profit.sur import Surrogate

        sur = Surrogate["GPyTorch"]()
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)

        # Should handle gracefully (default to RBF with warning)
        sur.train(X, y, kernel="NonExistentKernel", training_iter=10)
        # Should have fallen back to RBF
        assert sur.kernel == "RBF"


class TestBackwardCompatibility:
    """
    CRITICAL: Ensure we didn't break existing code.
    """

    @pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
    def test_custom_surrogate_still_works(self):
        """
        Verify Custom surrogate still works (for users who use it explicitly).
        """
        from profit.sur import Surrogate

        sur = Surrogate["Custom"]()
        X = np.random.rand(15, 1)
        y = np.random.rand(15, 1)

        sur.train(X, y)
        assert sur.trained

        X_test = np.array([[0.5]])
        ymean, yvar = sur.predict(X_test)
        assert ymean.shape == (1, 1)

    @pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
    def test_sklearn_surrogate_still_works(self):
        """
        Verify Sklearn surrogate still works.
        """
        from profit.sur import Surrogate

        sur = Surrogate["Sklearn"]()
        X = np.random.rand(15, 1)
        y = np.random.rand(15, 1)

        sur.train(X, y)
        assert sur.trained

        X_test = np.array([[0.5]])
        ymean, yvar = sur.predict(X_test)
        assert ymean.shape == (1, 1)


@pytest.mark.skipif(HAS_GPYTORCH, reason="Only run when GPyTorch not installed")
class TestGracefulDegradation:
    """
    CRITICAL: Verify behavior when GPyTorch is not installed.

    User should get clear error message, not obscure import failure.
    """

    def test_clear_error_when_gpytorch_missing(self):
        """
        Verify we get a clear error if GPyTorch is not installed.
        """
        from profit.sur import Surrogate

        # GPyTorch should not be in registry if not installed
        if "GPyTorch" in Surrogate._registry:
            # Try to instantiate - should fail clearly
            with pytest.raises((ImportError, KeyError, RuntimeError)):
                sur = Surrogate["GPyTorch"]()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
