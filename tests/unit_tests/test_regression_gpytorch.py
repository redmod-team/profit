#!/usr/bin/env python
# coding: utf-8
"""
CRITICAL REGRESSION TESTS - Prevent GPy/GPyTorch Configuration Issues

This test suite ensures:
1. Default surrogate is GPyTorch (not GPy or Custom)
2. GPy is NOT imported anywhere in production code
3. Dependencies are correctly configured
4. End-to-end workflow uses GPyTorch by default
5. No accidental fallbacks to old code

These tests MUST pass to prevent configuration regressions!
"""

import pytest
import sys
import importlib.util
from pathlib import Path


class TestDefaultConfiguration:
    """CRITICAL: Verify default configuration is GPyTorch."""

    def test_default_surrogate_is_gpytorch(self):
        """CRITICAL: Ensure default surrogate is GPyTorch, not GPy or Custom."""
        from profit.defaults import fit, fit_gaussian_process

        assert fit["surrogate"] == "GPyTorch", (
            f"CRITICAL FAILURE: Default surrogate is '{fit['surrogate']}' "
            f"but MUST be 'GPyTorch'! This breaks the entire migration."
        )

        assert fit_gaussian_process["surrogate"] == "GPyTorch", (
            f"CRITICAL FAILURE: fit_gaussian_process surrogate is "
            f"'{fit_gaussian_process['surrogate']}' but MUST be 'GPyTorch'!"
        )

    def test_gpytorch_is_registered(self):
        """CRITICAL: Ensure GPyTorch surrogate is properly registered."""
        from profit.sur import Surrogate

        assert "GPyTorch" in Surrogate._registry, (
            "CRITICAL FAILURE: GPyTorch not registered in Surrogate registry!"
        )

        # Verify we can instantiate it
        sur = Surrogate["GPyTorch"]()
        assert sur is not None
        assert sur.__class__.__name__ == "GPyTorchSurrogate"

    def test_multioutput_gpytorch_is_registered(self):
        """CRITICAL: Ensure MultiOutputGPyTorch is registered."""
        from profit.sur import Surrogate

        assert "MultiOutputGPyTorch" in Surrogate._registry, (
            "CRITICAL FAILURE: MultiOutputGPyTorch not registered!"
        )


class TestGPyIsGone:
    """CRITICAL: Verify GPy is completely removed from production code."""

    def test_no_gpy_imports_in_production(self):
        """CRITICAL: Ensure no production code imports GPy."""
        project_root = Path(__file__).parent.parent.parent.parent

        # Check main production directories
        production_dirs = [
            project_root / "profit",
        ]

        gpy_imports = []
        for prod_dir in production_dirs:
            if not prod_dir.exists():
                continue

            for py_file in prod_dir.rglob("*.py"):
                # Skip __pycache__ and draft folders
                if "__pycache__" in str(py_file) or "draft" in str(py_file):
                    continue

                try:
                    content = py_file.read_text()
                    if "import GPy" in content or "from GPy" in content:
                        # Exclude GPyTorch mentions
                        if "GPyTorch" not in content.split("import GPy")[0] if "import GPy" in content else "":
                            gpy_imports.append(str(py_file))
                except Exception:
                    pass

        assert len(gpy_imports) == 0, (
            f"CRITICAL FAILURE: Found GPy imports in production code:\n"
            + "\n".join(gpy_imports)
        )

    def test_gpy_surrogate_file_deleted(self):
        """CRITICAL: Ensure gpy_surrogate.py is deleted."""
        project_root = Path(__file__).parent.parent.parent.parent
        gpy_file = project_root / "profit" / "sur" / "gp" / "gpy_surrogate.py"

        assert not gpy_file.exists(), (
            f"CRITICAL FAILURE: gpy_surrogate.py still exists at {gpy_file}!"
        )

    def test_gpytorch_surrogate_exists(self):
        """CRITICAL: Ensure gpytorch_surrogate.py exists."""
        project_root = Path(__file__).parent.parent.parent.parent
        gpytorch_file = project_root / "profit" / "sur" / "gp" / "gpytorch_surrogate.py"

        assert gpytorch_file.exists(), (
            f"CRITICAL FAILURE: gpytorch_surrogate.py missing at {gpytorch_file}!"
        )


class TestDependencies:
    """CRITICAL: Verify dependencies are correctly configured."""

    def test_setup_cfg_has_torch_and_gpytorch(self):
        """CRITICAL: Ensure setup.cfg includes torch and gpytorch."""
        project_root = Path(__file__).parent.parent.parent.parent
        setup_cfg = project_root / "setup.cfg"

        if not setup_cfg.exists():
            pytest.skip("setup.cfg not found")

        content = setup_cfg.read_text()

        assert "torch" in content, (
            "CRITICAL FAILURE: 'torch' not in setup.cfg dependencies!"
        )
        assert "gpytorch" in content, (
            "CRITICAL FAILURE: 'gpytorch' not in setup.cfg dependencies!"
        )

    def test_setup_cfg_no_gpy_dependency(self):
        """CRITICAL: Ensure setup.cfg does NOT have GPy as core dependency."""
        project_root = Path(__file__).parent.parent.parent.parent
        setup_cfg = project_root / "setup.cfg"

        if not setup_cfg.exists():
            pytest.skip("setup.cfg not found")

        content = setup_cfg.read_text()

        # Check install_requires section doesn't have GPy
        in_install_requires = False
        for line in content.split('\n'):
            if '[options.extras_require]' in line:
                in_install_requires = False
            if 'install_requires' in line:
                in_install_requires = True
            if in_install_requires and 'GPy' in line and 'gpytorch' not in line.lower():
                pytest.fail(
                    f"CRITICAL FAILURE: GPy found in core dependencies: {line}"
                )


try:
    import torch
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
class TestEndToEndWorkflow:
    """CRITICAL: End-to-end smoke tests using default configuration."""

    def test_can_create_default_surrogate(self):
        """CRITICAL: Verify we can create surrogate using defaults."""
        from profit.sur import Surrogate
        from profit.defaults import fit

        # This is how users will use it - MUST work!
        surrogate_name = fit["surrogate"]
        sur = Surrogate[surrogate_name]()

        assert sur is not None
        assert sur.__class__.__name__ == "GPyTorchSurrogate"

    def test_end_to_end_training_with_defaults(self):
        """CRITICAL: Full training workflow with default configuration."""
        import numpy as np
        from profit.sur import Surrogate
        from profit.defaults import fit_gaussian_process

        # Generate simple test data
        np.random.seed(42)
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.sin(2 * np.pi * X) + 0.1 * np.random.randn(20, 1)

        # Create surrogate using EXACT default config
        surrogate_name = fit_gaussian_process["surrogate"]
        kernel = fit_gaussian_process["kernel"]

        sur = Surrogate[surrogate_name]()

        # Train with minimal iterations for speed
        sur.train(X, y, kernel=kernel, training_iter=50)

        # Verify it worked
        assert sur.trained
        assert sur.ndim == 1

        # Test prediction
        X_test = np.array([[0.5]])
        ymean, yvar = sur.predict(X_test)

        assert ymean.shape == (1, 1)
        assert yvar.shape == (1, 1)
        assert yvar[0, 0] > 0  # Variance must be positive

    def test_from_config_creates_gpytorch(self):
        """CRITICAL: Verify config-based instantiation uses GPyTorch."""
        from profit.config import BaseConfig
        from profit.sur import Surrogate
        import tempfile
        import os

        # Create a minimal config file
        config_content = """
fit:
    surrogate: GPyTorch
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_file = f.name

        try:
            config = BaseConfig.from_file(config_file)

            # Verify config was loaded correctly
            assert config["fit"]["surrogate"] == "GPyTorch"

            # Verify we can create the surrogate from config
            sur = Surrogate[config["fit"]["surrogate"]]()
            assert sur.__class__.__name__ == "GPyTorchSurrogate"

        finally:
            if os.path.exists(config_file):
                os.remove(config_file)


class TestRegressionPrevention:
    """Tests specifically designed to catch the GPy->GPyTorch regression."""

    def test_defaults_module_imports_cleanly(self):
        """Ensure defaults.py doesn't accidentally import GPy."""
        # Re-import to catch any import-time issues
        import importlib
        import profit.defaults
        importlib.reload(profit.defaults)

        from profit.defaults import fit, fit_gaussian_process

        # These MUST be GPyTorch after our migration
        assert fit["surrogate"] == "GPyTorch"
        assert fit_gaussian_process["surrogate"] == "GPyTorch"

    def test_no_gpy_in_defaults_comments(self):
        """Check that defaults.py comments don't mislead to GPy."""
        project_root = Path(__file__).parent.parent.parent.parent
        defaults_file = project_root / "profit" / "defaults.py"

        if not defaults_file.exists():
            pytest.skip("defaults.py not found")

        content = defaults_file.read_text()

        # Find the fit and fit_gaussian_process sections
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '"surrogate":' in line and 'GPy' in line:
                # Check if this is referring to GPy (not GPyTorch)
                if 'GPyTorch' not in line:
                    pytest.fail(
                        f"CRITICAL: Line {i+1} in defaults.py mentions GPy as surrogate:\n{line}"
                    )

    def test_example_files_use_gpytorch(self):
        """Ensure example files use GPyTorch, not GPy."""
        project_root = Path(__file__).parent.parent.parent.parent
        examples_dir = project_root / "examples"

        if not examples_dir.exists():
            pytest.skip("examples directory not found")

        for py_file in examples_dir.rglob("*.py"):
            content = py_file.read_text()

            # Skip if it's clearly a historical example
            if "draft" in str(py_file) or "old" in str(py_file):
                continue

            # If it imports from gpy_surrogate, that's wrong
            if "from profit.sur.gp.gpy_surrogate import" in content:
                pytest.fail(
                    f"CRITICAL: Example {py_file} still imports from gpy_surrogate!"
                )

            # If it uses GPySurrogate(), that's wrong
            if "GPySurrogate()" in content:
                pytest.fail(
                    f"CRITICAL: Example {py_file} still uses GPySurrogate()!"
                )


class TestImportSafety:
    """Ensure imports don't accidentally bring in GPy."""

    def test_profit_imports_without_gpy(self):
        """CRITICAL: Ensure we can import profit without GPy installed."""
        # This test verifies the try/except in __init__.py works
        try:
            # Temporarily hide GPy if it exists
            import sys
            gpy_module = sys.modules.get('GPy')
            if gpy_module:
                sys.modules['GPy'] = None

            # Try importing profit components
            from profit.sur import Surrogate
            from profit.sur.gp import GPyTorchSurrogate
            from profit.defaults import fit

            # Verify they work
            assert Surrogate is not None
            assert GPyTorchSurrogate is not None
            assert fit["surrogate"] == "GPyTorch"

        finally:
            # Restore GPy if it was there
            if gpy_module:
                sys.modules['GPy'] = gpy_module

    def test_gpytorch_import_is_optional(self):
        """Verify GPyTorch import is wrapped in try/except for graceful degradation."""
        project_root = Path(__file__).parent.parent.parent.parent
        gp_init = project_root / "profit" / "sur" / "gp" / "__init__.py"

        if not gp_init.exists():
            pytest.skip("__init__.py not found")

        content = gp_init.read_text()

        # Should have try/except for optional import
        assert "try:" in content, "Missing try/except for optional imports"
        assert "except ImportError:" in content, "Missing ImportError handling"
        assert "GPyTorchSurrogate" in content, "GPyTorchSurrogate not imported"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
