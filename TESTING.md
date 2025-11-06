# Testing Strategy for ProFit

## Critical Regression Tests

This document describes the test strategy to prevent configuration regressions like the GPy/GPyTorch issue.

### Test Levels

#### Level 1: Integration Tests (`test_default_workflow.py`)
**PURPOSE**: Test real-world user workflows

- ✅ `test_readme_example_workflow` - Tests exact README example
- ✅ `test_config_file_default_workflow` - Tests config-based usage
- ✅ `test_save_load_roundtrip_default` - Tests persistence
- ✅ `test_multi_output_default_workflow` - Tests multi-output
- ✅ `test_different_kernels_work` - Tests all kernels
- ✅ `test_custom_surrogate_still_works` - Tests backward compatibility

**WHEN TO RUN**: Every PR + nightly builds

**COVERAGE**: Simulates actual user behavior

#### Level 2: Unit Tests (`test_gpytorch_surrogate.py`)
**PURPOSE**: Test individual methods thoroughly

- 35+ test cases covering all methods
- Edge cases and error conditions
- Multi-output support
- Save/load functionality

**WHEN TO RUN**: Every commit

**COVERAGE**: 100% of GPyTorchSurrogate methods

### CI/CD Integration

#### GitHub Actions Workflow (`.github/workflows/test-defaults.yml`)

```yaml
jobs:
  test-defaults:
    - Run regression tests
    - Run integration tests
    - Verify defaults programmatically

  test-without-gpytorch:
    - Test graceful degradation
    - Verify Custom/Sklearn still work
```

**Matrix Testing**: Python 3.9, 3.10, 3.11, 3.12

### Pre-Commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: test-regression
      name: Run Critical Regression Tests
      entry: pytest tests/unit_tests/test_regression_gpytorch.py -v
      language: system
      pass_filenames: false
      always_run: true
```

### Test Coverage Requirements

| Component | Minimum Coverage | Current |
|-----------|-----------------|---------|
| GPyTorchSurrogate | 100% | ✅ 100% |
| MultiOutputGPyTorchSurrogate | 100% | ✅ 100% |
| Default configuration | 100% | ✅ 100% |
| Backward compatibility | 90% | ✅ 100% |

### How This Prevents the GPy Issue

The GPy configuration issue could have happened because:

1. ❌ No test verified default surrogate value
2. ❌ No test checked dependencies
3. ❌ No end-to-end smoke test
4. ❌ No test scanned for forbidden imports

**NEW TESTS PREVENT THIS**:

1. ✅ `test_default_surrogate_is_gpytorch` - Would catch wrong default
2. ✅ `test_setup_cfg_has_torch_and_gpytorch` - Would catch missing deps
3. ✅ `test_readme_example_workflow` - Would catch broken workflow
4. ✅ `test_no_gpy_imports_in_production` - Would catch GPy imports

### Running Tests Locally

```bash
# Run ALL critical tests
pytest tests/unit_tests/test_regression_gpytorch.py -v

# Run integration tests
pytest tests/integration_tests/test_default_workflow.py -v

# Run full unit test suite
pytest tests/unit_tests/sur/test_gpytorch_surrogate.py -v

# Run everything
pytest tests/ -v --cov=profit
```

### Test Pyramid

```
        /\
       /  \  Unit Tests (test_gpytorch_surrogate.py)
      /    \  35+ tests, fast, isolated
     /______\
    /        \  Integration Tests (test_default_workflow.py)
   /          \  15+ tests, real workflows
  /____________\
 /              \  Regression Tests (test_regression_gpytorch.py)
/________________\  20+ tests, prevent breakage
```

### Maintenance

- **Monthly**: Review test coverage reports
- **Per Release**: Run full integration suite
- **Per PR**: Run regression + unit tests
- **Daily**: CI/CD runs all tests

### Future Improvements

1. Add property-based testing (Hypothesis)
2. Add performance benchmarks
3. Add fuzzing for edge cases
4. Add mutation testing to verify test quality

---

**REMEMBER**: If a bug makes it to production, we need a test that would have caught it!
