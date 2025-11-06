#!/usr/bin/env python
"""
Standalone script to generate GP comparison plots.
Run this to create visual comparisons of GPyTorch vs Custom vs Sklearn.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from profit.sur import Surrogate

    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install: pip install numpy matplotlib scikit-learn torch gpytorch")
    sys.exit(1)

# Output directory
PLOT_DIR = Path(__file__).parent / "test_output" / "gp_comparison_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory: {PLOT_DIR}")


# Test function
def sine_1d(x):
    """Simple 1D test function: sin(2*pi*x)"""
    return np.sin(2 * np.pi * x)


def generate_all_surrogates_comparison():
    """Generate comparison plot of all three GP surrogates"""
    print("\n1. Generating All Surrogates Comparison...")
    np.random.seed(202)

    X_train = np.random.rand(25, 1)
    y_train = sine_1d(X_train) + 0.05 * np.random.randn(25, 1)

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_true = sine_1d(X_test)

    surrogates_to_test = ["GPyTorch", "Custom", "Sklearn"]
    results = {}
    predictions = {}

    for sur_name in surrogates_to_test:
        print(f"   Training {sur_name}...", end="")
        sur = Surrogate[sur_name]()

        if sur_name == "GPyTorch":
            sur.train(X_train, y_train, training_iter=100)
        else:
            sur.train(X_train, y_train)

        y_pred, y_var = sur.predict(X_test)
        r2 = r2_score(y_true, y_pred)
        results[sur_name] = r2
        predictions[sur_name] = (y_pred, y_var)
        print(f" R²={r2:.3f}")

    # Generate comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot true function
    ax.plot(
        X_test, y_true, "k-", linewidth=2, label="True function: sin(2πx)", alpha=0.8
    )

    # Plot training data
    ax.scatter(
        X_train,
        y_train,
        c="red",
        s=50,
        marker="o",
        label="Training data",
        zorder=10,
        edgecolors="black",
    )

    # Plot each surrogate's prediction with uncertainty
    colors = {"GPyTorch": "blue", "Custom": "green", "Sklearn": "orange"}
    for sur_name in surrogates_to_test:
        y_pred, y_var = predictions[sur_name]
        std = np.sqrt(y_var.flatten())

        ax.plot(
            X_test,
            y_pred,
            color=colors[sur_name],
            linewidth=2,
            label=f"{sur_name} (R²={results[sur_name]:.3f})",
            alpha=0.8,
        )
        ax.fill_between(
            X_test.flatten(),
            y_pred.flatten() - 2 * std,
            y_pred.flatten() + 2 * std,
            color=colors[sur_name],
            alpha=0.15,
            label=f"{sur_name} ±2σ",
        )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        "GP Surrogate Comparison: GPyTorch vs Custom vs Sklearn",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plot_path = PLOT_DIR / "all_surrogates_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: {plot_path}")
    return str(plot_path)


def generate_gpytorch_vs_custom():
    """Generate direct GPyTorch vs Custom comparison"""
    print("\n2. Generating GPyTorch vs Custom Direct Comparison...")
    np.random.seed(303)

    X_train = np.linspace(0, 1, 20).reshape(-1, 1)
    y_train = 2 * X_train + 0.5 + 0.02 * np.random.randn(20, 1)

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)

    print("   Training GPyTorch...", end="")
    sur_gpytorch = Surrogate["GPyTorch"]()
    sur_gpytorch.train(X_train, y_train, training_iter=150)
    print(" Done")

    print("   Training Custom...", end="")
    sur_custom = Surrogate["Custom"]()
    sur_custom.train(X_train, y_train)
    print(" Done")

    y_pred_gpytorch, y_var_gpytorch = sur_gpytorch.predict(X_test)
    y_pred_custom, y_var_custom = sur_custom.predict(X_test)

    diff = np.abs(y_pred_gpytorch - y_pred_custom).mean()
    print(f"   Mean absolute difference: {diff:.4f}")

    # Generate plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Both predictions overlaid
    ax1.scatter(
        X_train,
        y_train,
        c="red",
        s=50,
        marker="o",
        label="Training data",
        zorder=10,
        edgecolors="black",
    )
    ax1.plot(X_test, y_pred_gpytorch, "b-", linewidth=2, label="GPyTorch", alpha=0.8)
    ax1.fill_between(
        X_test.flatten(),
        y_pred_gpytorch.flatten() - 2 * np.sqrt(y_var_gpytorch.flatten()),
        y_pred_gpytorch.flatten() + 2 * np.sqrt(y_var_gpytorch.flatten()),
        color="blue",
        alpha=0.15,
        label="GPyTorch ±2σ",
    )
    ax1.plot(X_test, y_pred_custom, "g--", linewidth=2, label="Custom", alpha=0.8)
    ax1.fill_between(
        X_test.flatten(),
        y_pred_custom.flatten() - 2 * np.sqrt(y_var_custom.flatten()),
        y_pred_custom.flatten() + 2 * np.sqrt(y_var_custom.flatten()),
        color="green",
        alpha=0.15,
        label="Custom ±2σ",
    )
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("GPyTorch vs Custom: Linear Function", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Difference
    diff_plot = np.abs(y_pred_gpytorch - y_pred_custom).flatten()
    ax2.plot(X_test, diff_plot, "r-", linewidth=2)
    ax2.axhline(y=diff, color="k", linestyle="--", label=f"Mean diff: {diff:.4f}")
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("|GPyTorch - Custom|", fontsize=12)
    ax2.set_title(
        "Absolute Difference Between Predictions", fontsize=14, fontweight="bold"
    )
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plot_path = PLOT_DIR / "gpytorch_vs_custom_direct.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: {plot_path}")
    return str(plot_path)


def generate_kernel_comparison():
    """Generate kernel comparison plot"""
    print("\n3. Generating Kernel Comparison...")
    np.random.seed(606)

    X_train = np.random.rand(25, 1)
    y_train = sine_1d(X_train) + 0.05 * np.random.randn(25, 1)

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_true = sine_1d(X_test)

    kernels = ["RBF", "Matern32", "Matern52"]
    kernel_results = {}

    for kernel_name in kernels:
        print(f"   Training with {kernel_name} kernel...", end="")
        sur = Surrogate["GPyTorch"]()
        sur.train(X_train, y_train, kernel=kernel_name, training_iter=150)

        y_pred, y_var = sur.predict(X_test)
        r2 = r2_score(y_true, y_pred)
        kernel_results[kernel_name] = (y_pred, y_var, r2)
        print(f" R²={r2:.3f}")

    # Generate plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot true function
    ax.plot(
        X_test,
        y_true,
        "k-",
        linewidth=3,
        label="True function: sin(2πx)",
        alpha=0.9,
        zorder=5,
    )

    # Plot training data
    ax.scatter(
        X_train,
        y_train,
        c="red",
        s=80,
        marker="o",
        label="Training data",
        zorder=10,
        edgecolors="black",
        linewidth=1.5,
    )

    # Plot predictions for each kernel
    colors = {"RBF": "blue", "Matern32": "green", "Matern52": "purple"}
    linestyles = {"RBF": "-", "Matern32": "--", "Matern52": "-."}

    for kernel_name in kernels:
        y_pred, y_var, r2 = kernel_results[kernel_name]
        std = np.sqrt(y_var.flatten())

        ax.plot(
            X_test,
            y_pred,
            color=colors[kernel_name],
            linewidth=2,
            linestyle=linestyles[kernel_name],
            label=f"{kernel_name} (R²={r2:.3f})",
            alpha=0.8,
        )
        ax.fill_between(
            X_test.flatten(),
            y_pred.flatten() - 2 * std,
            y_pred.flatten() + 2 * std,
            color=colors[kernel_name],
            alpha=0.12,
        )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        "GPyTorch Kernel Comparison: RBF vs Matern32 vs Matern52",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plot_path = PLOT_DIR / "kernel_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: {plot_path}")
    return str(plot_path)


def generate_gpytorch_accuracy():
    """Generate GPyTorch accuracy plot with residuals"""
    print("\n4. Generating GPyTorch Accuracy Plot...")
    np.random.seed(123)

    X_train = np.linspace(0, 1, 30).reshape(-1, 1)
    y_train = sine_1d(X_train) + 0.05 * np.random.randn(30, 1)

    print("   Training GPyTorch...", end="")
    sur = Surrogate["GPyTorch"]()
    sur.train(X_train, y_train, training_iter=200)
    print(" Done")

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_true = sine_1d(X_test)
    y_pred, y_var = sur.predict(X_test)

    from sklearn.metrics import mean_squared_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"   R²={r2:.4f}, RMSE={rmse:.4f}")

    # Generate plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Fit plot
    std = np.sqrt(y_var.flatten())
    ax1.plot(
        X_test, y_true, "k-", linewidth=2, label="True function: sin(2πx)", alpha=0.8
    )
    ax1.scatter(
        X_train,
        y_train,
        c="red",
        s=50,
        marker="o",
        label="Noisy training data",
        zorder=10,
        edgecolors="black",
    )
    ax1.plot(
        X_test,
        y_pred,
        "b-",
        linewidth=2,
        label=f"GPyTorch (R²={r2:.4f}, RMSE={rmse:.4f})",
        alpha=0.8,
    )
    ax1.fill_between(
        X_test.flatten(),
        y_pred.flatten() - 2 * std,
        y_pred.flatten() + 2 * std,
        color="blue",
        alpha=0.2,
        label="±2σ confidence",
    )
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("GPyTorch Sine Function Fit", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Residuals plot
    residuals = (y_pred - y_true).flatten()
    ax2.scatter(X_test, residuals, c="blue", s=30, alpha=0.6)
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=2)
    ax2.axhline(
        y=2 * std.mean(), color="r", linestyle=":", linewidth=1.5, label="±2σ mean"
    )
    ax2.axhline(y=-2 * std.mean(), color="r", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("Residuals (Predicted - True)", fontsize=12)
    ax2.set_title("Residuals Analysis", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plot_path = PLOT_DIR / "gpytorch_sine_accuracy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: {plot_path}")
    return str(plot_path)


if __name__ == "__main__":
    print("=" * 70)
    print("GP COMPARISON PLOT GENERATOR")
    print("=" * 70)

    plots = []
    try:
        plots.append(generate_gpytorch_accuracy())
        plots.append(generate_all_surrogates_comparison())
        plots.append(generate_gpytorch_vs_custom())
        plots.append(generate_kernel_comparison())
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"✓ Successfully generated {len(plots)} plots!")
    print("=" * 70)
    print("\nGenerated files:")
    for plot in plots:
        print(f"  - {plot}")
    print()
