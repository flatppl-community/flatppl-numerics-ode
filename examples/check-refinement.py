"""Compare fixed-grid ODE log densities and gradients against exact flows."""

import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call

EXAMPLES = Path(__file__).resolve().parent
LOG_DENSITY_BUDGET = 1e-3
GRADIENT_BUDGET = 1e-2


def gaussian_target(theta, means, means_gradient, observed, sigma, prior_sigma):
    residuals = observed - means
    log_prior = -0.5 * (theta / prior_sigma) ** 2 - np.log(
        prior_sigma * np.sqrt(2 * np.pi)
    )
    log_likelihood = (
        -0.5 * (residuals / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
    ).sum(axis=1)
    gradient = (
        -theta / prior_sigma**2 + (residuals * means_gradient).sum(axis=1) / sigma**2
    )
    return log_prior + log_likelihood, gradient


def decay_exact(theta):
    time = np.array([0.0, 0.5, 1.0])
    kt = np.exp(theta[:, None]) * time
    mean = np.exp(-kt)
    return gaussian_target(
        theta, mean, -kt * mean, np.array([1.02, 0.57, 0.38]), 0.05, 0.5
    )


def stiff_exact(theta):
    time = np.array([0.025, 0.05, 0.1, 0.2])
    s = 1 / (1 + np.exp(-theta[:, None]))
    k = 64 + 32 * s
    fast = np.exp(-k * time)
    difference = np.exp(-time) - fast
    mean = k / (k - 1) * difference
    partial_k = -difference / (k - 1) ** 2 + k / (k - 1) * time * fast
    return gaussian_target(
        theta,
        mean,
        partial_k * 32 * s * (1 - s),
        np.array([0.855, 0.938, 0.919, 0.822]),
        0.015,
        1.0,
    )


def high_density_points(exact, dtype):
    search = np.linspace(-5.0, 5.0, 10001)
    score, _ = exact(search)
    region = search[score >= score.max() - 4.5]
    assert search[0] < region[0] and region[-1] < search[-1]
    return np.linspace(region[0], region[-1], 33, dtype=dtype)


def check_model(flatppl, dtype, model, exact, step_sizes):
    mlir = subprocess.check_output(
        [
            flatppl,
            "stablehlo",
            str(EXAMPLES / f"refine-{model}.flatppl"),
            "--dtype",
            dtype,
        ],
        text=True,
    ).replace("@logdensity(", "@main(")

    def evaluate(theta):
        return hlo_call(theta, source=mlir)

    def scores(theta):
        return jnp.stack(evaluate(theta)[:4])

    points = high_density_points(exact, np.float64 if dtype == "f64" else np.float32)
    args = jnp.asarray(points)
    values = tuple(map(np.asarray, jax.jit(jax.vmap(evaluate))(args)))
    gradients = np.asarray(jax.jit(jax.vmap(jax.jacfwd(scores)))(args))
    log_densities = np.column_stack(values[:4])
    if len(values) == 8:
        np.testing.assert_array_equal(values[-1], 0)
    assert np.isfinite(log_densities).all() and np.isfinite(gradients).all()

    exact_scores, exact_gradients = exact(points.astype(np.float64))
    tol = 2e-11 if dtype == "f64" else 5e-5
    np.testing.assert_allclose(log_densities[:, 3], exact_scores, rtol=0, atol=tol)
    np.testing.assert_allclose(gradients[:, 3], exact_gradients, rtol=0, atol=tol)
    score_errors = np.max(np.abs(log_densities[:, :3] - exact_scores[:, None]), axis=0)
    gradient_errors = np.max(
        np.abs(gradients[:, :3] - exact_gradients[:, None]), axis=0
    )

    grids = []
    for h, score_error, gradient_error, step_errors in zip(
        step_sizes, score_errors, gradient_errors, values[4:7], strict=True
    ):
        np.testing.assert_array_equal(step_errors[:, 0], 0)
        absolute = np.abs(step_errors[:, 1:])
        assert np.isfinite(absolute).all()
        _, interval, _ = np.unravel_index(absolute.argmax(), absolute.shape)
        grids.append(
            {
                "step_size": h,
                "max_log_density_error": float(score_error),
                "max_gradient_error": float(gradient_error),
                "max_embedded_error": float(absolute.max()),
                "largest_error_interval": [
                    float(interval * h),
                    float((interval + 1) * h),
                ],
                "within_example_budget": bool(
                    score_error <= LOG_DENSITY_BUDGET
                    and gradient_error <= GRADIENT_BUDGET
                ),
            }
        )
    return {
        "model": model,
        "dtype": dtype,
        "points": len(points),
        "parameter_range": [float(points[0]), float(points[-1])],
        "grids": grids,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flatppl", default="flatppl")
    parser.add_argument("--dtype", choices=("f32", "f64"), default="f64")
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", args.dtype == "f64")
    jax.config.update("jax_platforms", "cpu")
    checks = [
        check_model(args.flatppl, args.dtype, "decay", decay_exact, (0.5, 0.25, 0.125)),
        check_model(
            args.flatppl, args.dtype, "stiff", stiff_exact, (0.0125, 0.00625, 0.003125)
        ),
    ]
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
