# flatppl-numerics-ode

Fixed-grid ordinary differential equation solvers for FlatPPL 0.1. The RHS has
the signature `f(t, y, p)`, returns a vector, and receives parameters explicitly.

```flatppl
flatppl_compat = "0.1"
ode = load_module("main.flatppl")
rate(t, y, p) = -p .* y
trajectory = ode.rk4(rate, [1.0], [0.0, 0.1, 0.2], 2.0)
```

The result contains one state per input time, including the initial state.
Choose the grid: these solvers do not adapt the step size. See the
[reference](docs/reference.md) for method choices, stiff-solver convergence,
and numerical limits.

The [decay example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-numerics-ode%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)
infers a decay rate from noisy measurements. Its library import is pinned to a
specific commit. To run the JavaScript checks with Node 24 or newer:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js node tests/check.cjs
FLATPPL_JS_DIR=/path/to/flatppl-js node tests/orders.cjs
```
