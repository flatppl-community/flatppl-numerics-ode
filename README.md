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
