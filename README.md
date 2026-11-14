# flatppl-numerics-ode

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-numerics-ode%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Fixed-grid ordinary differential equation solvers for FlatPPL. Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
ode = load_module("main.flatppl")
rate(t, y, p) = -p .* y
trajectory = ode.rk4(rate, [1.0], [0.0, 0.1, 0.2], 2.0)
```

The right-hand side `f(t, y, p)` returns a vector and receives parameters explicitly. Results include the initial state. The caller supplies the time grid. Step sizes do not adapt.

The [decay example](examples/playground.flatppl) infers a rate from noisy measurements.

See the reference for definitions, domains, numerical limits, and validation commands.
