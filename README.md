# flatppl-numerics-ode

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-numerics-ode%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Fixed-grid ordinary differential equation solvers for FlatPPL. Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
ode = load_module("main.flatppl")
rate(t, y, p) = -p .* y
trajectory = ode.rk4(rate, [1.0], [0.0, 0.1, 0.2], 2.0)
```

Solvers integrate $dy/dt=f(t,y,p)$. For a state of length $d$ and a time grid of length $N$, the result has shape $N\times d$, including the initial state. The RHS returns a vector of length $d$. The time grid is strictly monotone; step sizes do not adapt.

The stiff solvers accept [analytic derivative functions](docs/reference.md#analytic-derivatives). Rodas4P2 gives fourth-order stiff integration. The [affine solver](docs/reference.md#exact-affine-flow) propagates $y'=Ay+b$ by matrix exponential. [Velocity Verlet](docs/reference.md#velocity-verlet) handles separable mechanical systems.

Vern7 and Rodas4P2 offer [embedded error estimates](docs/reference.md#embedded-error-estimates) for checking a fixed grid.

The [decay example](examples/playground.flatppl) infers a rate from noisy measurements. The
[grid-refinement examples](docs/grid-refinement.md) compare log densities and gradients
for decay and coupled stiff inference.

[Third-party notices](THIRD-PARTY-NOTICES.md)
