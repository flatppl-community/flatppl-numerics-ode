# ODE reference

Each solver takes `(f, initial, times, p)` with a vector initial state, a strictly
monotone vector of at least two times, and a parameter object `p`. The function
`f(t, y, p)` returns a vector of the same length as `y`. A corresponding
`*_step(f, t, y, h, p)` returns a single next state. Every solver follows the
supplied grid, including uneven intervals. Refining the grid is necessary to
assess integration error; none estimates or controls it automatically.

| Solver | Intended role | Order | Method |
|---|---|---:|---|
| `euler` | Quick | 1 | Forward Euler |
| `heun` | Quick with correction | 2 | Explicit trapezoid |
| `rk4` | General-purpose smooth | 4 | Classical Runge–Kutta |
| `vern7` | Precise smooth | 7 | Verner Runge–Kutta |
| `rosenbrock23` | Quick stiff | 2 | Two-stage Rosenbrock |
| `sdirk2` | General-purpose stiff | 2 | Two-stage L-stable SDIRK |

The stiff methods use a dense forward-difference Jacobian, with component
increment `0.0001 * max(1, abs(y[i]))`. Rosenbrock23 also uses a forward
finite difference in time and two dense linear solves per step. SDIRK2 runs
eight Newton iterations per stage. These defaults do not guarantee convergence,
accuracy, or a solution for every stiff problem. Check each SDIRK2 step with
`sdirk2_residual(f, t, y, h, p, result)`, which returns the two stage residual
vectors; compare their magnitudes to the scales of your problem. The first
stage is recomputed using the same fixed iteration budget. Small residuals do
not by themselves prove integration accuracy. Refine the grid and inspect
sensitivity to its spacing and numerical precision.

The RHS must be defined at finite-difference probe and intermediate stage
points. Systems with domain-restricted states may need a reformulation. A
singular linear system follows the engine's existing `linsolve` error behavior.
No state or gradient is clamped. Dense Jacobians and solves target modest state
dimensions. Stiff order reduction and finite precision can limit accuracy.

The Vern7 coefficients follow the principal seventh-order formula in
[SciML OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl).
Rosenbrock23 follows its [Rosenbrock step implementation](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl), without adaptive error estimation.
