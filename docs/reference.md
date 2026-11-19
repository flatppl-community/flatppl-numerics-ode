# ODE reference

Library version **0.1** · [Package overview](../README.md)

General-purpose solvers take a deterministic RHS `f(t, y, p)`, a real vector `initial`, a
strictly monotone real vector `times` of at least two entries, and a value `p`
passed to `f` unchanged. The RHS returns a real vector matching the state. A
solver returns one vector per time, including `initial` at `times[1]`. It uses
each consecutive time interval as a step, including unequal or negative steps.
The basic `*_step(f, t, y, h, p)` functions take the same RHS and parameters, a state vector,
a start time, and a step size, and returns the next state vector.

| Solver | Single step | Order |
|---|---|---:|
| [`euler`](#euler) | [`euler_step`](#euler) | 1 |
| [`heun`](#heun) | [`heun_step`](#heun) | 2 |
| [`rk4`](#rk4) | [`rk4_step`](#rk4) | 4 |
| [`vern7`](#vern7) | [`vern7_step`](#vern7) | 7 |
| [`rosenbrock23`](#rosenbrock23) | [`rosenbrock23_step`](#rosenbrock23) | 2 |
| [`sdirk2`](#sdirk2) | [`sdirk2_step`](#sdirk2) | 2 |
| [`rodas4p2`](#rodas4p2) | [`rodas4p2_step`](#rodas4p2) | 4 |
| [`velocity_verlet`](#velocity-verlet) | [`velocity_verlet_step`](#velocity-verlet) | 2 |
| [`affine`](#exact-affine-flow) | [`affine_step`](#exact-affine-flow) | Exact flow |
| [`sdirk2_residual`](#sdirk2_residual) | — | — |

These methods take fixed grids. Vern7 and Rodas4P2 offer opt-in
[embedded error estimates](#embedded-error-estimates). Refine the grid to
assess integration accuracy.
Nominal orders need not hold on a coarse grid, near floating-point limits, or
in stiff order-reduction regimes. The stiff methods target modest vector sizes.
Finite differences use one extra RHS evaluation per state component. Dense
linear solves scale cubically in the state dimension.

## Euler

`euler(f, initial, times, p)` and `euler_step(f, t, y, h, p)` use forward Euler:

$$y_{n+1} = y_n + h f(t_n,y_n,p).$$

## Heun

`heun(f, initial, times, p)` and `heun_step(f, t, y, h, p)` use the explicit
trapezoidal rule, with a forward prediction followed by correction:

$$k_1=f(t,y,p),\quad k_2=f(t+h,y+hk_1,p),\quad y_{n+1}=y+\tfrac{h}{2}(k_1+k_2).$$

## RK4

`rk4(f, initial, times, p)` and `rk4_step(f, t, y, h, p)` use the classical
four-stage, fourth-order Runge–Kutta rule. Its stage times are
$t$, $t+h/2$, $t+h/2$, $t+h$ and weights are $1/6$, $1/3$, $1/3$, $1/6$.

## Vern7

`vern7(f, initial, times, p)` and `vern7_step(f, t, y, h, p)` use the principal
seventh-order Verner formula with nine RHS evaluations per step. See the
[principal and embedded coefficients](https://github.com/SciML/OrdinaryDiffEq.jl/blob/ee14b61c0e78cc6cbb732989b14e765df20b105f/lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl).

## Rosenbrock23

`rosenbrock23(f, initial, times, p)` and
`rosenbrock23_step(f, t, y, h, p)` use the second-order fixed-step part of
[Rosenbrock23](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl).
With $g=1-1/\sqrt{2}$, $J=\partial f/\partial y$, $D=\partial f/\partial t$,
and $A=I-hgJ$, the stages solve

$$Ak_1=f(t,y,p)+hgD,\qquad Ak_2=f(t+h/2,y+hk_1/2,p)-k_1,$$

and return $y+h(k_1+k_2)$. The state Jacobian and time derivative are computed
with forward differences. The embedded estimator and adaptive controller are
not used. L-stability does not guarantee integration accuracy on a coarse grid.

## SDIRK2

`sdirk2(f, initial, times, p)` and `sdirk2_step(f, t, y, h, p)` use a two-stage,
stiffly accurate L-stable diagonally implicit Runge–Kutta method. For
$g=1-1/\sqrt{2}$ the equations are

$$Y_1=y+hg f(t+hg,Y_1,p),$$
$$Y_2=y+h(1-g)f(t+hg,Y_1,p)+hg f(t+h,Y_2,p),\qquad y_{n+1}=Y_2.$$

Each stage runs eight Newton iterations from the stage's base state and
recomputes the Jacobian at each iterate. Use the stage residuals below to
check convergence.

## `sdirk2_residual`

`sdirk2_residual(f, t, y, h, p, result)` returns a record with vector fields
`first` and `second`, the left-minus-right residuals of the SDIRK2 stage
equations above. It recomputes the first stage with the same Newton budget and
uses `result` as the second stage. Compare each component with the scale and
accuracy required by the problem. Small residuals do not prove the time grid is
accurate; repeat on a finer grid.

## Analytic derivatives

For a state of length $d$, supply `jac(t, y, p)` as a real $d\times d$ matrix
with entry $J_{ij}=\partial f_i/\partial y_j$.
For Rosenbrock23 and Rodas4P2, supply `dfdt(t, y, p)` as a vector of length $d$
containing $\partial f/\partial t$ at fixed state and parameters. For autonomous
systems, use `dfdt(t, y, p) = 0.0 .* y`.

| Trajectory | Single step |
|---|---|
| `rosenbrock23_analytic(f, jac, dfdt, initial, times, p)` | `rosenbrock23_analytic_step(f, jac, dfdt, t, y, h, p)` |
| `sdirk2_analytic(f, jac, initial, times, p)` | `sdirk2_analytic_step(f, jac, t, y, h, p)` |

Rosenbrock23 evaluates both derivatives at the start of each step. SDIRK2
evaluates `jac` at each Newton iterate. All functions receive the same `p`.
Use `sdirk2_analytic_residual(f, jac, t, y, h, p, result)` to check the
residuals of an analytic SDIRK2 step.

```flatppl
f(t, y, k) = [-k * y[1] + t, k * y[1] - y[2]]
jac(t, y, k) = rowstack([[-k, 0.0], [k, -1.0]])
dfdt(t, y, k) = [1.0, 0.0]
trajectory = ode.rosenbrock23_analytic(f, jac, dfdt, [1.0, 0.0], [0.0, 0.1, 0.2], 2.0)
```

## Finite differences and limits

The default stiff solvers approximate the Jacobian using
$d_j=0.0001\max(1,|y_j|)$ and
$J_{:,j}=[f(t,y+d_j e_j,p)-f(t,y,p)]/d_j$. Rosenbrock23 approximates the time
derivative analogously using $d_t=0.0001\max(1,|t|)$. These defaults trade
roundoff and truncation error and do not guarantee accuracy at every scale or
precision. Check sensitivity to grid and floating-point precision. The RHS must
be defined at intermediate stage points and at any finite-difference
perturbations. Singular matrices follow the engine's
`linsolve` error behavior. No state or gradient is clamped.

## Rodas4P2

`rodas4p2(f, jac, dfdt, initial, times, p)` and
`rodas4p2_step(f, jac, dfdt, t, y, h, p)` use the fourth-order
L-stable [Rodas4P2 method](https://docs.sciml.ai/OrdinaryDiffEq/dev/massmatrixdae/Rosenbrock/).
The six stages share $I-h\gamma J$, with $\gamma=1/4$ and $J=\partial f/\partial y$
evaluated at the start of the step. Supply the [analytic derivative functions](#analytic-derivatives).
See the [coefficient table](https://github.com/SciML/OrdinaryDiffEq.jl/blob/ee14b61c0e78cc6cbb732989b14e765df20b105f/lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl).

## Embedded error estimates

| Trajectory | Single step |
|---|---|
| `vern7_with_error(f, initial, times, p)` | `vern7_step_with_error(f, t, y, h, p)` |
| `rodas4p2_with_error(f, jac, dfdt, initial, times, p)` | `rodas4p2_step_with_error(f, jac, dfdt, t, y, h, p)` |

Each returns `record(state, error)`. The state uses the principal formula.
The signed error vector is the principal result minus the embedded result
from the same starting state. A trajectory has one row per grid time.
Its first state is the supplied initial vector and its first error is zero.
For later rows, the error belongs to the interval ending at that row.

Rodas4P2 returns the sixth stage increment, with no extra RHS evaluations or
linear solves. Vern7 evaluates a tenth RHS stage for its estimate. The
estimate measures a local state difference. To check accuracy across the
trajectory or in a fitted model, refine the grid and compare the states,
log density, and parameter gradients.

## Exact affine flow

For $y'=Ay+b$ with constant $A\in\mathbb R^{d\times d}$ and $b\in\mathbb R^d$,
`affine_step(A, b, y, h)` computes the first $d$ components of

```math
\exp\!\left(h\begin{bmatrix}A & b \\ 0 & 0\end{bmatrix}\right)
\begin{bmatrix}y \\ 1\end{bmatrix}.
```

This formula also applies to singular $A$. `affine(A, b, initial, h, steps)`
computes one transition for the uniform grid and reuses it across steps. Row
$i$ gives the state after $(i-1)h$, including the initial state in row one.
The finite step size $h$ may be negative or zero. The step count is a
nonnegative integer fixed before compilation.

```flatppl
A = rowstack([[-2.0, 0.0], [2.0, 0.0]])
states = ode.affine(A, [0.0, 0.1], [1.0, 0.0], 0.25, 8)
```

## Velocity Verlet

For $q'=v$, $v'=a(t,q,p)$, `velocity_verlet_step(acceleration, t, q, v, h, p)` computes

$$v_{1/2}=v+\tfrac h2 a(t,q,p),\qquad q_1=q+h v_{1/2},
\qquad v_1=v_{1/2}+\tfrac h2 a(t+h,q_1,p).$$

The position $q$, velocity $v$, and acceleration $a$ are real vectors of the
same length. The step returns `position` and `velocity` fields.
`velocity_verlet(acceleration, q0, v0, times, p)` returns the position and velocity
trajectories on a strictly monotone grid of at least two times. Each field
includes its initial vector. The trajectory reuses acceleration between
steps. The method is second order and symplectic for separable Hamiltonian
dynamics.

## Examples and validation

The [decay example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-numerics-ode%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)
infers a decay rate from noisy measurements. Its library import is pinned to a
specific commit. Run the JavaScript checks with Node 24 or newer.

```sh
export FLATPPL_JS_DIR=/path/to/flatppl-js
for test in check orders affine rodas4p2 verlet vern7-error rodas4p2-error error-grid; do
    node "tests/$test.cjs" || exit
done
```
