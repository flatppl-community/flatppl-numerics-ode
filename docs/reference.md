# ODE reference

Library version **0.1** · [Package overview](../README.md)

Every solver takes a deterministic RHS `f(t, y, p)`, a real vector `initial`, a
strictly monotone real vector `times` of at least two entries, and a value `p`
passed to `f` unchanged. The RHS returns a real vector matching the state. A
solver returns one vector per time, including `initial` at `times[1]`. It uses
each consecutive time interval as a step, including unequal or negative steps.
Each `*_step(f, t, y, h, p)` takes the same RHS and parameters, a state vector,
a start time, and a step size, and returns the next state vector.

| Solver | Single step | Intended role | Order |
|---|---|---|---:|
| [`euler`](#euler) | [`euler_step`](#euler) | Quick | 1 |
| [`heun`](#heun) | [`heun_step`](#heun) | Quick with correction | 2 |
| [`rk4`](#rk4) | [`rk4_step`](#rk4) | General-purpose smooth | 4 |
| [`vern7`](#vern7) | [`vern7_step`](#vern7) | Precise smooth | 7 |
| [`rosenbrock23`](#rosenbrock23) | [`rosenbrock23_step`](#rosenbrock23) | Quick stiff | 2 |
| [`sdirk2`](#sdirk2) | [`sdirk2_step`](#sdirk2) | General-purpose stiff | 2 |
| [`sdirk2_residual`](#sdirk2_residual) | — | Check implicit stage residuals | — |

These are fixed-grid methods, with no adaptive step selection, error estimate,
events, or interpolation. Refine the grid to assess integration accuracy.
Nominal orders need not hold on a coarse grid, near floating-point limits, or
in stiff order-reduction regimes. The role labels describe use cases, not
performance guarantees. The stiff methods target modest vector sizes: their
dense Jacobians cost one RHS evaluation per component and their dense solves
scale cubically in state dimension.

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
seventh-order Verner formula with nine RHS evaluations per step. The
[coefficients](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl)
are those of SciML's `Vern7` principal formula. The embedded estimator and
adaptive controller are not used.

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

Each stage runs eight Newton iterations from the stage's base state, with a
recomputed finite-difference Jacobian. A fixed budget does not guarantee
convergence or choose among multiple roots; inspect the stage residuals below.

## `sdirk2_residual`

`sdirk2_residual(f, t, y, h, p, result)` returns a record with vector fields
`first` and `second`, the left-minus-right residuals of the SDIRK2 stage
equations above. It recomputes the first stage with the same Newton budget and
uses `result` as the second stage. Compare each component with the scale and
accuracy required by the problem. Small residuals do not prove the time grid is
accurate; repeat on a finer grid.

## Finite differences and limits

For state component $j$, both stiff methods approximate the Jacobian using
$d_j=0.0001\max(1,|y_j|)$ and
$J_{:,j}=[f(t,y+d_j e_j,p)-f(t,y,p)]/d_j$. Rosenbrock23 approximates the time
derivative analogously using $d_t=0.0001\max(1,|t|)$. These defaults trade
roundoff and truncation error and do not guarantee accuracy at every scale or
precision. Check sensitivity to grid and floating-point precision. The RHS must
be defined at perturbed and intermediate stage points; restricted-domain
systems may need a reformulation. Singular matrices follow the engine's
`linsolve` error behavior. No state or gradient is clamped.

## Examples and validation

The result contains one state per input time, including the initial state.
Choose the grid: these solvers do not adapt the step size. See the
[reference](reference.md) for method choices, stiff-solver convergence,
and numerical limits.

The [decay example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-numerics-ode%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)
infers a decay rate from noisy measurements. Its library import is pinned to a
specific commit. To run the JavaScript checks with Node 24 or newer:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js node tests/check.cjs
FLATPPL_JS_DIR=/path/to/flatppl-js node tests/orders.cjs
```
