# Check ODE grids for inference

Keep observation times and values fixed. Halve the integration step twice.
Compare the change in unnormalized log posterior density and its parameter
gradient. Use the embedded errors to locate intervals that need attention.

- [Exponential decay](../examples/refine-decay.flatppl) uses $y'=-ky$,
  $y(0)=1$, and $\log k\sim\mathcal N(0,0.5^2)$. The observation times
  are $0$, $0.5$, and $1$. The Vern7 step sizes are $0.5$, $0.25$, and $0.125$.
- [Coupled stiff decay](../examples/refine-stiff.flatppl) uses $x'=-x$ and
  $y'=k(x-y)$ from $(x_0,y_0)=(1,0)$. The model infers
  $\theta\sim\mathcal N(0,1)$ with $`k=64+32\,\operatorname{invlogit}(\theta)`$.
  Synthetic observations measure $y$ at $0.025$, $0.05$, $0.1$, and $0.2$.
  The Rodas4P2 step sizes are $0.0125$, $0.00625$, and $0.003125$.

Each example exposes `Pi_post` on the fine grid and `Pi_exact` under the
closed-form solution. The exact observation means are $e^{-kt}$ for decay
and $k(e^{-t}-e^{-kt})/(k-1)$ for the coupled model. The compiled outputs
are the coarse, medium, fine, and exact log densities, followed by the
three signed embedded error arrays.

## Run the comparison

Run the checker in an environment with JAX and Enzyme-JAX, such as the
FlatPPL testsuite `stablehlo` environment.

```sh
python examples/check-refinement.py --flatppl /path/to/flatppl
```

The checker locates the region within 4.5 log units of the exact peak on a
search grid over $[-5,5]$. It evaluates 33 equally spaced points spanning
that region. It differentiates the compiled log densities and checks the
exact model against independent Gaussian likelihood and gradient formulas.
Use `--dtype f32` to repeat in single precision.

The example budgets are $10^{-3}$ in log density and $10^{-2}$ in gradient,
using the maximum absolute error across the probes. Choose budgets and
probe regions for the model you fit.

## Results

These f64 maxima cover 33 probes in $\log k\in[-0.292,0.332]$ for decay and
$\theta\in[-1.390,2.069]$ for the stiff model.

| Model | Step size | Embedded state error | Log-density error | Gradient error | Meets budget |
|---|---:|---:|---:|---:|---|
| Decay | 0.5 | 6.30e-7 | 4.96e-7 | 3.70e-6 | Yes |
| Decay | 0.25 | 7.98e-9 | 6.82e-9 | 6.90e-8 | Yes |
| Decay | 0.125 | 7.30e-11 | 5.44e-11 | 5.60e-10 | Yes |
| Stiff | 0.0125 | 1.98e-3 | 5.63e-2 | 3.31e-2 | No |
| Stiff | 0.00625 | 2.24e-4 | 3.34e-3 | 1.97e-3 | No |
| Stiff | 0.003125 | 1.94e-5 | 2.05e-4 | 1.21e-4 | Yes |

The stiff model's largest embedded error lies in the first interval on
each grid. Its log density first meets the example budget at step size
0.003125. The coarse decay grid meets both budgets. f32 gives the same
budget decisions in this comparison.
