const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const source = `flatppl_compat = "0.1"
ode = load_module("../main.flatppl")
linear(t, y, p) = p .* y
nonautonomous(t, y, p) = t .- y
nonlinear(t, y, p) = -p .* (y .* y .- 1)
coupled(t, y, p) = [p * (y[2] - y[1]^2), p * (y[1]^2 - y[2])]
euler = ode.euler(linear, [1.0, 2.0], [0.0, 0.1, 0.2], -1.0)
heun = ode.heun_step(nonautonomous, 0.0, [1.0], 0.1, 0.0)
rk4 = ode.rk4_step(nonautonomous, 0.0, [1.0], 0.1, 0.0)
vern7 = ode.vern7_step(nonautonomous, 0.0, [1.0], 0.1, 0.0)
rosen = ode.rosenbrock23_step(linear, 0.0, [1.0], 0.1, -1000.0)
sdirk = ode.sdirk2_step(linear, 0.0, [1.0], 0.1, -1000.0)
nonlinear_step = ode.sdirk2_step(nonlinear, 0.0, [1.5], 0.01, 50.0)
residual = ode.sdirk2_residual(nonlinear, 0.0, [1.5], 0.01, 50.0, nonlinear_step)
coupled_step = ode.sdirk2_step(coupled, 0.0, [0.9, 0.1], 0.01, 50.0)`;
const bindings = ['euler', 'heun', 'rk4', 'vern7', 'rosen', 'sdirk', 'nonlinear_step', 'residual', 'coupled_step'];
const results = JSON.parse(execFileSync(process.execPath, [path.join(__dirname, 'evaluate.cjs')], {
  input: JSON.stringify({ source, bindings }), encoding: 'utf8', env: process.env,
}));
const scalar = value => value.data['0'];
const close = (a, b, tol) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);
close(results[0][2].data['0'], 0.81, 1e-12);
const exact = 2 * Math.exp(-0.1) + 0.1 - 1;
close(scalar(results[1]), exact, 4e-4);
close(scalar(results[2]), exact, 2e-7);
close(scalar(results[3]), exact, 1e-10);
assert.ok(Math.abs(scalar(results[4])) < 0.05);
assert.ok(Math.abs(scalar(results[5])) < 0.05);
const c = 0.2 * Math.exp(-1);
close(scalar(results[6]), (1 + c) / (1 - c), 0.02);
assert.ok(Math.max(...Object.values(results[7].first.data).map(Math.abs)) < 1e-8);
assert.ok(Math.max(...Object.values(results[7].second.data).map(Math.abs)) < 1e-8);
const upper = (Math.sqrt(5) - 1) / 2;
const lower = -(Math.sqrt(5) + 1) / 2;
const ratio = (0.9 - upper) / (0.9 - lower) * Math.exp(-0.5 * Math.sqrt(5));
close(results[8].data['0'], (upper - ratio * lower) / (1 - ratio), 0.01);
close(results[8].data['0'] + results[8].data['1'], 1, 1e-12);

// The example's posterior score is compared with the exponential solution and
// independent normal and log-normal density formulas at rate = 1.
const fs = require('node:fs');
const example = fs.readFileSync(path.join(__dirname, '../examples/playground.flatppl'), 'utf8')
  .replace(/https:\/\/raw\.githubusercontent\.com\/BJMCox\/flatppl-numerics-ode\/[^/]+\/main\.flatppl/, '../main.flatppl')
  + '\nexample_score=logdensityof(Pi_post, record(rate=1.0))';
const [score] = JSON.parse(execFileSync(process.execPath, [path.join(__dirname, 'evaluate.cjs')], {
  input: JSON.stringify({ source: example, bindings: ['example_score'] }), encoding: 'utf8', env: process.env,
}));
const observed = [1.02, 0.57, 0.38];
const times = [0, 0.5, 1];
const lognormalPrior = -Math.log(0.5) - 0.5 * Math.log(2 * Math.PI);
const expectedScore = observed.reduce((sum, y, i) =>
  sum - 0.5 * ((y - Math.exp(-times[i])) / 0.05) ** 2 - Math.log(0.05) - 0.5 * Math.log(2 * Math.PI), lognormalPrior);
close(score, expectedScore, 1e-7);
console.log('ODE checks passed');
