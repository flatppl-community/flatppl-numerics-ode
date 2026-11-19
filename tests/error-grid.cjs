const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');

const source = `flatppl_compat = "0.1"
ode = load_module("../main.flatppl")
f(t, y, k) = k .* y .+ [t, -t]
jac(t, y, k) = rowstack([[k, 0.0], [0.0, k]])
dfdt(t, y, k) = [1.0, -1.0]
initial = [0.7, 0.2]
times = [0.5, 0.25, 0.1, -0.2]
k = -1.2
vern = ode.vern7(f, initial, times, k)
rodas = ode.rodas4p2(f, jac, dfdt, initial, times, k)
vern_error = ode.vern7_with_error(f, initial, times, k)
rodas_error = ode.rodas4p2_with_error(f, jac, dfdt, initial, times, k)
vern_step = ode.vern7_step_with_error(f, times[2], vern_error.state[2], times[3] - times[2], k)
rodas_step = ode.rodas4p2_step_with_error(f, jac, dfdt, times[2], rodas_error.state[2], times[3] - times[2], k)`;
const bindings = ['vern', 'rodas', 'vern_error', 'rodas_error', 'vern_step', 'rodas_step'];
const [vern, rodas, vernError, rodasError, vernStep, rodasStep] = JSON.parse(
  execFileSync(process.execPath, [path.join(__dirname, 'evaluate.cjs')], {
    input: JSON.stringify({ source, bindings }), encoding: 'utf8', env: process.env,
  }));

const vector = value => Object.values(value.data);
for (const [reference, trajectory, step] of [[vern, vernError, vernStep], [rodas, rodasError, rodasStep]]) {
  assert.deepEqual(trajectory.state.shape, [4, 2]);
  assert.deepEqual(trajectory.error.shape, [4, 2]);
  assert.deepEqual(vector(trajectory.state), reference.flatMap(vector));
  const errors = vector(trajectory.error);
  assert.deepEqual(errors.slice(0, 2), [0, 0]);
  assert.deepEqual(errors.slice(4, 6), vector(step.error));
  assert.deepEqual(vector(trajectory.state).slice(4, 6), vector(step.state));
  assert.ok(vector(step.error).every(Number.isFinite));
  assert.ok(vector(step.error).some(value => value !== 0));
}
console.log('ODE embedded error grid checks passed');
