const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');

const source = `flatppl_compat = "0.1"
ode = load_module("../main.flatppl")
A = rowstack([[0.0, 2.0], [0.0, 0.0]])
b = [1.0, 3.0]
initial = [4.0, 5.0]
singular = ode.affine(A, b, initial, 0.25, 4)
step = ode.affine_step(A, b, initial, 0.25)
backward = ode.affine(A, b, initial, -0.25, 4)
empty = ode.affine(A, b, initial, 0.25, 0)
zero = ode.affine(A, b, initial, 0.0, 3)
forced = ode.affine(rowstack([[-1000.0]]), [2000.0], [5.0], 0.1, 2)
rotation = ode.affine(rowstack([[0.0, -1.0], [1.0, 0.0]]), [0.0, 0.0], [1.0, 0.0], 0.2, 5)
constant = ode.affine(rowstack([[0.0, 0.0], [0.0, 0.0]]), b, initial, 0.25, 4)`;
const bindings = ['singular', 'step', 'backward', 'empty', 'zero', 'forced', 'rotation', 'constant'];
const results = JSON.parse(execFileSync(process.execPath, [path.join(__dirname, 'evaluate.cjs')], {
  input: JSON.stringify({ source, bindings }), encoding: 'utf8', env: process.env,
}));
const values = value => Object.values(value.data);
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 2e-12 * Math.max(1, Math.abs(expected)), `${actual} vs ${expected}`);
const trajectory = (value, count, dimension, oracle) => {
  assert.deepEqual(value.shape, [count, dimension]);
  values(value).forEach((actual, i) => close(actual, oracle(Math.floor(i / dimension))[i % dimension]));
};
const polynomial = t => [4 + 11 * t + 3 * t * t, 5 + 3 * t];
trajectory(results[0], 5, 2, i => polynomial(i * 0.25));
values(results[1]).forEach((actual, i) => close(actual, polynomial(0.25)[i]));
trajectory(results[2], 5, 2, i => polynomial(-i * 0.25));
trajectory(results[3], 1, 2, () => [4, 5]);
trajectory(results[4], 4, 2, () => [4, 5]);
trajectory(results[5], 3, 1, i => [2 + 3 * Math.exp(-100 * i)]);
trajectory(results[6], 6, 2, i => [Math.cos(i * 0.2), Math.sin(i * 0.2)]);
trajectory(results[7], 5, 2, i => [4 + i * 0.25, 5 + 3 * i * 0.25]);
console.log('Affine ODE checks passed');
