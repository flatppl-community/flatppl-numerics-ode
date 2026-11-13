const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const methods = {euler: 1, heun: 2, rk4: 4, vern7: 7, rosenbrock23: 2, sdirk2: 2};
let source = 'flatppl_compat="0.1"\node=load_module("../main.flatppl")\nf(t,y,p)=p .* y\n';
const bindings = [];
for (const [method, order] of Object.entries(methods)) {
  for (let j = 0; j < 3; j++) {
    const n = (order === 7 ? 1 : 4) * 2 ** j;
    const times = Array.from({length: n + 1}, (_, i) => (i / n).toFixed(12));
    const name = `${method}_${j}`;
    source += `${name}=ode.${method}(f,[1.0],[${times.join(',')}],-1.0)\n`;
    bindings.push(name);
  }
}
const values = JSON.parse(execFileSync(process.execPath, [path.join(__dirname, 'evaluate.cjs')], {
  input: JSON.stringify({ source, bindings }), encoding: 'utf8', env: process.env,
}));
for (const [index, [method, order]] of Object.entries(Object.entries(methods))) {
  const error = values.slice(index * 3, index * 3 + 3).map(trajectory =>
    Math.abs(trajectory.at(-1).data['0'] - Math.exp(-1)));
  const ratio = error[1] / error[2];
  assert.ok(ratio > 2 ** (order - 0.8), `${method}: ${error}, ratio ${ratio}`);
  console.log(`${method}: ${error.join(', ')}; ratio ${ratio}`);
}
