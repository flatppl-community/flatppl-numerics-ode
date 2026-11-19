const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');

let source = `flatppl_compat = "0.1"
ode = load_module("../main.flatppl")
constant(t, q, p) = p
timed(t, q, p) = t .* p
harmonic(t, q, p) = -p .* q
driven(t, q, p) = -p .* q .+ [sin(t), cos(t)]
step = ode.velocity_verlet_step(constant, 0.2, [1.0,-2.0], [0.5,1.0], 0.3, [2.0,-3.0])
decreasing = ode.velocity_verlet(constant, [1.0,-2.0], [0.5,1.0], [1.2,0.8,0.55,-0.1], [2.0,-3.0])
time_force = ode.velocity_verlet(timed, [1.0], [0.3], [0.0,0.1,0.3,0.6], [2.0])
grid = [0.2,0.3,0.55,1.0,1.4]
forward = ode.velocity_verlet(driven, [0.7,-0.2], [-0.3,0.5], grid, 1.7)
backward = ode.velocity_verlet(driven, forward.position[5], forward.velocity[5], reverse(grid), 1.7)
long_run = ode.velocity_verlet(harmonic, [1.0], [0.0], linspace(0.0,200.0,2001), 1.0)
`;
const bindings = ['step','decreasing','time_force','forward','backward','long_run'];
for (const n of [10,20,40]) {
  source += `order${n}=ode.velocity_verlet(harmonic,[0.7],[-0.2],linspace(0.0,1.3,${n+1}),2.89)\n`;
  bindings.push(`order${n}`);
}
source += `pulse(t,q,p)=[ifelse(t<0.0,1.0,0.0)]\nendpoint=ode.velocity_verlet(pulse,[0.0],[0.0],[1.0,-1e-16],0.0)\n`;
bindings.push('endpoint');

function decode(value) {
  if (Array.isArray(value)) return value.map(decode);
  if (value && value.shape) {
    const data = Object.values(value.data).map(decode);
    if (value.shape.length === 0) return data[0];
    if (value.shape.length === 1) return data;
    return Array.from({length:value.shape[0]},(_,i)=>data.slice(i*value.shape[1],(i+1)*value.shape[1]));
  }
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,decode(v)]));
  return value;
}
const values = decode(JSON.parse(execFileSync(process.execPath,[path.join(__dirname,'evaluate.cjs')],{
  input:JSON.stringify({source,bindings}),encoding:'utf8',env:process.env,
})));
const close = (a,b,tolerance=3e-12)=>assert.ok(Number.isFinite(a) && Math.abs(a-b)<tolerance,`${a} vs ${b}`);
function vector(actual,expected) {
  assert.equal(actual.length,expected.length);
  actual.forEach((value,i)=>close(value,expected[i]));
}
vector(values[0].position,[1.24,-1.835]);
vector(values[0].velocity,[1.1,0.1]);
for (const [i,t] of [1.2,0.8,0.55,-0.1].entries()) {
  const h=t-1.2;
  vector(values[1].position[i],[1+0.5*h+h*h,-2+h-1.5*h*h]);
  vector(values[1].velocity[i],[0.5+2*h,1-3*h]);
}
let cubes=0;
const times=[0,0.1,0.3,0.6];
for (const [i,t] of times.entries()) {
  if(i)cubes+=(t-times[i-1])**3;
  close(values[2].position[i][0],1+0.3*t+(t**3-cubes)/3);
  close(values[2].velocity[i][0],0.3+t*t);
}
vector(values[4].position.at(-1),[0.7,-0.2]);
vector(values[4].velocity.at(-1),[-0.3,0.5]);
const energies=values[5].position.map((q,i)=>(q[0]**2+values[5].velocity[i][0]**2)/2);
assert.ok(Math.max(...energies.map(e=>Math.abs(e/0.5-1)))<=0.1**2/4+2e-11);
const exactQ=0.7*Math.cos(1.7*1.3)-0.2/1.7*Math.sin(1.7*1.3);
const exactV=-0.7*1.7*Math.sin(1.7*1.3)-0.2*Math.cos(1.7*1.3);
const errors=values.slice(6,9).map(state=>Math.hypot(state.position.at(-1)[0]-exactQ,state.velocity.at(-1)[0]-exactV));
assert.ok(errors[0]/errors[1]>3.9 && errors[1]/errors[2]>3.9,JSON.stringify(errors));
close(values.at(-1).velocity[1][0],-0.5);
console.log(JSON.stringify({order_errors:errors,max_relative_energy_error:Math.max(...energies.map(e=>Math.abs(e/0.5-1)))}));
