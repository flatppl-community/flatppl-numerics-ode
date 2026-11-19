const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');

// Independently derive high and embedded stability polynomials from their
// rational Butcher weights, rather than applying the implementation's btilde sum.
const high=[1,1,1/2,1/6,1/24,1/120,1/720,1/5040,2.5177692902871564e-5,2.957324245145149e-6];
const difference=[0,0,0,0,0,0,0,-1.699697733070476e-5,-1.100863361978215e-5,2.957324245145149e-6];
const polynomial=(coefficients,x)=>coefficients.reduceRight((sum,c)=>sum*x+c,0);
const cases=[{p:-1,h:1,y:[1,-0.5]},{p:1.3,h:0.7,y:[0.8,1.2]},{p:0.8,h:-0.9,y:[-0.4,0.7]}];
let source='flatppl_compat="0.1"\node=load_module("../main.flatppl")\nf(t,y,p)=p .* y\nforce(t,y,p)=t^6 .* p\n';
const bindings=[];
for(const [i,c] of cases.entries()) {
  source+=`v${i}=ode.vern7_step_with_error(f,0.2,${JSON.stringify(c.y)},${c.h},${c.p})\n`;
  source+=`s${i}=ode.vern7_step(f,0.2,${JSON.stringify(c.y)},${c.h},${c.p})\n`;
  bindings.push(`v${i}`,`s${i}`);
}
for(const [i,h] of [1,0.5,0.25].entries()) {
  source+=`q${i}=ode.vern7_step_with_error(force,0.2,[0.3,-0.7],${h},[1.0,-2.0])\n`;
  bindings.push(`q${i}`);
}
source+='zero=ode.vern7_step_with_error(f,0.2,[0.3,-0.7],0.0,1.3)\n';
bindings.push('zero');
const raw=JSON.parse(execFileSync(process.execPath,[path.join(__dirname,'evaluate.cjs')],{
  input:JSON.stringify({source,bindings}),encoding:'utf8',env:process.env,
}));
const vector=value=>Object.values(value.data);
const close=(a,b)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<3e-14*Math.max(1,Math.abs(b)),`${a} vs ${b}`);
for(const [i,c] of cases.entries()) {
  const result=raw[2*i];
  const state=vector(result.state);
  assert.deepEqual(state,vector(raw[2*i+1]));
  state.forEach((v,j)=>close(v,c.y[j]*polynomial(high,c.h*c.p)));
  vector(result.error).forEach((v,j)=>close(v,c.y[j]*polynomial(difference,c.h*c.p)));
}
// The principal quadrature integrates t^6 exactly. The embedded defect is
// p*h^7 times its sixth moment, independent of the starting time.
const moment=9.444274853486327e-5;
const defects=[];
for(const [i,h] of [1,0.5,0.25].entries()) {
  const result=raw[6+i];
  const integral=((0.2+h)**7-0.2**7)/7;
  vector(result.state).forEach((v,j)=>close(v,[0.3,-0.7][j]+[1,-2][j]*integral));
  vector(result.error).forEach((v,j)=>close(v,[1,-2][j]*moment*h**7));
  defects.push(vector(result.error)[0]);
}
assert.ok(Math.abs(defects[0]/defects[1]-128)<1e-7);
assert.ok(Math.abs(defects[1]/defects[2]-128)<1e-7);
assert.deepEqual(vector(raw.at(-1).state),[0.3,-0.7]);
vector(raw.at(-1).error).forEach(v=>close(v,0));
console.log(JSON.stringify({signed_defects:cases.map((_,i)=>vector(raw[2*i].error)),time_force_defects:defects}));
