const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
let source = `flatppl_compat="0.1"
ode=load_module("../main.flatppl")
f(t,y,p)=-p .* y
jac(t,y,p)=-p .* eye(lengthof(y))
dfdt(t,y,p)=y .* 0.0
zero=ode.rodas4p2_step(f,jac,dfdt,0.0,[1.0,2.0],0.0,3.0)
backward=ode.rodas4p2_step(f,jac,dfdt,0.1,[1.0],-0.1,1.0)
`;
const bindings = ['zero', 'backward'];
for (const n of [10, 20, 40]) {
  source += `order${n}=ode.rodas4p2(f,jac,dfdt,[1.0],[${Array.from({length:n+1},(_,i)=>i/n)}],1.0)\n`;
  bindings.push(`order${n}`);
}
source += `cp=record(rate=1000.0,a=1.3,b=0.6)
forcing(t,p)=[p.a*sin(t),p.b*cos(t)]
forcing_dt(t,p)=[p.a*cos(t),-p.b*sin(t)]
J(t,y,p)=rowstack([[-(p.rate+2)/2,(2-p.rate)/2],[(2-p.rate)/2,-(p.rate+2)/2]])
coupled(t,y,p)=J(t,y,p)*(y .- forcing(t,p)) .+ forcing_dt(t,p)
partial_t(t,y,p)=-J(t,y,p)*forcing_dt(t,p) .- forcing(t,p)
curve=ode.rodas4p2(coupled,J,partial_t,[0.1,0.7],[0.0,0.003,0.01,0.03,0.1,0.25,0.5,1.0],cp)
`;
bindings.push('curve');
const values = JSON.parse(execFileSync(process.execPath,[path.join(__dirname,'evaluate.cjs')],{
  input:JSON.stringify({source,bindings}),encoding:'utf8',env:process.env,
}));
const data = value => Object.values(value.data);
assert.deepEqual(data(values[0]),[1,2]);
assert.ok(Math.abs(data(values[1])[0]-Math.exp(.1))<1e-7);
const errors = values.slice(2,5).map(curve=>Math.abs(data(curve.at(-1))[0]-Math.exp(-1)));
assert.ok(errors[0]/errors[1]>14 && errors[1]/errors[2]>14,`fourth order: ${errors}`);
// Independently solve all six stages as a single12-by12 block system using the
// pinned SciML tableau. This is a coupled nonautonomous stiff system, not decay.
const expected = [
  [.1,.7], [.010492485373919516,.6065897912259419],
  [.01362585803035065,.6005960749459494], [.03910131404102929,.5998371840301143],
  [.12979405803098665,.5970131168107302], [.3216257154545471,.5813480528070162],
  [.6232533260910422,.5265495653307148], [1.0939250789771078,.32416974375239593],
];
values[5].forEach((state,i)=>data(state).forEach((v,j)=>
  assert.ok(Math.abs(v-expected[i][j])<2e-13,`coupled state ${i}/${j}: ${v} vs ${expected[i][j]}`)));
console.log(JSON.stringify({orderErrors:errors,coupledStates:expected.length,zeroStep:true,backward:true}));
