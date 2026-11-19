const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const source = `flatppl_compat="0.1"
r=load_module("../src/rodas4p2.flatppl")
f(t,y,p)=-p .* y
jac(t,y,p)=-p .* eye(lengthof(y))
dfdt(t,y,p)=y .* 0.0
full=r.rodas4p2_step_with_error(f,jac,dfdt,0.0,[1.0],0.1,1.0)
half=r.rodas4p2_step_with_error(f,jac,dfdt,0.0,[1.0],0.05,1.0)
plain=r.rodas4p2_step(f,jac,dfdt,0.0,[1.0],0.1,1.0)
zero=r.rodas4p2_step_with_error(f,jac,dfdt,0.0,[1.0,2.0],0.0,1.0)
back=r.rodas4p2_step_with_error(f,jac,dfdt,0.0,[1.0],-0.1,1.0)
p=record(rate=120.0,a=1.2,b=0.8)
forcing(t,p)=[p.a*sin(t),p.b*cos(t)]
forcing_dt(t,p)=[p.a*cos(t),-p.b*sin(t)]
J(t,y,p)=rowstack([[-(p.rate+2)/2,(2-p.rate)/2],[(2-p.rate)/2,-(p.rate+2)/2]])
coupled(t,y,p)=J(t,y,p)*(y .- forcing(t,p)) .+ forcing_dt(t,p)
partial_t(t,y,p)=-J(t,y,p)*forcing_dt(t,p) .- forcing(t,p)
z=forcing(0.2,p) .+ (0.1*exp(-p.rate*0.2))
value=r.rodas4p2_step_with_error(coupled,J,partial_t,0.2,z,0.03,p)
adapter=r.rodas4p2_grid_step_with_error(coupled,0.2,z,0.03,(J,partial_t,p))
`;
const names = ['full','half','plain','zero','back','value','adapter'];
const [full,half,plain,zero,back,value,adapter] = JSON.parse(execFileSync(process.execPath,
  [path.join(__dirname,'evaluate.cjs')],{input:JSON.stringify({source,bindings:names}),encoding:'utf8',env:process.env}));
const data = value => Object.values(value.data);
const near = (a,b,tol) => assert.ok(Math.abs(a-b)<tol,`${a} vs ${b}`);
assert.deepEqual(data(full.state),data(plain));
assert.deepEqual(data(zero.state),[1,2]);
assert.ok(data(zero.error).every(v=>v===0));
assert.ok(data(back.state)[0]>1);
near(data(back.error)[0],4.4334004004950594e-7,1e-16);
// Independently solve the pinned 6-stage tableau as one block system; the last
// stage is the signed high-minus-embedded error, without subtracting two states.
near(data(full.error)[0],3.4525558110156403e-7,1e-16);
assert.ok(data(full.error)[0]/data(half.error)[0]>14);
const expectedState = [0.2735730286080971,0.7789331163894625];
const expectedError = [-9.755422164464274e-10,6.256435154364584e-10];
data(value.state).forEach((v,i)=>near(v,expectedState[i],1e-14));
data(value.error).forEach((v,i)=>near(v,expectedError[i],1e-15));
assert.ok(data(value.error)[0]<0 && data(value.error)[1]>0);
assert.deepEqual(value,adapter);
console.log(JSON.stringify({sameState:true,zeroError:true,signedError:expectedError,embeddedOrder:true}));
