import matplotlib.pyplot as plt
from casadi import *

x = MX.sym("x")
y = MX.sym("y")
p = MX.sym("p")
xy = vertcat(x,y)

f = (1-x)**2 + 0.2*(y-x**2)**2
g = vertcat(
  (x+0.5)**2+y**2-(p/2)**2,
  (x+0.5)**2+y**2-p**2
)

lbg = [0, -inf]
ubg = [inf, 0]
lbx = [0]
ubx = [inf]
x0  = [0,0]

nlp = {"x":xy, "p":p, "f":f, "g":g}

opts = {};
opts["ipopt.print_level"] = 0;
opts["verbose"] = 0;
opts["print_time"] = False;
solver = nlpsol("nlp",'ipopt', nlp, opts); 
print(solver)
N = 100
M = Function('M',[p],[solver(x0=x0,p=p,lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg)["x"]]);
Mmap = M.map(N)
z = lambda xy: xy[1,:]-xy[0,:];

pvec = linspace(1,2,N);
S = Mmap(pvec).full();

plt.figure(1)
plt.clf()
plt.plot(pvec,z(S),'o',color='black',markersize=5);

opts = {};
opts["qpsol"] = 'qrqp';
opts["verbose"] = 0;
opts["print_time"] = False;
solver = nlpsol('qpsol', 'sqpmethod', nlp, opts)

Z = Function('Z',[p,xy],[z(solver(x0=xy,p=p,lbx=lbx,ubx=ubx,lbg=lbg,ubg=ubg)["x"])]);
zp = Z(p,xy);

j = jacobian(zp,p);
h = hessian(zp,p)[0];
Z_taylor = Function('Z_taylor',[p, xy],[zp,j,h]);

# plt.figure(2)
# plt.clf()
colors = ['red','green','blue']
for p0,color in  zip([1.25,1.5,2],colors):
  [F,J,H] = Z_taylor(p0,M(p0));
  plt.plot(p0,F.full(),'x',markersize=15, color=color);
  plt.plot(pvec,(F+J*(pvec-p0)+1/2*H*(pvec-p0).full()**2),'-',color=color);

plt.ylim([-0.6, 0.2])

plt.grid()
plt.show()
