from casadi import SX, Function, vertcat, nlpsol, external
from casadi import hessian, gradient, jacobian, triu, dot
from os import system
from multiprocessing import  Pool


# Example pack Rosenbrock
x = SX.sym("x")
y = SX.sym("y")
z = SX.sym("z")
f = x**2 + 100*z**2
g = z + (1-x)**2 - y
nlp = {'x':vertcat(x,y,z), 'f':f, 'g':g}

w = nlp['x']
J = nlp['f']
g = nlp['g']

lam_f = SX.sym("lam_f")
lam_g = SX.sym("lam_g", g.sparsity())
param = SX.sym('param')

L = lam_f*J + dot(lam_g, g)
H, j = hessian(L,w)
Htriu = triu(H)

grad_f = Function("grad_f",[w, param],[J, gradient(J,w)])
jac_g = Function("jac_g",[w, param],[g, jacobian(g,w)])
hess_lag = Function("hess_lag", [w, param, lam_f, lam_g], [Htriu])

def compile_fun(fun):
  compiler = "gcc"
  flags = "-fPIC -shared -O3"
  src_file = fun.generate()
  so_file = src_file.replace(".c",".so")
  system(f"{compiler} {flags} {src_file} -o {so_file}")
  compiled_fun = external(fun.name(), so_file)
  return compiled_fun

# sequential
# [gf, jg, hl] = [compile_fun(fun) for fun in funs]
# parallel
pool = Pool(processes=3)
funs = [grad_f, jac_g, hess_lag]
[gf, jg, hl] = pool.map(compile_fun, funs)

opts = {'grad_f':gf, 'jac_g':jg, 'hess_lag':hl}

solver = nlpsol('solver', 'ipopt', nlp, opts)
res = solver(x0 = [2.5,3.0,0.75],
             ubg = 0,
             lbg = 0)

print()
print(f"{'Optimal cost:':>50}: {res['f']}")
print(f"{'Primal solution':>50}: {res['x']}")
print(f"{'Dual solution (simple bounds)':>50}: {res['lam_x']}")
print(f"{'Dual solution (nonlinear bounds)':>50}: {res['lam_g']}")
