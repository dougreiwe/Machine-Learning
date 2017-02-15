import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# a = b
a, b, m = 0.8, 0.8, 5
# a < b
#a, b, m = 0.5, 0.6, 4
# a > b
a, b, m = 3, 0.05, 1.2
def f(u,v):
    return u * (1 - (u + (a * v)))
def g(u,v):
    return m * v * (1 - (v + (b * u)))
# initialize lists containing values
u = []
v = []
# define system in terms of a Numpy array
def Sys(X, t=0):
    # here X[0] = u and X[1] = v
    return np.array([ X[0]*(1 - (X[0]+ (a*X[1]))) , m*X[1]*(1 - (X[1] + (b*X[0]))) ])
# generate 1000 linearly spaced numbers for x-axes
t = np.linspace(0, 20,  1500)
# initial values: u0 = 0.8, v0 = 0.2
Sys0 = np.array([0.8, 0.2])
# type "help(integrate.odeint)" if you want more information about integrate.odeint inputs and outputs.
X, infodict = integrate.odeint(Sys, Sys0, t, full_output=True)
u,v = X.T
#plot
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(u, 'r-', label='Existing')
ax1.plot(v, 'b-', label='New')
ax1.set_title("Dynamics in Time")
ax1.set_xlabel("Time")
ax1.grid()
ax1.legend(loc='best')
ax2.plot(u, v, color="blue")
ax2.set_xlabel("u")
ax2.set_ylabel("v")
ax2.set_title("Phase Space")
ax2.grid()
plt.show()
print 'Steady States:'
print ''
fp = []
def find_fixed_points(r):
    for u in range(r):
        for v in range(r):
            if ((f(u,v) == 0) and (g(u,v) == 0)):
                fp.append((u,v))
                print('The system has a steady state at (%s,%s)' % (u,v))
    return fp
find_fixed_points(10)
print ''
print ''
print 'Stability:'
print ''
from scipy import sqrt
def eigenvalues(u, v):
    a11 = 1 - (2 * u) - (a * v)  # differentiated with respect to u
    a12 = - (u * a)  # differentiated with respect to v
    a21 = - (m * v * b)  # differentiated with respect to u
    a22 = m * (1 - (2 * v)-(b * u))  # differentiated with respect to v
    tr = a11 + a22
    det = a11 * a22 - a12 * a21
    lambda1 = (tr - sqrt(tr ** 2 - 4 * det)) / 2
    lambda2 = (tr + sqrt(tr ** 2 - 4 * det)) / 2
    print('Checking the fixed point (%s, %s)...' % (u, v))
    print('The real part of the first eigenvalue is %s' % lambda1.real)
    print('The real part of the second eigenvalue is %s' % lambda2.real)
    if (lambda1.real < 0 and lambda2.real < 0):
        print('The fixed point in %s, %s is a sink. It is stable' % (u, v))
    if (lambda1.real > 0 and lambda2.real > 0):
        print('The fixed point in %s, %s is a source. It is unstable' % (u, v))
    if (lambda1.real > 0 and lambda2.real < 0):
        print('The fixed point in %s, %s is a saddle. It is unstable' % (u, v))
    if (lambda1.real < 0 and lambda2.real > 0):
        print('The fixed point in (%s, %s) is unstable' % (u, v))
    print ''
    return lambda1, lambda2
# iterate through list of fixed points
for u, v in fp:
    eigenvalues(u, v)
fig2 = plt.figure(figsize=(8,6))
ax4 = fig2.add_subplot(1,1,1)
u = np.linspace(0,2,20)
v = np.arange(0,2,20)
# plot nullclines
ax4.plot([0,1],[1,0], 'r-', lw=2, label='x-nullcline')
ax4.plot([1,1],[0,2], 'b-', lw=2, label='y-nullcline')
# plot fixed points
for point in fp:
    ax4.plot(point[0],point[1],"red", marker = "o", markersize = 10.0)
ax4.set_title("Quiverplot with Nullclines")
ax4.legend(loc='best')
# quiverplot
# define a grid and compute direction at each point
u = np.linspace(0, 2, 20)
v = np.linspace(0, 2, 20)
U1 , V1  = np.meshgrid(u, v)                    # create a grid
DU1, DV1 = Sys([U1, V1])                        # compute growth rate on the grid
M = (np.hypot(DU1, DV1))                        # norm growth rate
M[ M == 0] = 1.                                 # avoid zero division errors
DU1 /= M                                        # normalize each arrows
DV1 /= M
ax4.quiver(U1, V1, DU1, DV1, M, pivot='mid')
ax4.legend()
ax4.grid()
plt.show()