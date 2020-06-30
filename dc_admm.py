import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

n = 100
dim = 2
pi = 3.14159265

X = np.random.rand(n, dim)
y = np.sin(3*pi*X[:,0]) + np.cos(3*pi*X[:,1]) + 3*np.log(np.abs(3*pi*X[:,0]+3*pi*X[:,1])+1)

# X = 4*np.random.rand(n, dim) - 2
# y = np.sum(np.abs(X)**2, axis = 1) + 0.25*np.random.randn(n)

# X = np.load('X.npy')
# y = np.load('y.npy')

lanbda = 0.0001

rho = 0.01
T = 300

# initial values
# primal
y_hat = np.zeros(n)
z = np.zeros(n)
a = np.zeros([n, dim])
b = np.zeros([n, dim])
p = np.zeros([n, dim])
q = np.zeros([n, dim])

L = 0

# slack
s = np.zeros([n, n])
t = np.zeros([n, n])
u = np.zeros(n)

# dual
alpha = np.zeros([n, n])
beta = np.zeros([n, n])
gamma = np.zeros(n)
eta = np.zeros([n, dim])
zeta = np.zeros([n, dim])

# preprocess1
XjXj = 0
for i in range(n):
    XjXj = XjXj + np.outer(X[i],X[i])
Xbar = np.mean(X, axis = 0)

Sigma_i = []
for i in range(n):
    Sigma_i += [n*np.outer(X[i],X[i]) + XjXj - n*(np.outer(Xbar,X[i]) + np.outer(X[i],Xbar))]
    Sigma_i[-1] = linalg.inv(Sigma_i[i] + np.eye(dim))

# ADMM iteration
for iter in tqdm(range(T)):
    #   primal updates
    #   y_hat & z update
    for i in range(n):
        y_hat[i] = 2/(2+n*rho) * y[i]
        z[i] = -1/(2+n*rho)*y[i]
        for j in range(n):
            temp1 = alpha[j,i] -  alpha[i,j] + s[j,i] - s[i,j] + np.dot(a[i] + a[j], X[i] - X[j]) + 2*y[j]
            temp2 = beta[j,i] - beta[i,j] + t[j,i] - t[i,j] + np.dot(b[i] + b[j], X[i] - X[j])

            y_hat[i] +=   rho/(2+n*rho)/2 * temp1  - rho/(2+n*rho)/2 * temp2
            z[i] += 1/(2*n)/(2+n*rho)* temp1 + (1+n*rho)/(2*n)/(2+n*rho)* temp2

    #   a update
    for i in range(n):
        a[i] = p[i] - eta[i]
        for j in range(n):
            a[i] += (alpha[i,j] + s[i,j] + y_hat[i] - y_hat[j] + z[i] - z[j])*(X[i]-X[j])
        a[i] = np.matmul(Sigma_i[i], a[i])

    #   b update
    for i in range(n):
        b[i] = q[i] - zeta[i]
        for j in range(n):
            b[i] += (beta[i,j] + t[i,j] + z[i] - z[j])*(X[i]-X[j])
        b[i] = np.matmul(Sigma_i[i], b[i])

    #   p updates
    for i in range(n):
        temp3 = 0
        for d in range(dim):
            temp3 += np.abs(p[i,d]) + np.abs(q[i,d])
        for d in range(dim):
            temp1 = 1/2* ( a[i,d] + eta[i,d])
            temp2 = 1/2*( L - u[i] - gamma[i] + np.abs(p[i,d]) - temp3)
            p[i,d] = np.sign(temp1)*np.maximum(np.abs(temp1)+temp2, 0)

    #   q updates
    for i in range(n):
        temp3 = 0
        for d in range(dim):
            temp3 += np.abs(p[i,d]) + np.abs(q[i,d])
        for d in range(dim):
            temp1 = 1/2* ( b[i,d] + zeta[i,d])
            temp2 = 1/2*( L - u[i] - gamma[i] + np.abs(q[i,d]) - temp3)
            q[i,d] = np.sign(temp1)*np.maximum(np.abs(temp1)+temp2, 0)

    #   L update
    L = -1/(n*rho)* lanbda
    for i in range(n):
        L +=  1/n*( gamma[i]  + u[i])
        for d in range(dim):
            L += 1/n* (np.abs(p[i,d]) +np.abs(q[i,d]))
    
    #   slack updates
    #   s & t update
    for i in range(n):
        for j in range(n):
            s[i,j] = -alpha[i,j] - y_hat[i] + y_hat[j] - z[i] + z[j] + np.dot(a[i], X[i]-X[j])
            s[i,j] = np.maximum(s[i,j] ,0)

            t[i,j] = -beta[i,j] - z[i] + z[j] + np.dot(b[i], X[i]-X[j])
            t[i,j] = np.maximum(t[i,j] ,0)
            
    #   u update
    for i in range(n):
        u[i] = -gamma[i] + L
        for d in range(dim):
            u[i] +=  - np.abs(q[i,d]) - np.abs(p[i,d])
        u[i] = np.maximum(u[i], 0)
    
    #   dual updates
    for i in range(n):
        for j in range(n):
            alpha[i,j] +=  s[i,j] + y_hat[i] - y_hat[j] + z[i] - z[j] - np.dot(a[i], X[i]-X[j])
            beta[i,j] +=  t[i,j] + z[i] - z[j] - np.dot(b[i], X[i]-X[j])
   
    for i in range(n):
        gamma[i] += u[i] - L 
        for d in range(dim):
            gamma[i] +=  np.abs(p[i,d]) + np.abs(q[i,d])
            eta[i,d] +=  a[i,d] - p[i,d]
            zeta[i,d] +=  b[i,d] - q[i,d]

y_hat = y_hat + z - np.sum(a*X, axis = 1)
z = z - np.sum(b*X, axis = 1)

f_hat  = lambda X: np.max(y_hat.reshape(1,-1) + np.matmul(X,a.T), axis=1) - np.max(z.reshape(1,-1) + np.matmul(X,b.T), axis=1)

# plotting
fig = plt.figure()
ax = fig.gca(projection='3d')

# make data.
d_mesh = 0.01
X_mesh = np.arange(np.min(X[:,0]), np.max(X[:,0]), d_mesh)
Y_mesh = np.arange(np.min(X[:,1]), np.max(X[:,1]), d_mesh)
X_mesh, Y_mesh = np.meshgrid(X_mesh, Y_mesh)

f_hat_mesh = f_hat(np.concatenate((X_mesh.reshape(-1,1),
Y_mesh.reshape(-1,1)), axis=1)).reshape(X_mesh.shape)

# Plot the surface.
surf = ax.plot_surface(X_mesh, Y_mesh, f_hat_mesh, cmap = cm.coolwarm)
ax.scatter(X[:,0],X[:,1],y)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])

plt.show()