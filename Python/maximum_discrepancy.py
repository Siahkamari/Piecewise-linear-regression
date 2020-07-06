import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

n = 20
dim = 2

# X = np.random.rand(n, dim)

X = 4*np.random.rand(n, dim) - 2
y = np.sum(np.abs(X)**2, axis = 1) + 0.25*np.random.randn(n)

# X = np.load('X2.npy')
# y = np.load('y.npy')

rho = 0.1
T = 1000

# initial values
# primal
y_hat = np.zeros(n)
z = np.zeros(n)
a = np.zeros([n, dim])
b = np.zeros([n, dim])
p = np.zeros([n, dim])
q = np.zeros([n, dim])


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
        for j in range(n):
            temp1 = alpha[j,i] -  alpha[i,j] + s[j,i] - s[i,j] + np.dot(a[i] + a[j], X[i] - X[j])
            temp2 = beta[j,i] - beta[i,j] + t[j,i] - t[i,j] + np.dot(b[i] + b[j], X[i] - X[j])

            y_hat[i] =   (-1)**(i)/n/rho + (temp1 - temp2)/(2*n)
            z[i] = (-1)**(i+1)/(2*n*rho) + temp2/(2*n)
            # y_hat[i] =   -z[i] + 1/(2*n*rho)*((-1)**i+rho*temp1)
            # z[i] = -y_hat[i]/2 + (temp1+temp2)/4/n

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
            temp2 = 1/2*( 1 - u[i] - gamma[i] + np.abs(p[i,d]) - temp3)
            p[i,d] = np.sign(temp1)*np.maximum(np.abs(temp1)+temp2, 0)

    #   q updates
    for i in range(n):
        temp3 = 0
        for d in range(dim):
            temp3 += np.abs(p[i,d]) + np.abs(q[i,d])
        for d in range(dim):
            temp1 = 1/2* ( b[i,d] + zeta[i,d])
            temp2 = 1/2*( 1 - u[i] - gamma[i] + np.abs(q[i,d]) - temp3)
            q[i,d] = np.sign(temp1)*np.maximum(np.abs(temp1)+temp2, 0)

    
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
        u[i] = -gamma[i] + 1
        for d in range(dim):
            u[i] +=  - np.abs(q[i,d]) - np.abs(p[i,d])
        u[i] = np.maximum(u[i], 0)
    
    #   dual updates
    for i in range(n):
        for j in range(n):
            alpha[i,j] +=  s[i,j] + y_hat[i] - y_hat[j] + z[i] - z[j] - np.dot(a[i], X[i]-X[j])
            beta[i,j] +=  t[i,j] + z[i] - z[j] - np.dot(b[i], X[i]-X[j])
   
    for i in range(n):
        gamma[i] += u[i] - 1 
        for d in range(dim):
            gamma[i] +=  np.abs(p[i,d]) + np.abs(q[i,d])
            eta[i,d] +=  a[i,d] - p[i,d]
            zeta[i,d] +=  b[i,d] - q[i,d]

Dn = 0
for i in range(dim):
    Dn += 2/n* (-1)**(i) *y_hat[i]

print(Dn)
