import numpy as np
import numpy.linalg as linalg
from tqdm import tqdm

class dc_regression:

    def __init__(self):
        self.y_hat = 0
        self.z = 0
        self.a = 0
        self.b = 0
        self.lanbda = 1

    def auto_tune(self, X, y, max_hyper_iter = 10):
        n_folds = 5

        lanbdas = [1e-3,1e-2,1e-1,1,1e1,1e2,1e3]

        for _ in range(max_hyper_iter):
            i = 0
            loss = np.zeros(len(lanbdas))
            for lanbda in tqdm(lanbdas):
                self.lanbda = lanbda
                loss[i] = self.cross_validate(X, y, n_folds)
                i += 1

            arg_min = np.argmin(loss)
            lanbda = lanbdas[arg_min]
            if  lanbda == lanbdas[0]:
                lanbdas = [i*lanbda for i in [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1]]
            elif lanbda == lanbdas[-1]:
                lanbdas = [i*lanbda for i in [1e-1,1,1e1,1e2,1e3,1e4,1e5]]
            else:
                if len(lanbdas) == 7:
                    lanbdas = [i*lanbda for i in [0.0625,0.125,0.25,0.5,1,2,4,8,16]]
                else:
                    self.lanbda = lanbda
                    self.fit(X, y, lanbda)
                    break

    def fit(self, X, y, lanbda = False, T = False):

        if lanbda == False:
            self.auto_tune(X, y)
            return
        else:
            self.lanbda = lanbda


        n, dim = X.shape
        rho = 0.01
        if T == False:
            T = 2*n

        # initial values
        # primal
        y_hat = np.zeros(n)
        z = np.zeros(n)
        a = np.zeros([n, dim])
        b = np.zeros([n, dim])
        p = np.zeros([n, dim])
        q = np.zeros([n, dim])

        L = np.zeros([1, dim])

        # slack
        s = np.zeros([n, n])
        t = np.zeros([n, n])
        u = np.zeros([n, dim])

        # dual
        alpha = np.zeros([n, n])
        beta = np.zeros([n, n])
        gamma = np.zeros([n, dim])
        eta = np.zeros([n, dim])
        zeta = np.zeros([n, dim])

        # preprocess1
        XjXj = np.dot(X.T,X)
        Xbar = np.mean(X, axis = 0)
        ybar = np.mean(y)
        Sigma_i = np.zeros([n,dim,dim])

        for i in range(n):
            Sigma_i[i,:,:] = n*np.outer(X[i],X[i]) + XjXj - n*(np.outer(Xbar,X[i]) + np.outer(X[i],Xbar))
            Sigma_i[i,:,:] = linalg.inv(Sigma_i[i,:,:] + np.eye(dim))

        # ADMM iteration
        for _ in range(T):
    
            #   primal updates
            #   y_hat & z update
            temp1 = np.sum(alpha.T -  alpha + s.T - s -  np.dot(a,X.T) + np.dot(X,a.T), axis=1) + n*2*ybar + n*np.sum(a*X,axis=1) - np.sum(a*X)
            temp2 = np.sum(beta.T - beta + t.T - t - np.dot(b,X.T) + np.dot(X,b.T), axis=1) + n*np.sum(b*X, axis=1) - np.sum(b*X)

            y_hat = 2/(2+n*rho) * y +  rho/(2+n*rho)/2 * temp1  - rho/(2+n*rho)/2 * temp2
            z = -1/(2+n*rho)*y + 1/(2*n)/(2+n*rho)* temp1 + (1+n*rho)/(2*n)/(2+n*rho)* temp2
            
            #   a update
            a = p - eta
            a += np.sum(alpha + s + y_hat.reshape(-1,1) - y_hat.reshape(1,-1) +\
             z.reshape(-1,1) - z.reshape(1,-1), axis=1).reshape(-1,1)*X -\
             np.dot(alpha + s + y_hat.reshape(-1,1) - y_hat.reshape(1,-1) +\
             z.reshape(-1,1) - z.reshape(1,-1), X)
            a = np.matmul(Sigma_i,a.reshape(n,dim,1)).reshape(n,dim)
            
            #   b update
            b = q - zeta
            b += np.sum(beta + t + z.reshape(-1,1) - z.reshape(1,-1), axis=1).reshape(-1,1)*X - \
            np.dot(beta + t + z.reshape(-1,1) - z.reshape(1,-1), X)
            b = np.matmul(Sigma_i,b.reshape(n,dim,1)).reshape(n,dim)

            #   p updates
            temp1 = 1/2* (a + eta)
            temp2 = 1/2*(L - u - gamma - np.abs(q))
            p = np.sign(temp1)*np.maximum(np.abs(temp1)+temp2, 0)

            #   q updates
            temp1 = 1/2* (b + zeta)
            temp2 = 1/2*(L - u - gamma - np.abs(p))
            q = np.sign(temp1)*np.maximum(np.abs(temp1)+temp2, 0)

            #   L update
            L = -1/(n*rho)* self.lanbda
            L +=  1/n*np.sum( gamma  + u + np.abs(p) + np.abs(q), axis=0).reshape(1,-1)
            
            #   slack updates
            #   s &t update
            s = - alpha -y_hat.reshape(-1,1) + y_hat.reshape(1,-1) - z.reshape(-1,1) + z.reshape(1,-1) + np.sum(a*X,axis=1).reshape(-1,1) - np.dot(a, X.T)
            s = np.maximum(s ,0)
            t = -beta - z.reshape(-1,1) + z.reshape(1,-1) + np.sum(b*X,axis=1).reshape(-1,1) - np.dot(b, X.T)
            t = np.maximum(t ,0) 

            #   u update
            u = -gamma + L - np.abs(q) - np.abs(p)
            u = np.maximum(u, 0)

            #   dual updates
            alpha +=  s + y_hat.reshape(-1,1) - y_hat.reshape(1,-1) + z.reshape(-1,1) - z.reshape(1,-1) - np.sum(a*X,axis=1).reshape(-1,1) + np.dot(a, X.T)
            beta +=  t + z.reshape(-1,1) - z.reshape(1,-1) - np.sum(b*X,axis=1).reshape(-1,1) + np.dot(b, X.T)
            gamma += u - L + np.abs(p) + np.abs(q)
            eta += a - p
            zeta += b - q

        y_hat = y_hat + z - np.sum(a*X, axis = 1)
        z = z - np.sum(b*X, axis = 1)

        self.y_hat = y_hat
        self.z = z
        self.a = a
        self.b = b
    
    def predict(self, X):
        pred =  np.max(self.y_hat.reshape(1,-1) + np.matmul(X,self.a.T), axis=1) - np.max(self.z.reshape(1,-1) + np.matmul(X,self.b.T), axis=1)

        return pred

    def cross_validate(self, X, y, n_folds):

        n, _ = X.shape

        # Permute the rows of X and y
        rp = np.random.permutation(n)
        y = y[rp]
        X = X[rp]

        # Initializing different measure
        loss = np.zeros(n_folds)

        for i in range(n_folds):
            
            # splitting the data to test and train
            test_start = int(np.ceil(n/n_folds * i))
            test_end = int(np.ceil(n/n_folds * (i+1)))

            I_test = [i for i in range(test_start, test_end)]
            I_train = [i for i in range(test_start)] + [i for i in range(test_end, n)] 
            
            # learning with the x_train and predicting with it
            self.fit(X[I_train], y[I_train], self.lanbda)
            
            y_hat_test = self.predict(X[I_test])
            loss[i] = np.mean((y_hat_test-y[I_test])**2)

        return np.mean(loss)
    


