from tqdm import tqdm
import torch as th

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
            loss = th.zeros(len(lanbdas))
            for lanbda in tqdm(lanbdas):
                self.lanbda = lanbda
                loss[i] = self.cross_validate(X, y, n_folds)
                i += 1

            arg_min = th.argmin(loss)
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

        n = X.shape[0]
        dim = X.shape[1]
        if T == False:
            T = 2*n
        rho = 0.01

        if th.cuda.is_available():
            device = th.device("cuda")          # a CUDA device object
        else:
            device = th.device("cpu")               

        # initial values
        # primal
        y_hat = th.zeros(n, device=device)
        z = th.zeros(n, device=device)
        a = th.zeros(n, dim, device=device)
        b = th.zeros(n, dim, device=device)
        p = th.zeros(n, dim, device=device)
        q = th.zeros(n, dim, device=device)

        L = th.zeros(1, dim, device=device)

        # slack
        s = th.zeros(n, n, device=device)
        t = th.zeros(n, n, device=device)
        u = th.zeros(n, dim, device=device)

        # dual
        alpha = th.zeros(n, n, device=device)
        beta = th.zeros(n, n, device=device)
        gamma = th.zeros(n, dim, device=device)
        eta = th.zeros(n, dim, device=device)
        zeta = th.zeros(n, dim, device=device)

        # preprocess1
        XjXj = th.matmul(X.T,X)
        Xbar = th.mean(X, dim = 0)
        ybar = th.mean(y)
        Sigma_i = th.zeros([n,dim,dim], device=device)

        for i in range(n):
            Sigma_i[i,:,:] = n*th.outer(X[i],X[i]) + XjXj - n*(th.outer(Xbar,X[i]) + th.outer(X[i],Xbar))
            Sigma_i[i,:,:] = th.inverse(Sigma_i[i,:,:] + th.eye(dim, device=device))

        # ADMM iteration
        for _ in range(T):
    
            #   primal updates
            #   y_hat & z update
            temp1 = th.sum(alpha.T -  alpha + s.T - s -  th.matmul(a,X.T) + th.matmul(X,a.T), dim=1) + n*2*ybar + n*th.sum(a*X,dim=1) - th.sum(a*X)
            temp2 = th.sum(beta.T - beta + t.T - t - th.matmul(b,X.T) + th.matmul(X,b.T), dim=1) + n*th.sum(b*X, dim=1) - th.sum(b*X)

            y_hat = 2/(2+n*rho) * y +  rho/(2+n*rho)/2 * temp1  - rho/(2+n*rho)/2 * temp2
            z = -1/(2+n*rho)*y + 1/(2*n)/(2+n*rho)* temp1 + (1+n*rho)/(2*n)/(2+n*rho)* temp2
            
            #   a update
            a = p - eta
            a += th.sum(alpha + s + y_hat.reshape(-1,1) - y_hat.reshape(1,-1) +\
             z.reshape(-1,1) - z.reshape(1,-1), dim=1).reshape(-1,1)*X -\
             th.matmul(alpha + s + y_hat.reshape(-1,1) - y_hat.reshape(1,-1) +\
             z.reshape(-1,1) - z.reshape(1,-1), X)
            a = th.matmul(Sigma_i,a.reshape(n,dim,1)).reshape(n,dim)
            
            #   b update
            b = q - zeta
            b += th.sum(beta + t + z.reshape(-1,1) - z.reshape(1,-1), dim=1).reshape(-1,1)*X - \
            th.matmul(beta + t + z.reshape(-1,1) - z.reshape(1,-1), X)
            b = th.matmul(Sigma_i,b.reshape(n,dim,1)).reshape(n,dim)

            #   p updates
            temp1 = 1/2* (a + eta)
            temp2 = 1/2*(L - u - gamma - th.abs(p))
            p = th.sign(temp1)*th.maximum(th.abs(temp1)+temp2, th.zeros(1, device=device))

            #   q updates
            temp1 = 1/2* (b + zeta)
            temp2 = 1/2*(L - u - gamma - th.abs(q))
            q = th.sign(temp1)*th.maximum(th.abs(temp1)+temp2, th.zeros(1,device=device))

            #   L update
            L = -1/(n*rho)* self.lanbda
            L +=  1/n*th.sum( gamma  + u + th.abs(p) + th.abs(q), dim=0).reshape(1,-1)
            
            #   slack updates
            #   s &t update
            s = - alpha -y_hat.reshape(-1,1) + y_hat.reshape(1,-1) - z.reshape(-1,1) + z.reshape(1,-1) + th.sum(a*X,dim=1).reshape(-1,1) - th.matmul(a, X.T)
            s = th.maximum(s, th.zeros(1,device=device))
            t = -beta - z.reshape(-1,1) + z.reshape(1,-1) + th.sum(b*X,dim=1).reshape(-1,1) - th.matmul(b, X.T)
            t = th.maximum(t, th.zeros(1,device=device)) 

            #   u update
            u = -gamma + L - th.abs(q) - th.abs(p)
            u = th.maximum(u, th.zeros(1,device=device))

            #   dual updates
            alpha +=  s + y_hat.reshape(-1,1) - y_hat.reshape(1,-1) + z.reshape(-1,1) - z.reshape(1,-1) - th.sum(a*X,dim=1).reshape(-1,1) + th.matmul(a, X.T)
            beta +=  t + z.reshape(-1,1) - z.reshape(1,-1) - th.sum(b*X,dim=1).reshape(-1,1) + th.matmul(b, X.T)
            gamma += u - L + th.abs(p) + th.abs(q)
            eta += a - p
            zeta += b - q

        y_hat = y_hat + z - th.sum(a*X, dim = 1)
        z = z - th.sum(b*X, dim = 1)

        self.y_hat = y_hat
        self.z = z
        self.a = a
        self.b = b
    
    def predict(self, X):
        f1, _ =  th.max(self.y_hat.reshape(1,-1) + th.matmul(X,self.a.T), dim=1) 
        f2, _ = th.max(self.z.reshape(1,-1) + th.matmul(X,self.b.T), dim=1)
        pred = f1-f2

        return pred

    def cross_validate(self, X, y, n_folds):

        n, _ = X.shape

        # Permute the rows of X and y
        rp = th.randperm(n)
        y = y[rp]
        X = X[rp]

        # Initializing different measure
        loss = th.zeros(n_folds)

        for i in range(n_folds):
            
            # splitting the data to test and train
            test_start = int(th.ceil(th.tensor(n/n_folds * i)))
            test_end = int(th.ceil(th.tensor(n/n_folds * (i+1))))

            I_test = [i for i in range(test_start, test_end)]
            I_train = [i for i in range(test_start)] + [i for i in range(test_end, n)] 
            
            # learning with the x_train and predicting with it
            self.fit(X[I_train], y[I_train], self.lanbda)
            
            y_hat_test = self.predict(X[I_test])
            loss[i] = th.mean((y_hat_test-y[I_test])**2)

        return th.mean(loss)
    


