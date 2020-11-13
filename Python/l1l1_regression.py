import numpy as np
import numpy.linalg as linalg
from tqdm import tqdm

class l1l1_regression:

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
