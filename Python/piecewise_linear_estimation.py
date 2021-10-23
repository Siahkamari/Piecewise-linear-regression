import torch as th
import torch.linalg as linalg
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
# from metric_learning_objectives import objectives

global zero

class tuner:
  def __init__(self):
    self.lanbda = None
    self.n_iter = None
    self.rho = 0.01
    self.score_val = None
    self.L = None
    self.sensitivity = 1e-3 # Hyper-parameter search stops if progress is less than this.
    self.n_folds = 5
    self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    global zero
    zero = th.zeros(1,device=self.device)
  
  def fit(self, X, y, lanbda = 'auto', n_iter = 'auto', rho = 0.01):
    self.rho = rho
    X = X.clone().detach().to(self.device).double()
    y = y.clone().detach().to(self.device).double()

    if n_iter != 'auto':
      self.n_iter = n_iter

    score_val = None

    if lanbda == 'auto':
      score_val, n_iter, lanbda = self.auto_tune(X, y)
    elif n_iter == 'auto':
      score_val, n_iter, _ = self.auto_tune(X, y, lanbda=lanbda)
    
    if self.n_iter == None:
      self.n_iter = int(self.n_folds/(self.n_folds-1)*n_iter)

    if score_val:
      self.score_val = score_val

    self.lanbda = lanbda
    self.fit_core(X, y)
  
  def auto_tune(self, X, y, lanbda = 'auto', max_hyper_iter = 5):
    if self.lanbda !=None:
      lanbda = self.lanbda
    if lanbda != 'auto':
      self.lanbda = lanbda
      score_val, n_iter = self.cross_validate(X, y)
      return score_val, n_iter, lanbda
  
    # m is the number of supervsions
    if self.m == 'n':
      m = th.sqrt(th.tensor(y.shape))
    elif self.m == 'n^2':
      m = th.tensor(y.shape)

    lanbdas = th.multiply(m,th.tensor([1e3,1e2,1e1,1,1e-1,1e-2,1e-3]))
    opt_grade = 1

    status = False
    score_val_old = th.tensor(float('-inf'))
    for _ in range(max_hyper_iter):
      i = 0
      score_vals = th.zeros(len(lanbdas), device=self.device)
      n_iters = th.zeros(len(lanbdas), device=self.device)
      
      for lanbda in tqdm(lanbdas, desc='Search for lanbda'):
      # for lanbda in lanbdas:
        self.lanbda = lanbda
        score_vals[i], n_iters[i] = self.cross_validate(X, y)
        i += 1

      arg_max = th.argmax(score_vals)
      score_val = score_vals[arg_max]
      lanbda = lanbdas[arg_max]
      n_iter = n_iters[arg_max]

      if score_val - score_val_old <= self.sensitivity:
        break
      else:
        score_val_old = th.clone(score_val)

      if opt_grade == 1:
        if  lanbda == lanbdas[0] and status != 'Last':
          lanbdas = [i*lanbda for i in [1e3,1e2,1e1,1] ]
          status = 'First'
        elif lanbda == lanbdas[-1] and status != 'First':
          lanbdas = [i*lanbda for i in [1,1e-1,1e-2,1e-3]]
          status = 'Last'
        else:
          lanbdas = [i*lanbda for i in [4,2,1,.5,.25]]
          opt_grade = 2
      else:    
        return score_val, int(n_iter), lanbda 
    return score_val, int(n_iter), lanbda 

  def cross_validate(self, X, y):
    n_folds = self.n_folds
    n, _ = X.shape

    # Permute the rows of X and y
    rng = th.Generator(device = self.device).manual_seed(0)
    rp = th.randperm(n, generator=rng, device=self.device)

    # Initializing different measure
    score_val = th.zeros(n_folds, device=self.device)
    score_train = th.zeros(n_folds, device=self.device)
    n_iter = th.zeros(n_folds, device=self.device)
    L = th.zeros(n_folds, device=self.device)

    # for i in tqdm(range(n_folds), desc= 'Cross Validation', leave=False):
    for i in range(n_folds):
      # splitting the data to test and train
      val_start = int(th.ceil(th.tensor(n/n_folds * i)))
      val_end = int(th.ceil(th.tensor(n/n_folds * (i+1))))

      I_val = [i for i in range(val_start, val_end)]
      I_train = [i for i in range(val_start)] + [i for i in range(val_end, n)] 
      I_val = rp[I_val]
      I_train = rp[I_train]
      
      # learning with the x_train and predicting with it
      n_iter[i] = self.fit_core(X[I_train], y[I_train], X_val = X[I_val], y_val = y[I_val])

      score_val[i] = self.score(X[I_val], y[I_val], X[I_train], y[I_train])
      score_train[i] = self.score(X[I_train], y[I_train])
      L[i] = th.sum(self.L)

    print("lanbda = ", "{:.2e}".format(float(self.lanbda)), 
    ", n_iter =", int(th.mean(n_iter)),
    ", training score = ", "{:.3f}".format(th.mean(score_train)),
    ", validation score = ", "{:.3f}".format(th.mean(score_val)),
    # ", L = ","{:.3e}".format(th.mean(L)),
    )

    return th.mean(score_val), int(th.mean(n_iter))
  
class convex_regression(tuner):
  def __init__(self):
    super().__init__()
    self.y_bar = 0
    self.X_bar = 0
    self.y_hat = 0
    self.a = 0
    self.m = 'n'
    self.task = 'R^2'

  def convex_params(self, dtype, n, dim):
    device = self.device
    y_hat = th.zeros(n, device=device, dtype=dtype)   
    a = th.zeros([n, dim], device=device, dtype=dtype) 
    p = th.zeros([n, dim], device=device, dtype=dtype)
    L = th.zeros([1, dim], device=device, dtype=dtype) 
    # slack
    s = th.zeros([n, n], device=device, dtype=dtype)
    u = th.zeros([n, dim], device=device, dtype=dtype)
    # dual
    alpha = th.zeros([n, n], device=device, dtype=dtype)
    gamma = th.zeros([n, dim], device=device, dtype=dtype)
    eta = th.zeros([n, dim], device=device, dtype=dtype)
    return y_hat, a, p, L, s, u, alpha, gamma, eta

  def fit_core(self, X, y, X_val = None, y_val = None):
    n, dim = X.shape
    n_iter = n if self.n_iter == None else self.n_iter    
    rho = self.rho
      
    # initial values
    dtype = X.dtype
    y_hat, a, p, L, s, u, alpha, gamma, eta = self.convex_params(dtype, n, dim)
    
    # preprocess1
    self.X_bar = th.mean(X, dim = 0)
    self.y_bar = th.mean(y)
    X -= self.X_bar
    y -= self.y_bar 

    Sigma = self.compute_Sigma(X)
    D = self.compute_D(X, Sigma)

    h = 2*(1+n*rho)/rho -\
      n**2*th.matmul(X.reshape(n,1,dim),th.matmul(Sigma, X.reshape(n,dim,1))).reshape(-1)

    Omega = th.linalg.inv(th.diag(h)  - n*D)

    # ADMM iteration
    self.y_hat = y_hat - th.sum(a*X, dim = 1) 
    self.a = th.clone(a)
    self.L = th.clone(L)

    R2_old = self.score(X_val, y_val) if (X_val != None) else self.score(X+self.X_bar, y+self.y_bar)

    mul_T = 1
    while True:
      message = 'ADMM iterations, R^2 = ' + "{:.3f}".format(R2_old.cpu().numpy())
      # for _ in tqdm(range(n_iter), desc=message, leave = False):
      for _ in range(n_iter):
        #   1st block primals (y_hat, a)
        theta = self.theta_update(X, p, eta, alpha, s)
        beta = self.beta_update(alpha, s)
        nu = self.nu_update(X, Sigma, theta)
        y_hat = Omega@(2*y/rho - beta + nu)
        a = self.a_update(X, y_hat, theta, Sigma)

        #   2nd block primals (L, u, p, s)
        L = self.L_update(gamma, th.abs(eta+a), self.lanbda/rho)  
        u = self.u_update(L, gamma, a, eta)
        p = self.p_update(a, eta, L, u, gamma)
        s = self.s_update(X, alpha, y_hat, a)

        #   dual updates
        alpha += self.alpha_update(X, s, y_hat, a)
        gamma += self.gamma_update(u, L, p)
        eta += self.eta_update(a, p)

      #   new checkpoint
      y_hat_old = th.clone(self.y_hat)
      a_old = th.clone(self.a)
      L_old = th.clone(self.L)
      self.y_hat = y_hat - th.sum(a*X, dim = 1) 
      self.a = th.clone(a)
      self.L = th.clone(L)
      R2 = self.score(X_val, y_val) if (X_val != None) else self.score(X+self.X_bar, y+self.y_bar)
      #  enough training or not?
      if self.n_iter != None:
        break
      elif R2 - R2_old <= self.sensitivity:
        if R2 - R2_old < 0:
          self.y_hat = y_hat_old
          self.a = a_old
          self.L = L_old
          mul_T -= 1
        break
      else:
        R2_old = th.clone(R2)
        mul_T += 1

    X += self.X_bar
    y += self.y_bar 
    return mul_T*n_iter
  
  def compute_Sigma(self, X):
    n, dim = X.shape
    XjXj = th.matmul(X.T,X) + th.eye(dim, device=self.device)
    Sigma= linalg.inv(n*th.matmul(X.reshape(n,dim,1), X.reshape(n,1,dim))+ XjXj ) 
    return Sigma 
  
  def compute_D(self, X, Sigma):
    n, dim = X.shape
    Sigma_bar = th.mean(Sigma, dim=0)
    x_Sigma_bar = th.mean(th.matmul(X.reshape(n,1,dim),Sigma).reshape(n,dim),dim=0)

    D = -(x_Sigma_bar@X.T).reshape(1,-1)  +\
        -th.matmul(X.reshape(n,1,dim),th.matmul(Sigma, X.reshape(n,dim,1))).reshape(1,-1)+\
            (th.matmul(X.reshape(n,1,dim),Sigma)).reshape(n,dim)@(X.T) +\
                X@((th.matmul(X.reshape(n,1,dim),Sigma)).reshape(n,dim)).T +\
                X@Sigma_bar@X.T    
    return D  

  def theta_update(self, X, p, eta, alpha, s):
    return p - eta + th.sum(alpha + s , dim=1).reshape(-1,1)*X - th.matmul(alpha + s , X)
  
  def beta_update(self, alpha, s):
    return th.sum(alpha -  alpha.T + s - s.T, dim=1) 

  def nu_update(self, X, Sigma, theta):
    n, dim = X.shape
    Sigma_theta_bar = th.mean(th.matmul(Sigma, theta.reshape(n,dim,1)).reshape(n,dim),dim=0)
    x_Sigma_theta_bar = th.mean(th.matmul(X.reshape(n,1,dim),th.matmul(Sigma,theta.reshape(n,dim,1))))

    nu = n*th.sum(X* th.matmul(Sigma, theta.reshape(n,dim,1)).reshape(n,dim),dim=1) +\
        n*th.matmul(X,Sigma_theta_bar) -n*x_Sigma_theta_bar
    return nu

  def a_update(self, X, y_hat, theta, Sigma):
    n, dim = X.shape
    a = theta + n*y_hat.reshape(-1,1) * X + th.matmul(y_hat.reshape(1,-1), X)
    return th.matmul(Sigma,a.reshape(n,dim,1)).reshape(n,dim)
  
  def L_update(self, gamma, tau, rhorho):
    knots = th.cat((gamma+ tau,gamma-tau), dim=0)
    knots = th.sort(knots, descending=True, dim=0)[0]
    n, dim = gamma.shape
    
    L = th.zeros(2*n, dim, device=self.device)
    ff = th.zeros(2*n, dim, device=self.device)
    fprim = 1/2* th.arange(1,2*n, device=self.device).reshape(-1,1)
    ff[:-1] = fprim*(knots[1:] - knots[:-1])
    ff[:-1] = rhorho + th.cumsum(ff[:-1], dim=0)
    L[:-1] = knots[1:] - ff[:-1]/fprim
    L[-1] =  knots[-1] - ff[-2]/n

    I = th.argmax((ff<=0).long(), dim=0)
    Ls = th.gather(L, dim=0, index=I.view(1,-1))
    return Ls

  def u_update(self, L, gamma, a, eta):
    return th.maximum(L - gamma  - th.abs(a + eta), zero)
  
  def p_update(self, a, eta, L, u, gamma):
    temp1 = a + eta
    temp2 = L - u - gamma 
    return 1/2*th.sign(temp1)*th.maximum(th.abs(temp1)+temp2, zero)
  
  def s_update(self, X, alpha, y_hat, a):
    s = -alpha -y_hat.reshape(-1,1) + y_hat.reshape(1,-1) + th.sum(a*X,dim=1).reshape(-1,1) - th.matmul(a, X.T)
    return th.maximum(s ,zero)

  def alpha_update(self, X, s, y_hat, a):
     return s + y_hat.reshape(-1,1) - y_hat.reshape(1,-1) - th.sum(a*X,dim=1).reshape(-1,1) + th.matmul(a, X.T)
    
  def gamma_update(self, u, L, p):
    return u - L + th.abs(p) 

  def eta_update(self, a, p):
    return a - p

  def predict(self, X):
    X = X.clone().detach().to(self.device).double()
    return th.max(self.y_hat.reshape(1,-1) + th.matmul(X - self.X_bar, self.a.T), dim=1)[0] + self.y_bar
  
  def score(self, X, y, X_val = None, y_val=None, task=None):
    y = y.clone().detach().to(self.device).double()
    return 1.0 - th.mean((self.predict(X)-y)**2)/th.var(y)


class dc_regression(convex_regression):
  def __init__(self):
    tuner.__init__(self)
    self.y_bar = 0
    self.X_bar = 0
    self.y_hat_1 = 0
    self.y_hat_2 = 0
    self.a_1 = 0
    self.a_2 = 0
    self.m = 'n'
    self.task = 'R^2'
    self.L1 = 0
    self.L2 = 0

  def fit_core(self, X, y, X_val = None, y_val = None):
    n, dim = X.shape
    n_iter = n if self.n_iter == None else self.n_iter
    rho = self.rho

    # initial values
    dtype = X.dtype
    # primal
    y_hat_1, a_1, p_1, L_1, s_1, u_1, alpha_1, gamma_1, eta_1 = self.convex_params(dtype, n, dim)
    y_hat_2, a_2, p_2, L_2, s_2, u_2, alpha_2, gamma_2, eta_2 = self.convex_params(dtype, n, dim)

    # preprocess1
    self.X_bar = th.mean(X, dim = 0)
    self.y_bar = th.mean(y)
    X -= self.X_bar
    y -= self.y_bar 
    
    Sigma = self.compute_Sigma(X)
    D = self.compute_D(X, Sigma)

    h_1 = 2*n - n**2*th.matmul(X.reshape(n,1,dim),th.matmul(Sigma, X.reshape(n,dim,1))).reshape(-1)
    h_2 = (4+2*n*rho)/rho - n**2*th.matmul(X.reshape(n,1,dim),th.matmul(Sigma, X.reshape(n,dim,1))).reshape(-1)

    Omega_1 = th.linalg.inv(th.diag(h_1) - n*D)
    Omega_2 = th.linalg.inv(th.diag(h_2) - n*D)

    # ADMM iteration
    self.y_hat_1 = y_hat_1 - th.sum(a_1*X, dim = 1) 
    self.y_hat_2 = y_hat_2 - th.sum(a_2*X, dim = 1) 
    self.a_1 = th.clone(a_1)
    self.a_2 = th.clone(a_2)
    self.L1 = th.clone(L_1)
    self.L2 = th.clone(L_2)

    R2_old = self.score(X_val, y_val) if (X_val != None) else self.score(X+self.X_bar, y+self.y_bar)
    mul_T = 1
    while True:
      message = 'ADMM iterations, R^2 = ' + "{:.3f}".format(R2_old.cpu().numpy())
      for _ in tqdm(range(n_iter), desc=message, leave = False):
      # for _ in range(n_iter):
        #   1st block
        theta_1 = self.theta_update(X, p_1, eta_1, alpha_1, s_1)
        theta_2 = self.theta_update(X, p_2, eta_2, alpha_2, s_2)
        beta_1 = self.beta_update(alpha_1, s_1)
        beta_2 = self.beta_update(alpha_2, s_2)
        nu_1 = self.nu_update(X, Sigma, theta_1)
        nu_2 = self.nu_update(X, Sigma, theta_2)

        y_hat_1 = 1/2*Omega_1@(nu_1 + nu_2 - beta_1 - beta_2) +\
              1/2*Omega_2@(4*y/rho + nu_1 - nu_2 - beta_1+beta_2)
        
        y_hat_2 = 1/2*Omega_1@(nu_1 + nu_2 - beta_1 - beta_2) -\
              1/2*Omega_2@(4*y/rho + nu_1 - nu_2 - beta_1+beta_2)

        a_1 = self.a_update(X, y_hat_1, theta_1, Sigma)
        a_2 = self.a_update(X, y_hat_2, theta_2, Sigma)

        #   2nd block
        L_1 = self.L_update(gamma_1, th.abs(eta_1+a_1), self.lanbda/rho)
        L_2 = self.L_update(gamma_2, th.abs(eta_2+a_2), self.lanbda/rho)
        u_1 = self.u_update(L_1, gamma_1, a_1, eta_1)
        u_2 = self.u_update(L_2, gamma_2, a_2, eta_2)
        p_1 = self.p_update(a_1, eta_1, L_1, u_1, gamma_1)
        p_2 = self.p_update(a_2, eta_2, L_2, u_2, gamma_2)
        s_1 = self.s_update(X, alpha_1, y_hat_1, a_1)
        s_2 = self.s_update(X, alpha_2, y_hat_2, a_2)

        #   dual updates
        alpha_1 += self.alpha_update(X, s_1, y_hat_1, a_1)
        alpha_2 += self.alpha_update(X, s_2, y_hat_2, a_2)
        gamma_1 += self.gamma_update(u_1, L_1, p_1)
        gamma_2 += self.gamma_update(u_2, L_2, p_2)
        eta_1 += self.eta_update(a_1, p_1)
        eta_2 += self.eta_update(a_2, p_2)

      #   new checkpoint
      y_hat_1_old = th.clone(self.y_hat_1)
      y_hat_2_old = th.clone(self.y_hat_2)
      a_1_old = th.clone(self.a_1)
      a_2_old = th.clone(self.a_2)
      L1_old = th.clone(self.L1)
      L2_old = th.clone(self.L2)
      self.y_hat_1 = y_hat_1 - th.sum(a_1*X, dim = 1) 
      self.y_hat_2 = y_hat_2 - th.sum(a_2*X, dim = 1) 
      self.a_1 = th.clone(a_1)
      self.a_2 = th.clone(a_2)
      self.L1 = th.clone(L_1)
      self.L2 = th.clone(L_2)
      R2 = self.score(X_val, y_val) if (X_val != None) else self.score(X+self.X_bar, y+self.y_bar)
      #   enough training or not?
      if self.n_iter != None:
        break
      elif R2 - R2_old <= self.sensitivity:
        if R2 - R2_old < 0:
          self.y_hat_1 = y_hat_1_old
          self.y_hat_2 = y_hat_2_old
          self.a_1 = a_1_old
          self.a_2 = a_2_old
          self.L1 = L1_old
          self.L2 = L2_old
          mul_T -= 1
        break
      else:
        R2_old = th.clone(R2)
        mul_T += 1

    self.L = self.L1 + self.L2
    X += self.X_bar
    y += self.y_bar 
    return mul_T*n_iter

  def predict(self, X):
    X = X.clone().detach().to(self.device).double()
    out = self.y_bar + \
        th.max(self.y_hat_1.reshape(1,-1) + th.matmul(X - self.X_bar, self.a_1.T), dim=1)[0] +\
        - th.max(self.y_hat_2.reshape(1,-1) + th.matmul(X - self.X_bar, self.a_2.T), dim=1)[0]
    return  out

class PBDL(convex_regression):
  def __init__(self):
    tuner.__init__(self)
    self.X_bar = 0
    self.z = 0
    self.a = 0
    self.m = 'n^2'
    self.task = 'pairwise'

  def fit_core(self, X, y, X_val = None, y_val = None):
    n, dim = X.shape
    n_iter = n if self.n_iter == None else self.n_iter 
    rho = self.rho
      
    # initial values
    dtype = X.dtype

    dtype = X.dtype
    z, a, p, L, s, u, alpha, gamma, eta = self.convex_params(dtype, n, dim)
    zeta = th.zeros([n, n], device=self.device, dtype=dtype)   # block 1
    t = th.zeros([n, n], device=self.device, dtype=dtype)   # block 2
    tau = th.zeros([n, n], device=self.device, dtype=dtype)
    
    # preprocess1
    iota = 2*(y.reshape(-1,1) == y.reshape(1,-1)).float() -1

    self.X_bar = th.mean(X, dim = 0)
    X -= self.X_bar
    Sigma = self.compute_Sigma(X)
    D = self.compute_D(X, Sigma)

    h = 2*n -\
      n**2*th.matmul(X.reshape(n,1,dim),th.matmul(Sigma, X.reshape(n,dim,1))).reshape(-1)

    Omega = th.linalg.inv(th.diag(h)  - n*D)

    # ADMM iteration
    self.z = z - th.sum(a*X, dim = 1) 
    self.a = th.clone(a)
    self.L = th.clone(L)
    
    if X_val != None:
      score_old = self.score(X_val, y_val, X+self.X_bar, y)
    else:
      score_old = self.score(X+self.X_bar, y)

    mul_T = 1
    while True:
      # for _ in tqdm(range(n_iter), desc='ADMM iterations', leave = False):
      for _ in range(n_iter):
        #   1st block primals
        zeta = self.zeta_update(rho, tau, iota, s, t)
        theta = self.theta_update(X, p, eta, alpha, s)
        beta = self.beta_update(alpha, s)
        nu = self.nu_update(X, Sigma, theta)
        z = Omega@(nu - beta)
        a = self.a_update(X, z, theta, Sigma)

        #   2nd block primals
        L = self.L_update(gamma, th.abs(eta+a), self.lanbda/rho)
        u = self.u_update(L, gamma, a, eta)
        p = self.p_update(a, eta, L, u, gamma)
        s = self.s_update(tau, iota, zeta, alpha, z, a, X)
        t = self.t_update(tau, iota, zeta, s)

        #   dual updates
        alpha += self.alpha_update(X, s, z, a)
        tau += self.tau_update(iota, s, t, zeta)
        gamma += self.gamma_update(u, L, p)
        eta += self.eta_update(a, p)

      #   new checkpoint
      z_old = th.clone(self.z)
      a_old = th.clone(self.a)
      L_old = th.clone(self.L)
      self.z = z - th.sum(a*X, dim = 1) 
      self.a = th.clone(a)
      self.L = th.clone(L)
      
      if X_val != None:
        score_new = self.score(X_val, y_val, X+self.X_bar, y)
      else:
        score_new = self.score(X+self.X_bar, y)

      #   enough training or not?
      if self.n_iter != None:
        break
      elif score_new - score_old <= self.sensitivity:
        if score_new - score_old < 0:
          self.z = z_old
          self.a = a_old
          self.L = L_old
          mul_T -= 1
        break
      else:
        score_old = th.clone(score_new)
        mul_T += 1

    X += self.X_bar
    return mul_T*n_iter

  def zeta_update(self, rho, tau, iota, s, t):
    return th.maximum(-1/rho + tau + iota*s - iota + t + 1, zero)
  
  def s_update(self,tau, iota, zeta, alpha, z, a, X):
    pi1 = -tau + iota - 1 + zeta
    pi2 = -alpha -z.reshape(-1,1) + z.reshape(1,-1) + th.sum(a*X,dim=1).reshape(-1,1) - th.matmul(a, X.T)
    s = 1/2*(pi2 + iota*pi1 - iota*th.maximum(pi1-iota*pi2,th.zeros(1,device=self.device)))
    return th.maximum(s ,zero)

  def t_update(self, tau, iota, zeta, s):
    pi1 = -tau + iota - 1 + zeta
    return th.maximum(pi1 - iota*s ,zero)
  
  def tau_update(self, iota, s, t, zeta):
    return iota*s - iota + t + 1 - zeta

  def phi(self, X):
    X = X.clone().detach().to(self.device).double()
    return th.max(self.z.reshape(1,-1) + th.matmul(X - self.X_bar, self.a.T), dim=1)[0]
  
  def bregman_div(self, X1, X2):
    X1 = X1.clone().detach().to(self.device).double()
    X2 = X2.clone().detach().to(self.device).double()
    i1 = th.argmax(self.z.reshape(1,-1) + th.matmul(X1 - self.X_bar, self.a.T), dim=1)
    i2 = th.argmax(self.z.reshape(1,-1) + th.matmul(X2 - self.X_bar, self.a.T), dim=1)
    
     # acts one vs all or entry-wise
    return self.z[i1].reshape(1,-1) - self.z[i2].reshape(1,-1) + \
      th.sum((X1 - self.X_bar)*(self.a[i1] - self.a[i2]), dim=1)

  def is_similar(self, X_que, X_pool):
    divs = self.bregman_div(X_que, X_pool)[0]
    return divs < th.ones(1, device=self.device)

  def rank(self, X_que, X_pool):
    divs = self.bregman_div(X_que, X_pool)[0]
    return th.argsort(divs)

  def classify(self, X_que, y_pool, X_pool, k=5):
    n_que = X_que.shape[0]
    y_pred = th.zeros(n_que, device=self.device, dtype=th.int)
    for i in range(n_que):
      ranks = self.rank(X_que[i], X_pool)
      y_pred[i] = th.mode(y_pool[ranks[0:k]])[0].int()
    return y_pred

  def score(self, X_que, y_que, X_pool=None, y_pool=None, task=None):
    if X_pool == None:
      X_pool = X_que
      y_pool = y_que

    y_que = y_que.clone().detach().to(self.device).double()
    y_pool = y_pool.clone().detach().to(self.device).double()

    objective = objectives(rank=self.rank,
     classify=lambda X_que: self.classify(X_que, y_pool, X_pool, k=5),
     is_similar=self.is_similar)

    if (task == None) or (task == 'pairwise'):
      return objective.pairwise_score(y_que, X_que)
    elif task == 'map':
      return objective.mean_average_precision(y_que, X_que, y_pool, X_pool)
    elif task == 'auc':
      return objective.area_under_the_curve(y_que, X_que, y_pool, X_pool)
    elif task == 'knn':
      return objective.accuracy(y_que, X_que)
    else:
      raise("score functions chosen not recognized")
