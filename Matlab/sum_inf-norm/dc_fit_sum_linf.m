function [f_hat, L, params1, params2, phi_1, phi_2] = dc_fit_sum_linf(y, X, lambda, loss)

[n, d] = size(X);
if nargin == 2
    lambda = 0;
end
if nargin < 4
    loss = "L1";
    if length(unique(y)) == 2
        loss = "hinge";
        if sum(unique(y) == [0,1]) == 2
            y = 2*y - 1;
        end
    end
end

%% v part of constraints
I = zeros(n*(n-1)*2,1);
J = zeros(n*(n-1)*2,1);
V = zeros(n*(n-1)*2,1);

row = 0;
count = 0;
for i=1 : n
    for j=setdiff(1 : n, i)
        row = row + 1;
        count = count + 1;
        
        I(count) = row;
        J(count) = i;
        V(count) = 1;
        
        count = count + 1;
        I(count) = row;
        J(count) = j;
        V(count) = -1;
    end
end
A_v = sparse(I,J,V,n*(n-1), n);

%% g part of constraints
I = zeros(n*(n-1)*d,1);
J = zeros(n*(n-1)*d,1);
V = zeros(n*(n-1)*d,1);

row = 0;
count = 0;
for i=1 : n
    for j=setdiff(1 : n,i)
        row = row + 1;
        for dd=1:d
            count = count + 1;
            I(count) = row;
            J(count) = (i-1)*d + dd;
            V(count) = - X(i, dd) + X(j, dd);
        end
    end
end
A_g = sparse(I,J,V, n*(n-1), n*d);

%% norm constraints

I = 1:n*d;
J = 1:n*d;
V = ones(n*d,1);
A_norm = sparse(I,J,V, n*d, n*d);

I = 1:d;
J = 1:d;
V = ones(d,1);

A_norm_b = repmat(sparse(I,J,V,d,d),n,1);

%% loss constraints
A_loss = sparse(1:n,1:n,ones(n,1), n, n);
A_hing = sparse(1:n,1:n,-y, n, n);

%% interpolation constraint rhs
b_inter = zeros(n*(n-1),1);
count = 0;
for i=1 : n
    for j=setdiff(1 : n,i)
        count = count + 1;
        b_inter(count) = y(j) - y(i);
    end
end

%% building the constraints
% A1  :  yhat_i - y_hat_j + c_i - c_j - (a^+_i - a^-_i)_^T (x_i - x_j) < 0
% A2  :  c_i - c_j - (b^+_i - b^-_i)_^T (x_i - x_j) < 0
% A3  :  ||a^+_i|| + ||a^-_i|| + ||b^+_i|| + ||b^-_i|| - L < 0
% A4 L1  :  yhat_i  - eps_i < y_i && -yhat_i  - eps_i < -y_i
% A4 hinge  : - y_i yhat_i  - eps_i < -1

b1 = sparse(n*(n-1),1);
b2 = sparse(n*(n-1),1);
b3 = sparse(n*d,1);

if lambda > 0
    A1 = [sparse(n*(n-1),n), A_v, sparse(n*(n-1),n), A_g, -A_g, sparse(n*(n-1),2*n*d+d+1)];
    A2 = [sparse(n*(n-1),2*n), A_v, sparse(n*(n-1),2*n*d), A_g, -A_g, sparse(n*(n-1),d+1)];
    A3 = [sparse(n*d,3*n),A_norm,A_norm,A_norm,A_norm, -A_norm_b,sparse(n*d,1)];
    A32 =[sparse(1,3*n + 4*n*d),ones(1,d),-1];
    
    if  ( (loss == "L1") || (loss == "L2") )
        A4 = [-A_loss, A_loss, - A_loss, sparse(n, 4*n*d+d+1);
            -A_loss, -A_loss, A_loss, sparse(n, 4*n*d+d+1)];
        b4 = [y; -y];
    elseif loss == "hinge"
        A4 = [-A_loss, A_hing, -A_hing, sparse(n, 4*n*d+1)];
        b4 = -ones(n,1);
    end
    
elseif lambda == 0 && loss == "L1"
    A1 = [A_v, A_g, -A_g, sparse(n*(n-1),2*n*d+1)];
    b1 = b_inter;
    A2 = [A_v, sparse(n*(n-1),2*n*d), A_g, -A_g, sparse(n*(n-1),1)];
    A3 = [sparse(n,n),A_norm,A_norm,A_norm,A_norm, -ones(n,1)];
    A4 = []; b4 = [];
    
elseif lambda == 0 && loss == "hinge"
    A1 = [A_v, A_v, A_g, -A_g, sparse(n*(n-1),2*n*d+1)];
    A2 = [sparse(n*(n-1),n), A_v, sparse(n*(n-1),2*n*d), A_g, -A_g, sparse(n*(n-1),1)];
    A3 = [sparse(n,2*n),A_norm,A_norm,A_norm,A_norm, -ones(n,1)];
    A4 = [A_hing, sparse(n, n+ 4*n*d+1)];
    b4 = -ones(n,1);
end

%% cost vector;
if lambda > 0 && loss == "L2"
    c = zeros(1, 3*n + 4*n*d + d+1);
    c(end) = lambda;
    
elseif lambda > 0 && loss == "L1"
    c = zeros(1, 3*n + 4*n*d + d+1);
    c(1:n) = 1; 
    c(end) = lambda;
    
elseif lambda == 0 && loss == "L1"
    c = zeros(1, n + 4*n*d + 1);
    c(end) = 1;
    
elseif lambda == 0 && loss == "hinge"
    c = zeros(1, 2*n + 4*n*d + 1);
    c(end) = 1;
end
clearvars A_v A_g A_norm A_hing I J V

%% solving

if loss == "L2"
    AQ = [sparse(n,n), A_loss, -A_loss, sparse(n, 4*n*d+1)];
    model.Q = AQ'*AQ;
    model.A = -[A1;A2;A3;A32;A4];
    model.obj = c - 2*y'*AQ;
    model.rhs = full(-[b1;b2;b3;0;b4]);
    model.sense = '>';
    gurobi_write(model, 'qp.lp');
elseif loss == "L1"
    model.A = -[A1;A2;A3;A32;A4];
    model.obj = c;
    model.rhs = full(-[b1;b2;b3;0;b4]);
    model.sense = '>';
end
% params.OutputFlag = 0;
result = gurobi(model);

z = result.x;


if lambda > 0
    params1.phi = z(n+1 : 2*n);
    params2.phi = z(2*n+1 : 3*n);
    shift = 3*n;
    
elseif lambda == 0 && loss == "L1"
    params1.phi = z(1 : n) + y;
    params2.phi = z(1 : n);
    shift = n;
    
elseif lambda == 0 && loss == "hinge"
    params1.phi = z(1 : n) + z(n+1 : 2*n);
    params2.phi = z(n+1 : 2*n);
    shift = 2*n;
end

params1.grad = reshape( z(shift + 1 : shift + n*d) - z(shift + n*d + 1 :shift + 2*n*d), [d,n])' ;
params2.grad = reshape( z(shift + 2*n*d + 1 :shift + 3* n*d) - z(shift + 3*n*d + 1 : shift + 4*n*d), [d,n])' ;
params1.phi = params1.phi - dot(params1.grad,X,2);
params2.phi = params2.phi - dot(params2.grad,X,2);

L = z(end);

f_hat = @(X) dc_function(X, params1, params2);

params1_0.phi = 0*params1.phi;
params1_0.grad = 0*params1.grad;
params2_0.phi = 0*params2.phi;
params2_0.grad = 0*params2.grad;

phi_1 = @(X) dc_function(X, params1, params2_0);
phi_2 = @(X) dc_function(X, params2, params1_0);

