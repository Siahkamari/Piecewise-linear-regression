function f_hat = dc_fit(y, X, lambda)
% learns a DC functions given data and regularization constant

%% inputs:
% labels                    y : n x 1
% features matrix           X : n x d
% regularization constant   lambda > 0

%% outputs
% learned dc function       f_hat : R ^(n x d) -> R ^(n)

%% initialization
[n, d] = size(X);
if nargin == 2
    lambda = 0;
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
I = zeros(n*d,1);
J = zeros(n*d,1);
V = ones(n*d,1);
count = 0;
for i = 1:n
    for dd=1:d
        count = count+1;
        I(count) = i;
        J(count) = (i-1)*d + dd;
    end
end
A_norm = sparse(I,J,V, n, n*d);

%% loss constraints
A_loss = sparse(1:n,1:n,ones(n,1), n, n);

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
% A1  :  yhat1_i - y_hat1_j - (a^+_i - a^-_i)_^T (x_i - x_j) < 0
% A2  :  yhat2_i - y_hat2_j - (b^+_i - b^-_i)_^T (x_i - x_j) < 0
% A3  :  ||a^+_i|| + ||a^-_i|| + ||b^+_i|| + ||b^-_i|| - L < 0
% A4  :  yhat1_i - y_hat1_j  - eps_i < y_i && -yhat1_i + y_hat1_j  - eps_i < -y_i

b1 = sparse(n*(n-1),1);
b2 = sparse(n*(n-1),1);
b3 = sparse(n,1);

A1 = [sparse(n*(n-1),n), A_v, sparse(n*(n-1),n), A_g, -A_g, sparse(n*(n-1),2*n*d+1)];
A2 = [sparse(n*(n-1),2*n), A_v, sparse(n*(n-1),2*n*d), A_g, -A_g, sparse(n*(n-1),1)];
A3 = [sparse(n,3*n),A_norm,A_norm,A_norm,A_norm, -ones(n,1)];

A4 = [-A_loss, A_loss, - A_loss, sparse(n, 4*n*d+1);
    -A_loss, -A_loss, A_loss, sparse(n, 4*n*d+1)];
b4 = [y; -y];

%% cost vector;
c = zeros(1, 3*n + 4*n*d + 1);
c(1:n) = 1;
c(end) = lambda;

clearvars A_v A_g A_norm I J V

%% solving
try
    model.A = -[A1;A2;A3;A4];
    model.obj = c;
    model.rhs = full(-[b1;b2;b3;b4]);
    model.sense = '>';
    
    params.Threads = 16;
    params.OutputFlag = 0;
    result = gurobi(model, params);
    
    z = result.x;
catch
%     warning('Gurobi is not installed/working. Instead using MATLAB linear program solvers.')
    options = optimoptions('linprog','Display','off');
    lb = zeros( 3*n + 4*n*d + 1, 1);
    z = linprog(c, [A1;A2;A3;A4],[b1;b2;b3;b4], [],[], lb, [], options);
end

%% getting the parameters
params1.phi = z(n+1 : 2*n);
params2.phi = z(2*n+1 : 3*n);
shift = 3*n;

params1.grad = reshape( z(shift + 1 : shift + n*d) - z(shift + n*d + 1 :shift + 2*n*d), [d,n])' ;
params2.grad = reshape( z(shift + 2*n*d + 1 :shift + 3* n*d) - z(shift + 3*n*d + 1 : shift + 4*n*d), [d,n])' ;
params1.phi = params1.phi - dot(params1.grad,X,2);
params2.phi = params2.phi - dot(params2.grad,X,2);

f_hat = @(X) dc_function(X, params1, params2);
