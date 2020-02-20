function [f_hat, params1] = c_fit(y, X, lambda)

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
% A1  :  y_i - y_j - (a^+_i - a^-_i)_^T (x_i - x_j) < 0
% A3  :  ||a^+_i|| + ||a^-_i|| + ||b^+_i|| + ||b^-_i|| - L < 0

A1 = [A_g, -A_g, sparse(n*(n-1),1)];
b1 = b_inter;
A3 = [A_norm,A_norm -ones(n,1)];
b3 = sparse(n,1);

%% cost vector;
c = zeros(1, 2*n*d + 1);
c(end) = lambda;


%% solving
model.A = -[A1;A3];
model.obj = c;
model.rhs = full(-[b1;b3]);
model.sense = '>';

result = gurobi(model);

z = result.x;

params1.phi = y;

params1.grad = reshape( z(1 : n*d) - z(n*d + 1 : 2*n*d), [d,n])' ;
params1.phi = params1.phi - dot(params1.grad,X,2);

params2_0.phi = 0*params1.phi;
params2_0.grad = 0*params1.grad;

f_hat = @(X) dc_function(X, params1, params2_0);

