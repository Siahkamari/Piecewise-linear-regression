function f_hat = dc_fit_multi_class_sum_inf(y, X, lambda)

[n, d] = size(X);
k = max(y);

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
A_vivj = sparse(I,J,V,n*(n-1), n);

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
A_gxixj = sparse(I,J,V, n*(n-1), n*d);

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
Av = sparse(1:n,1:n,ones(n,1), n, n);

%% building the constraints

% A1  :  yhat1_i - y_hat1_j - (a^+_i - a^-_i)_^T (x_i - x_j) < 0
% A2  :  yhat2_i - y_hat2_j - (b^+_i - b^-_i)_^T (x_i - x_j) < 0
A1 = [A_vivj, sparse(n*(n-1),n)  , A_gxixj, -A_gxixj, sparse(n*(n-1),2*n*d), sparse(n*(n-1),n), sparse(n*(n-1),d)];
A2 = [sparse(n*(n-1),1*n), A_vivj, sparse(n*(n-1),2*n*d), A_gxixj, -A_gxixj, sparse(n*(n-1),n), sparse(n*(n-1),d)];
b1 = sparse(n*(n-1),1);
b2 = sparse(n*(n-1),1);

% A3  :  a^+_id+ a^-_id + b^+_id + b^-_id - Ld < 0

A3 = [sparse(n*d,2*n),A_norm,A_norm,A_norm,A_norm,sparse(n*d,n), -A_norm_b];
b3 = sparse(n*d,1);

% A3 = [sparse(n,2*n),A_norm,A_norm,A_norm,A_norm, sparse(n,n), -ones(n,1)];
% b3 = sparse(n,1);

% Wrapping it with zeros matrices for making it k functions
A_empty = sparse(n*(n-1), 2*n + 4*n*d + n + d);
A1_all = repmat(A_empty, k , k);
A2_all = repmat(A_empty, k , k);

for kk=1:k    
    A1_all = A1_all +  [repmat(A_empty, kk-1, k);                      
        repmat(A_empty, 1, kk-1), A1, repmat(A_empty, 1, k - kk)       
        repmat(A_empty, k - kk, k)];                                   
    
    A2_all = A2_all +  [repmat(A_empty, kk-1, k);
        repmat(A_empty, 1, kk-1), A2, repmat(A_empty, 1, k - kk)
        repmat(A_empty, k - kk, k)];
end

A_empty = sparse(n*d, 2*n + 4*n*d + n + d);
A3_all = repmat(A_empty, k , k);
for kk=1:k    
    A3_all = A3_all +  [repmat(A_empty, kk-1, k);                      
        repmat(A_empty, 1, kk-1), A3, repmat(A_empty, 1, k - kk)       
        repmat(A_empty, k - kk, k)];                                   
end
b1 = repmat(b1,k,1);
b2 = repmat(b2,k,1);
b3 = repmat(b3,k,1);

%% loss
I = 1:n;
V = -ones(n,1);
J = (1:n)' + (y-1)*(2*n+4*n*d + n + d);
A4t = sparse(I, J, V, n, k*(2*n + 4*n*d + n + d)) + ...
    sparse(I, n+ J, -V, n, k*(2*n + 4*n*d + n + d));

A4 = [Av, -Av, sparse(n,4*n*d), -Av, zeros(n,d)];
b4 = -ones(n,1);

A_empty = sparse(n, 2*n + 4*n*d + n + d);
A4_all = repmat(A_empty, k , k);
for kk=1:k    
    A4_all = A4_all +  [repmat(A_empty, kk - 1, k);                      
        repmat(A_empty, 1, kk-1), A4, repmat(A_empty, 1, k - kk)       
        repmat(A_empty, k - kk, k)];
    
    A4_all = A4_all + [repmat(A_empty, kk - 1, k);                      
                       A4t       
                       repmat(A_empty, k - kk, k)];
end
b4 = repmat(b4, k, 1);

%% cost vector;

c = zeros(1, 2*n + 4*n*d + n + d);
c(2*n + 4*n*d + 1 : 2*n + 4*n*d + n) = 1;
c(2*n + 4*n*d + n + 1 : 2*n + 4*n*d + n + d) = lambda;
c = repmat(c, 1, k);

%% solving
model.A = -[A1_all;A2_all;A3_all;A4_all];
model.obj = c;
model.rhs = full(-[b1;b2;b3;b4]);
model.sense = '>';

params.OutputFlag = 0;
params.Threads = 16;
result = gurobi(model, params);

z = result.x;

f_hat = cell(k,1);
for kk=1:k
    shift = (kk-1) *(2*n + 4*n*d + n + d);
    params1.phi = z(shift + 1 : shift + n);
    params2.phi = z(shift + n+1 : shift+ 2*n);
    shift2 = 2*n + shift;

    params1.grad = reshape( z(shift2 + 1 : shift2 + n*d) - z(shift2 + n*d + 1 :shift2 + 2*n*d), [d,n])' ;
    params2.grad = reshape( z(shift2 + 2*n*d + 1 :shift2 + 3* n*d) - z(shift2 + 3*n*d + 1 : shift2 + 4*n*d), [d,n])' ;
    params1.phi = params1.phi - dot(params1.grad,X,2);
    params2.phi = params2.phi - dot(params2.grad,X,2);

    f_hat{kk} = @(X) dc_function(X, params1, params2);
end


