function Dn = rademacher_dc_sum_linf(X, n_flips)

[n,d] = size(X);

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

%% building the constraints
% A1  :  yhat_i - y_hat_j + c_i - c_j - (a^+_i - a^-_i)_^T (x_i - x_j) < 0
% A2  :  c_i - c_j - (b^+_i - b^-_i)_^T (x_i - x_j) < 0
% A3  :  ||a^+_i|| + ||a^-_i|| + ||b^+_i|| + ||b^-_i|| - L < 0

A1 = [A_v, A_v, A_g, -A_g, sparse(n*(n-1),2*n*d + d)];
b1 = sparse(n*(n-1),1);
A2 = [sparse(n*(n-1),n), A_v, sparse(n*(n-1),2*n*d), A_g, -A_g, sparse(n*(n-1), d)];
b2 = sparse(n*(n-1),1);
A3 = [sparse(n*d,2*n),A_norm,A_norm,A_norm,A_norm, -A_norm_b];
b3 = zeros(n*d,1);
A32 =[sparse(1,2*n + 4*n*d),ones(1,d)];
b32 = 1;


%% solution bounds
lb = zeros(2*n + 4*n*d + d, 1);
lb(1:n) = -inf;

Dn = zeros(n_flips,1);
for flip = 1:n_flips
    % cost vector
    c_0 = [ones(floor(n/2),1);-ones(floor(n/2),1)];
    c = [c_0(randperm(n));zeros(n+4*n*d + d,1)]';
    
    %% solving
    try
        model.A = -[A1;A2;A3;A32];
        model.obj = c;
        model.rhs = full(-[b1;b2;b3;b32]);
        model.sense = '>';
        
        params.Threads = 16;
        params.OutputFlag = 0;
        result = gurobi(model, params);
        Dn(flip) = result.objval;
        
    catch
%         warning('Gurobi is not installed/working. Instead using MATLAB linear program solvers.')
        options = optimoptions('linprog','Display','off');
        lb = zeros(2*n+4*n*d,1);
        [~, Dn(flip) ] = linprog(c, [A1;A2;A3;A32],[b1;b2;b3;b32], [],[], lb, [], options );
    end
    
end
Dn = - 2*mean(Dn) / n;

