function Dn = dc_maximum_discrep(X, n_flips)
% computes empirical maximum-discrepency of DC_1 functions given data

%% inputs: 
% features matrix                   X : n x d
% number of evaluations             n_flips : 1, 2, 3, 4 ...

%% outputs
% empirical maximum-discrepency      Dn > 0

%% initialization
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

%% building the constraints
% A1  :  yhat1_i - y_hat1_j - (a^+_i - a^-_i)_^T (x_i - x_j) < 0
% A2  :  yhat2_i - y_hat2_j - (b^+_i - b^-_i)_^T (x_i - x_j) < 0
% A3  :  ||a^+_i|| + ||a^-_i|| + ||b^+_i|| + ||b^-_i|| - L < 0

b1 = sparse(n*(n-1),1);
b2 = sparse(n*(n-1),1);
b3 = ones(n,1);

A1 = [A_v, sparse(n*(n-1),n), A_g, -A_g, sparse(n*(n-1),2*n*d)];
A2 = [sparse(n*(n-1),n), A_v, sparse(n*(n-1),2*n*d), A_g, -A_g];
A3 = [sparse(n,2*n),A_norm,A_norm,A_norm,A_norm];

%% Solving

Dn = zeros(n_flips,1);
for flip = 1:n_flips
    % cost vector
    c_0 = [ones(floor(n/2),1);-ones(floor(n/2),1)];
    I_flips = randperm(n);
    c = [c_0(I_flips);-c_0(I_flips);zeros(4*n*d,1)]';
    
    %% solving
    try
        model.A = -[A1;A2;A3];
        model.obj = c;
        model.rhs = full(-[b1;b2;b3]);
        model.sense = '>';
        
        params.Threads = 16;
        params.OutputFlag = 0;
        result = gurobi(model, params);
        Dn(flip) = result.objval;
        
    catch
%         warning('Gurobi is not installed/working. Instead using MATLAB linear program solvers.')
        options = optimoptions('linprog','Display','off');
        lb = zeros(2*n+4*n*d,1);
        [~, Dn(flip) ] = linprog(c, [A1;A2;A3],[b1;b2;b3], [],[], lb, [], options );
    end
    
end
Dn = - mean(Dn) / n;

