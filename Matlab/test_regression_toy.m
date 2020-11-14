
% this script plots the regression function vs the learned functions on 
% synthetic data for theoretical and cross-validates values of
% regularization constant

clear,  clc
% rng(0);

%% settings
d = 100;                          % dimmension of data
n_train = 10;                   % number of data points
sigma = 0.0;                    % standard deviation of the noise

%% generating synthetic data
X_train = rand(n_train, d);
y_train = regfunction(X_train, sigma);

X_test = rand(10000, d);
y_test = regfunction(X_test, 0);

%% learning
Dn = rademacher_dc_sum_linf(X_train(1:2*floor(n_train/2),:), 2)
lambda_0 = 4*Dn*n_train;
f_hatTh = dc_fit(y_train, X_train, lambda_0);

f_hatCV = auto_tune_dc_fit(y_train, X_train, "regression");

%% plotting
regression_plot(X_train,y_train, @(X) regfunction(X,0), f_hatTh, f_hatCV);

%% regression function
function y = regfunction(X, sigma)
[n, d] = size(X);
X = 3*pi*X;

if d > 1
    y = sin(X(:,1)) + cos(X(:,2)) + 3*log(abs(X(:,1)+X(:,2))+1);
else
    y = sin(X)  + 3*log(abs(X) +1);
end
y = y + sigma*randn(n,1);
end

