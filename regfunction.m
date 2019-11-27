function y = regfunction(X, sigma)
[n, d] = size(X);

X = X*2*pi;

y = sin(X(:,1)) + cos(X(:,min(2,d)));
% y = X(:,1).^2 + 2*X(:,2).^2 ;
% y = (abs(X).^(1.5))*([3;1]);
% y = 1/2*sum(X.^2,2);
% y = sum(X,2);

y = y + sigma*randn(n,1);
end