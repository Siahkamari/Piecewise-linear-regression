function y = dc_function(X, params1, params2)

y = max(params1.phi' + X*params1.grad', [], 2);

if nargin == 3
    y = y - max(params2.phi' + X*params2.grad', [], 2);
end
    