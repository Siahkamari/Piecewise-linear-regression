function f_hat = auto_tune_dc_fit(y, X)  

% Runs dc_fit over various parameters of
% lambda, choosing that with the highest accuracy. 
%
% Returns: a learned dc_function

if length(unique(y)) == 2
    task = "classification";
else
    task = "regression";
end


lambdas = 10.^(-3:2);
n_folds = 3;

loss = zeros(length(lambdas), 1);
for i=1:length(lambdas)
    fprintf('Tuning dc_fit: lambda = %.3f \n', lambdas(i));
    loss(i) = cross_validate(y, X, @(y,X) dc_fit(y, X, lambdas(i)), n_folds, task);
end

[~,i] = min(loss);
lambda = lambdas(i);

fprintf('Optimal lambda value: %.3f\n', lambda); 

f_hat = dc_fit(y, X, lambda);
         