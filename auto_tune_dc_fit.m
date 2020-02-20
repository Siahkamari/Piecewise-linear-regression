function f_hat = auto_tune_dc_fit(y, X, task)

% learns dc functions for regression/classification tasks given data

%% inputs: 
% labels                    y : n x 1
% features matrix           X : n x d

%% outputs
% learned dc function      f_hat : R ^(n x d) -> R ^(n) 

%% initialization
n_flips = 2;
Dn = dc_maximum_discrep(X(1:2*floor(length(y)/2),:), n_flips);
lambda_0 = 4*Dn*length(y);

lambdas = 2.^(-9:0)*lambda_0;
n_folds = 5;

loss = zeros(length(lambdas), 1);
fprintf('\n\n'); 
for i=1:length(lambdas)
    fprintf('Tuning dc_fit: lambda = %.3f \n', lambdas(i));
    
    if task == "regression"
        loss(i) = cross_validate(y, X, @(y,X) dc_fit(y, X, lambdas(i)), n_folds, task);
    elseif task == "classification"
        loss(i) = cross_validate(y, X, @(y,X) dc_fit_multi_class(y, X, lambdas(i)), n_folds, task);
    end
end

[~,i] = min(loss);
lambda = lambdas(i);

fprintf('Optimal lambda value: %.3f\n\n', lambda); 

if task == "regression"
    f_hat = dc_fit(y, X, lambda);
elseif task == "classification"
    f_hat = dc_fit_multi_class(y, X, lambda);
end
         