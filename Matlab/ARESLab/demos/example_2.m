
% See Section 3.2 in user's manual for details.

clear; clc;

% The data
[X1,X2] = meshgrid(-1:0.2:1, -1:0.2:1);
X_train(:,1) = reshape(X1, numel(X1), 1);
X_train(:,2) = reshape(X2, numel(X2), 1);
clear X1 X2;
y_train = sin(0.83*pi*X_train(:,1)) .* cos(1.25*pi*X_train(:,2));
X_test = rand(10000,2);
y_test = sin(0.83*pi*X_test(:,1)) .* cos(1.25*pi*X_test(:,2));

%% MARS

% Parameters
% params = aresparams2('maxFuncs', 101, 'c', 0, 'maxInteractions', 2); % piecewise-cubic
params = aresparams2('maxFuncs', 101, 'c', 0, 'maxInteractions', 2, 'cubic', false); % piecewise-linear

% Building the model
disp('Building the model ==================================================');
[model, ~, resultsEval] = aresbuild(X_train, y_train, params);


% Testing on test data
disp('Testing on test data ================================================');
results = arestest(model, X_test, y_test);
disp(results.R2)
