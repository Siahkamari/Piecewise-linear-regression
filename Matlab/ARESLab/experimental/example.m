
% Example for ARES GLM logit/probit models

clear; clc;

load fisheriris;
X = meas;
Y = species;

% Configuration
params = aresparams2('cubic', false, 'maxInteractions', -1);
reduce = 'stepwise'; % 'off', 'stepwise', or 'regularize'
glmDistr = 'binomial';
glmParams = {'link', 'logit'}; % or {'link', 'probit'};

% Training
glm = glmaresbuild(X, Y, params, reduce, glmDistr, glmParams);
% Class prediction
Yhat = glmarespredict(glm, X);
% Assessing performance
cp = classperf(Y, Yhat);
fprintf('Training error rate: %f\n', cp.ErrorRate);
confusionmat(Y, Yhat)

% Cross-Validation
rng(1); % for reproducibility
k = 5; % number of folds
indices = crossvalind('Kfold', Y, k);
Yhat = Y; % so that Yhat has the same type and size
cp = classperf(Y);
for i = 1 : k
    disp(['Fold ' int2str(i)]);
    testWhich = indices == i;
    trainWhich = ~testWhich;
    % Training
    glm = glmaresbuild(X(trainWhich,:), Y(trainWhich,:), params, reduce, glmDistr, glmParams, false);
    % Class prediction
    Yhat(testWhich) = glmarespredict(glm, X(testWhich,:));
    % Storing results
    classperf(cp, Yhat(testWhich), testWhich);
end
% Assessing performance
fprintf('CV error rate: %f\n', cp.ErrorRate);
confusionmat(Y, Yhat)
