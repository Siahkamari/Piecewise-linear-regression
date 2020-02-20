
% See Sections 3.1 and 3.3 in user's manual for details.

clear; clc;

% The data
X = rand(200,10);
Y = 10*sin(pi*X(:,1).*X(:,2)) + 20*(X(:,3)-0.5).^2 + 10*X(:,4) + 5*X(:,5) + 0.5*randn(200,1);

%%

% Parameters
%params = aresparams2('maxFuncs', 21, 'maxInteractions', 2); % from user's manual
% Better parameters if we know that variables 1 and 2 should be the only
% ones interacting and variables 4 and 5 should enter the model linearly:
params = aresparams2('maxFuncs', 21, 'maxInteractions', 2, 'yesInteract', [1 2], 'forceLinear', [4 5]);
% add "'cubic', false" for piecewise-linear model

% Building the model
disp('Building the model ==================================================');
[model, ~, resultsEval] = aresbuild(X, Y, params);
model

% Plotting model selection from the backward pruning phase
figure;
hold on; grid on; box on;
h(1) = plot(resultsEval.MSE, 'Color', [0 0.447 0.741]);
h(2) = plot(resultsEval.GCV, 'Color', [0.741 0 0.447]);
numBF = numel(model.coefs);
h(3) = plot([numBF numBF], get(gca, 'ylim'), '--k');
xlabel('Number of basis functions');
ylabel('MSE, GCV');
legend(h, 'MSE', 'GCV', 'Selected model');

% Plotting the model. Varying variables 1 and 2. Variables 3, 4, and 5 are fixed at their (min+max)/2
aresplot(model, [1 2]);

% Variable importance
disp('Variable importances ================================================');
aresimp(model, X, Y, resultsEval);

% ANOVA decomposition
disp('ANOVA decomposition =================================================');
aresanova(model, X, Y);

% Plots of ANOVA functions (we know which ones to plot thanks to the ANOVA decomposition)
modelReduced = aresanovareduce(model, [1 2]);
aresplot(modelReduced);
for i = 3 : 5
    modelReduced = aresanovareduce(model, i);
    aresplot(modelReduced);
end

% Info on the basis functions
disp('Info on the basis functions =========================================');
aresinfo(model, X, Y);

% Printing the model
disp('The model ===========================================================');
areseq(model, 5);

%%

% 10-fold Cross-Validation
disp('Cross-Validation ====================================================');
rng(1);
resultsCV = arescv(X, Y, params)

%%

% Code from Section 3.3

clc;

% Parameters
%params = aresparams2('maxFuncs', 51, 'maxInteractions', 2); % from user's manual
params = aresparams2('maxFuncs', 51, 'maxInteractions', 2, 'yesInteract', [1 2], 'forceLinear', [4 5]);

% 10-fold Cross-Validation
rng(1);
[resultsTotal, resultsFolds, resultsPruning] = arescv(X, Y, params, [], [], [], [], [], true);

% Plotting the results
figure;
hold on; grid on; box on;
for i = 1 : size(resultsPruning.GCV,1)
    plot(resultsPruning.GCV(i,:), ':', 'Color', [0.259 0.706 1]);
    plot(resultsPruning.MSEoof(i,:), ':', 'Color', [1 0.259 0.706]);
end
plot(resultsPruning.meanGCV, 'Color', [0 0.447 0.741], 'LineWidth', 2);
plot(resultsPruning.meanMSEoof, 'Color', [0.741 0 0.447], 'LineWidth', 2);

ylim = get(gca, 'ylim');
posY = resultsPruning.meanGCV(resultsPruning.nBasisGCV);
plot([resultsPruning.nBasisGCV resultsPruning.nBasisGCV], [ylim(1) posY], '--', 'Color', [0 0.447 0.741]);
plot(resultsPruning.nBasisGCV, posY, 'o', 'MarkerSize', 8, 'Color', [0 0.447 0.741]);
posY = resultsPruning.meanMSEoof(resultsPruning.nBasisMSEoof);
plot([resultsPruning.nBasisMSEoof resultsPruning.nBasisMSEoof], [ylim(1) posY], '--', 'Color', [0.741 0 0.447]);
plot(resultsPruning.nBasisMSEoof, posY, 'o', 'MarkerSize', 8, 'Color', [0.741 0 0.447]);

xlabel('Number of basis functions');
ylabel('GCV, MSE_{oof}');
