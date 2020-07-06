function aresanova(model, Xtr, Ytr, weights)
% aresanova
% Performs ANOVA decomposition of given ARES model and reports the results.
% For details, see remarks on aresanova in user's manual as well as
% Sections 3.5 and 4.3 in (Friedman, 1991a) and Sections 2.4 and 4.1 in
% (Friedman, 1991b).
% For multi-response modelling, supply one submodel at a time.
%
% Call:
%   aresanova(model, Xtr, Ytr, weights)
%
% Input:
%   model         : ARES model.
%   Xtr, Ytr      : Training data observations. The same data that was used
%                   when the model was built.
%   weights       : Optional. A vector of weights for observations. The
%                   same weights that were used when the model was built.

% =========================================================================
% ARESLab: Adaptive Regression Splines toolbox for Matlab/Octave
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (C) 2009-2016  Gints Jekabsons
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% =========================================================================

% Last update: May 15, 2016

if nargin < 3
    error('Not enough input arguments.');
end

if isempty(Xtr) || isempty(Ytr)
    error('Data is empty.');
end
if iscell(Xtr) || iscell(Ytr)
    error('Xtr and Ytr should not be cell arrays.');
end
if islogical(Ytr)
    Ytr = double(Ytr);
elseif ~isfloat(Ytr)
    error('Ytr data type should be double or logical.');
end
if ~isfloat(Xtr)
    error('Xtr data type should be double.');
end
if any(any(isnan(Xtr)))
    error('ARESLab cannot handle missing values (NaN).');
end

[n, d] = size(Xtr); % number of observations and number of input variables
if size(Ytr,1) ~= n
    error('The number of rows in Xtr and Ytr should be equal.');
end
if length(model) > 1
    error('This function works with single-response models only. You can supply one submodel at a time.');
else
    if iscell(model)
        model = model{1};
    end
end
if length(model.minX) ~= d
    error('The number of columns in Xtr is different from the number when the model was built.');
end
if size(Ytr,2) ~= 1
    error('Ytr should have one column.');
end
if (nargin < 4)
    weights = [];
else
    if (~isempty(weights)) && ...
       ((size(weights,1) ~= n) || (size(weights,2) ~= 1))
        error('weights vector is of wrong size.');
    end
end

gcvNull = gcv(1, var2(Ytr, weights), n, 0);
nVars = length(model.minX);
nBasisExI = length(model.knotdims); % excluding intercept

if model.trainParams.cubic
    fprintf('Type: piecewise-cubic\n');
else
    fprintf('Type: piecewise-linear\n');
end
fprintf('GCV: %g\n', model.GCV);
fprintf('R2GCV: %g\n', 1 - model.GCV / gcvNull);
fprintf('Total number of basis functions (including intercept): %d\n', length(model.coefs));
fprintf('Total effective number of parameters: %g\n', ...
        length(model.coefs) + model.trainParams.c * nBasisExI / 2);

if nBasisExI <= 0
    return;
end

fprintf('ANOVA decomposition:\n');
fprintf('Function        STD             GCV         R2GCV       #basis  #params     variable(s)\n');
maxNumInteract = 0; % max number of interactions
for i = 1 : nBasisExI
    if maxNumInteract < length(model.knotdims{i})
        maxNumInteract = length(model.knotdims{i});
    end
end
priority = zeros(1,nBasisExI); % for ordering ANOVA functions according to number of variables and variable indices
for i = 1 : nBasisExI
    sorted = sort(model.knotdims{i});
    sorted = [zeros(1, maxNumInteract - length(sorted)) sorted];
    priority(i) = sum(sorted .* (nVars .^ ((maxNumInteract-1):-1:0)));
end
[~, idx] = sort(priority);
% Show ANOVA functions ordered by their priority (i.e., depending on the subset of used variables)
counterANOVA = 0;
for i = 1 : length(idx)
    if idx(i) <= 0
        continue;
    end
    counterANOVA = counterANOVA + 1;
    usedBasis = printLine(model, Xtr, Ytr, weights, gcvNull, counterANOVA, model.knotdims{idx(i)});
    for j = usedBasis
        idx(idx == j) = 0;
    end
end
return

function usedBasis = printLine(model, Xtr, Ytr, weights, gcvNull, counterANOVA, combination)
    [modelReduced, usedBasis] = aresanovareduce(model, combination, true);
    counterStr = num2str(counterANOVA);
    counterStr = [counterStr repmat(' ', 1, 4-length(counterStr))];
    fprintf(counterStr);
    % standard deviation of the ANOVA function
    fprintf('%15f', sqrt(var2(arespredict(modelReduced, Xtr), weights)));
    % GCV when the basis functions of the ANOVA function are deleted
    modelReduced = aresdel(model, usedBasis, Xtr, Ytr, weights);
    fprintf('%16f', modelReduced.GCV);
    if modelReduced.GCV / gcvNull - 1e-10 <= model.GCV / gcvNull % the same result as with gcvs alone but allows that 1e-10 stuff
        fprintf(' !');
    else
        fprintf('  ');
    end
    fprintf('%12.5f', 1 - modelReduced.GCV / gcvNull);
    % the number of basis functions for that ANOVA function
    fprintf('%13d', length(usedBasis));
    % effective number of parameters
    fprintf('%9.2f     ', length(usedBasis) + model.trainParams.c * length(usedBasis) / 2);
    % used variables
    fprintf('%d ', sort(combination));
    fprintf('\n');
return

function res = var2(values, weights)
% Calculates variance either normally or using weights
if isempty(weights)
    res = var(values, 1);
else
    valuesMean = sum(values(:,1) .* weights) / sum(weights);
    res = sum(((values(:,1) - valuesMean) .^ 2) .* weights) / sum(weights);
end
return

function g = gcv(nBasis, MSE, n, c)
% Calculates GCV from model complexity, its Mean Squared Error, number of
% observations n, and penalty c.
enp = nBasis + c * (nBasis - 1) / 2; % model's effective number of parameters
if enp >= n
    g = Inf;
else
    p = 1 - enp / n;
    g = MSE / (p * p);
end
return
