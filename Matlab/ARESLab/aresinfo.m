function aresinfo(model, Xtr, Ytr, weights, showBF, sortByGCV, binarySimple, expandParentBF, cubicAsLinear)
% aresinfo
% Takes an ARES model, prints each basis function together with MSE, GCV,
% and R2GCV (GCV estimate of the Coefficient of Determination, R2) for a
% model from which the basis function was removed. By default, the
% functions are listed in the order of decreasing GCV - bigger is
% better. This can be used to judge whether, in the specific context of the
% given full model, a basis function is making an important contribution,
% or whether it just slightly helps to improve the global GCV score. See
% remarks below.
% For multi-response modelling, supply one submodel at a time.
%
% Call:
%   aresinfo(model, Xtr, Ytr, weights, showBF, sortByGCV, binarySimple, ...
%       expandParentBF, cubicAsLinear)
%
% All the input arguments, except the first three, are optional. Empty
% values are also accepted (the corresponding defaults will be used).
%
% Input:
%   model         : ARES model.
%   Xtr, Ytr      : Training data observations.
%   weights       : A vector of weights for observations. The same weights
%                   that were used when the model was built.
%   showBF        : Whether to show equations of basis functions or just
%                   list input variables the basis functions are using
%                   (default value = true).
%   sortByGCV     : Whether to list basis functions in the order of
%                   decreasing GCV or in the order in which they were
%                   included in the model (default value = true).
%   binarySimple  : See description of input argument of the same name for
%                   function areseq. (default value = false)
%   expandParentBF : See description of input argument of the same name for
%                   function areseq. (default value = false)
%   cubicAsLinear : This is for piecewise-cubic models only. Set to
%                   false (default) to show piecewise-cubic basis
%                   functions in their own mathematical form (Equation 34
%                   in Friedman, 1991a). Set to true to hide cubic
%                   smoothing - see the basis functions as if the model
%                   would be piecewise-linear. It's easier to understand
%                   the equations if smoothing is hidden. Note that, while
%                   the basis functions then look like from a piecewise-
%                   linear model, the coefficients are from the actual
%                   piecewise-cubic model.
%
% Remarks:
% 1. If it is determined that by deleting a one specific basis function GCV
%    would decrease (i.e., model would get better) or stay about the same,
%    you will see an exclamation mark next to the GCV value of that basis
%    function. This can happen either because the basis function is
%    irrelevant or it's redundant with some other basis function(s) in the
%    model. But note that if more than one basis function has such mark, it
%    does not mean that all of them should be deleted at once or at all.
%    Instead it means that you can try deleting them one after another
%    (using function aresdel) starting from the least important one, each
%    time recalculating this table, until all of the basis functions still
%    left in model stop having that mark. This is similar to what the
%    backward pruning phase does, except that it continues until model
%    consists only of the intercept term and then selects the model with
%    the best GCV from all tried sizes.
% 2. If you are using piecewise-cubic modelling with the default value for
%    parameter cubicFastLevel you may sometimes see that a basis function
%    has an exclamation mark even though you didn't disable the backward
%    pruning phase and therefore all irrelevant and redundant basis
%    functions should be already deleted. This is because by default models
%    are pruned as piecewise-linear and only after pruning they become
%    piecewise-cubic therefore it's possible that a basis function
%    inclusion of which previously slightly reduced GCV, suddenly slightly
%    increases it.
% 3. The column "hinges" shows types of functions that are multiplied to
%    comprise the basis function. Hinge functions are shown as "_/" or
%    "\_". Linear functions for variables that entered linearly are shown
%    as "/". The functions are showed in the same order as in the column
%    "basis function".

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
if (nargin < 5) || isempty(showBF)
    showBF = true;
end
if (nargin < 6) || isempty(sortByGCV)
    sortByGCV = true;
end
if (nargin < 7) || isempty(binarySimple)
    binarySimple = false;
end
if (nargin < 8) || isempty(expandParentBF)
    expandParentBF = false;
end
if (nargin < 9) || isempty(cubicAsLinear)
    cubicAsLinear = false;
end

gcvNull = gcv(1, var2(Ytr, weights), n, 0);

maxDeg = 0;
if ~isempty(model.knotdims)
    for i = 1 : length(model.knotdims)
        if length(model.knotdims{i}) > maxDeg
            maxDeg = length(model.knotdims{i});
        end
    end
end
if maxDeg > 2
    strAdd = repmat(' ', 1, (maxDeg-2) * 3);
else
    strAdd = '';
end

if model.trainParams.cubic
    fprintf('Type: piecewise-cubic\n');
else
    fprintf('Type: piecewise-linear\n');
end
fprintf('MSE: %g\n', model.MSE);
fprintf('GCV: %g\n', model.GCV);
fprintf('R2GCV: %g\n', 1 - model.GCV / gcvNull);
fprintf('Total number of basis functions (including intercept): %d\n', length(model.coefs));
fprintf('Total effective number of parameters: %g\n', ...
        length(model.coefs) + model.trainParams.c * length(model.knotdims) / 2);
fprintf('Basis functions:\n');
fprintf(['BF          MSE          GCV        R2GCV           coef    hinges   ' strAdd]);
if showBF
    fprintf('basis function\n');
else
    fprintf('variable(s)\n');
end
% Gather all MSEs and GCVs
mses = zeros(1,length(model.knotdims));
gcvs = zeros(1,length(model.knotdims));
for i = 1 : length(model.knotdims)
    modelReduced = aresdel(model, i, Xtr, Ytr, weights);
    mses(i) = modelReduced.MSE;
    gcvs(i) = modelReduced.GCV;
end
if sortByGCV
    [~, idx] = sort(gcvs, 'descend');
else
    idx = 1 : length(model.knotdims);
end
fprintf(['0             -            -            -%15.5g             ' strAdd '(intercept)\n'], model.coefs(1));
% Info about each basis function
for i = idx
    iStr = num2str(i);
    iStr = [iStr repmat(' ', 1, 4-length(iStr))];
    fprintf(iStr);
    fprintf('%11.5f', mses(i));
    fprintf('%13.5f', gcvs(i));
    if gcvs(i) / gcvNull - 1e-10 <= model.GCV / gcvNull % the same result as with gcvs alone but allows that 1e-10 stuff
        fprintf(' !');
    else
        fprintf('  ');
    end
    fprintf('%11.5f', 1 - gcvs(i) / gcvNull);
    fprintf('%15.5g', model.coefs(i+1));
    
    fprintf('   ');
    for dirs = model.knotdirs{i}
        if dirs == 2
            fprintf(' / ');
        elseif dirs > 0
            fprintf(' _/');
        else
            fprintf(' \\_');
        end
    end
    if length(model.knotdims{i}) < maxDeg
        fprintf(repmat(' ', 1, (maxDeg-length(model.knotdims{i})) * 3)); % fill the rest with spaces
    end
    if maxDeg < 2
        fprintf('   '); % fill the rest with spaces
    end
    
    if showBF
        if (~model.trainParams.cubic) || cubicAsLinear
            p = '%.5g';
        else
            p = '%.3g';
        end
        fprintf('    %s\n', getbfstr(model, i, p, binarySimple, expandParentBF, [], cubicAsLinear));
    else
        fprintf('    ');
        fprintf('%d ', model.knotdims{i});
        fprintf('\n');
    end
end
return

function res = var2(values, weights)
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
