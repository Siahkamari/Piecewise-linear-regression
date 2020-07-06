function model = aresdel(model, funcsToDel, Xtr, Ytr, weights)
% aresdel
% Deletes basis functions from ARES model, recalculates model's
% coefficients and relocates additional knots for piecewise-cubic models
% (as opposed to aresanovareduce which does not recalculate and relocate
% anything).
%
% Call:
%   model = aresdel(model, funcsToDel, Xtr, Ytr, weights)
%
% Input:
%   model         : ARES model or, for multi-response modelling, a cell
%                   array of ARES models.
%   funcsToDel    : A vector of indices for basis functions to delete.
%                   Intercept term is not indexed, i.e., the numbering is
%                   the same as in model.knotdims, model.knotsites, and
%                   model.knotdirs.
%   Xtr, Ytr      : Training data observations. The same data that was used
%                   when the model was built.
%   weights       : Optional. A vector of weights for observations. The
%                   same weights that were used when the model was built.
%
% Output:
%   model         : Reduced ARES model.

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

if nargin < 4
    error('Not enough input arguments.');
end
if isempty(funcsToDel)
    return;
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

funcsToDel = unique(funcsToDel(:)');
[n, d] = size(Xtr); % number of observations and number of input variables
if size(Ytr,1) ~= n
    error('The number of rows in Xtr and Ytr should be equal.');
end
numModels = length(model);
if (numModels == 1) && iscell(model)
    model = model{1};
end
if (numModels ~= size(Ytr,2))
    error('model should contain as many models as there are columns in Ytr.');
end
if ((numModels == 1) && any(funcsToDel > length(model.knotdims))) || ...
   ((numModels > 1) && any(funcsToDel > length(model{1}.knotdims)))
    error('funcsToDel contains one or more indices that are out of range.');
end
if ((numModels == 1) && (length(model.minX) ~= d)) || ...
   ((numModels > 1) && (length(model{1}.minX) ~= d))
    error('The number of columns in Xtr is different from the number when the model was built.');
end
if (nargin < 5)
    weights = [];
else
    if (~isempty(weights)) && ...
       ((size(weights,1) ~= n) || (size(weights,2) ~= 1))
        error('weights vector is of wrong size.');
    end
end

if numModels == 1
    model = delBF(model, funcsToDel, Xtr, Ytr, weights);
else
    for k = 1 : numModels
        model{k} = delBF(model{k}, funcsToDel, Xtr, Ytr(:,k), weights);
    end
end
return

function model = delBF(model, listDel, Xtr, Ytr, weights)
% Deletes basis functions from one model
listDel = sort(listDel, 2, 'descend');
for i = listDel % deleting in reverse order
    %model.coefs(i+1) = []; % we don't need this because coefs are completely recalculated below
    model.knotdims(i) = [];
    model.knotsites(i) = [];
    model.knotdirs(i) = [];
    model.parents(i) = [];
    model.parents = updateParents(model.parents, i);
    if model.trainParams.cubic
        model.t1(i,:) = [];
        model.t2(i,:) = [];
    end
    if isfield(model, 'X')
        model.X(:,i+1) = [];
    end
end

n = size(Xtr,1);

if isfield(model, 'X') % in this case we can do it a bit faster
    % For piecewise-linear models, there is nothing to recalculate in model.X
    if model.trainParams.cubic
        % correct the side knots for the modified model
        [model.t1, model.t2, diff] = ...
            findsideknots(model, [], [], size(model.t1,2), model.minX, model.maxX, model.t1, model.t2);
        % recalculate only columns that have changed
        for i = diff
            model.X(:,i+1) = createbasisfunction(Xtr, model.X, model.knotdims{i}, model.knotsites{i}, ...
                model.knotdirs{i}, 0, model.minX, model.maxX, model.t1(i,:), model.t2(i,:));
        end
    end
    origWarningState = warning('off');
    [model.coefs, model.MSE] = lreg(model.X, Ytr, weights);
    warning(origWarningState);
else
    X = ones(n,length(model.knotdims)+1);
    if model.trainParams.cubic
        % correct the side knots for the modified model
        [model.t1, model.t2] = findsideknots(model, [], [], size(model.t1,2), model.minX, model.maxX, [], []);
        % calculate all columns
        for i = 1 : length(model.knotdims)
            X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                model.knotdirs{i}, 0, model.minX, model.maxX, model.t1(i,:), model.t2(i,:));
        end
    else
        for i = 1 : length(model.knotdims)
            X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                model.knotdirs{i}, 0, model.minX, model.maxX);
        end
    end
    origWarningState = warning('off');
    [model.coefs, model.MSE] = lreg(X, Ytr, weights);
    warning(origWarningState);
end

if isempty(weights)
    model.MSE = model.MSE / n;
else
    model.MSE = model.MSE / sum(weights);
end
model.GCV = gcv(length(model.coefs), model.MSE, n, model.trainParams.c);
return

function parents = updateParents(parents, deletedIdx)
% Updates direct parent indices after deletion of a basis function.
parents(parents == deletedIdx) = 0;
tmp = parents > deletedIdx;
parents(tmp) = parents(tmp) - 1;
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

function [coefs, err] = lreg(x, y, w)
% Linear regression (unweighted and weighted)
if isempty(w)
    coefs = (x' * x) \ (x' * y);
    err = sum((y-x*coefs).^2);
else
    xw = bsxfun(@times, x, w)';
    coefs = (xw * x) \ (xw * y);
    err = sum((y-x*coefs).^2.*w); % later in code this is divided by sum of weights
end
return
