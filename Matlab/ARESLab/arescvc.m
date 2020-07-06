function [cBest, results] = arescvc(X, Y, trainParams, cTry, k, shuffle, ...
    nCross, weights, testWithWeights, verbose)
% arescvc
% Finds the "best" value for penalty c of the Generalized Cross-Validation
% criterion from a set of candidate values using Cross-Validation assuming
% that all the other parameters of function aresparams would stay fixed.
% For a better alternative to using this function, see Section 3.3 in
% user's manual.
%
% Call:
%   [cBest, results] = arescvc(X, Y, trainParams, cTry, k, shuffle, ...
%   	nCross, weights, testWithWeights, verbose)
%
% All the input arguments, except the first three, are optional. Empty
% values are also accepted (the corresponding defaults will be used).
% Note that, if argument shuffle is set to true, this function employs
% random number generator for which you can set seed before calling the
% function.
%
% Input:
%   X, Y          : The data. See description of function aresbuild.
%   trainParams   : A structure of training parameters (see function
%                   aresparams for details).
%   cTry          : A set of candidate values for c. (default value = 1:5)
%   k             : Value of k for k-fold Cross-Validation. The typical
%                   values are 5 or 10. For Leave-One-Out Cross-Validation
%                   set k equal to n. (default value = 10)
%   shuffle       : Whether to shuffle the order of the observations before
%                   performing Cross-Validation. (default value = true)
%   nCross        : How many times to repeat Cross-Validation with
%                   different data partitioning. This can be used to get
%                   more stable results. Default value = 1, i.e., no
%                   repetition. Useless if shuffle = false.
%   weights       : A vector of weights for observations. See description
%                   of function aresbuild.
%   testWithWeights : Set to true to use weights vector for both, training
%                   and testing. Set to false to use it for training only.
%                   This argument has any effect only when weights vector
%                   is provided. (default value = true)
%   verbose       : Whether to output additional information to console.
%                   (default value = true)
%
% Output:
%   cBest         : The best found value for penalty c.
%   results       : A matrix with two columns. First column contains all
%                   values from cTry. Second column contains the calculated
%                   MSE values (averaged across all Cross-Validation folds)
%                   for the corresponding cTry values.
%
% Remarks:
% This function finds the "best" penalty c value in a clever way. In each
% Cross-Validation iteration, the forward phase in aresbuild is done only
% once while the backward phase is done separately for each cTry value. The
% results will be the same as if each time a full model building process
% would be performed because in the forward phase the GCV criterion is not
% used. Except if aresparams parameter terminateWhenInfGCV is set to true -
% in that case the results may sometimes slightly differ.

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

if isempty(X) || isempty(Y)
    error('Data is empty.');
end
if iscell(X) || iscell(Y)
    error('X and Y should not be cell arrays.');
end
if islogical(Y)
    Y = double(Y);
elseif ~isfloat(Y)
    error('Y data type should be double or logical.');
end
if ~isfloat(X)
    error('X data type should be double.');
end
if any(any(isnan(X)))
    error('ARESLab cannot handle missing values (NaN).');
end

[n, d] = size(X); % number of observations and number of input variables
[ny, dy] = size(Y); % number of observations and number of output variables
if ny ~= n
    error('The number of rows in X and Y should be equal.');
end

if ~trainParams.prune
	error('Model pruning is disabled (does not make sense because penalty c is for controlling pruning).');
end

if (nargin < 4) || isempty(cTry)
    cTry = 1:5;
end
cNum = length(cTry);
if (nargin < 5) || isempty(k)
    k = 10;
end
if k < 2
    error('k should not be smaller than 2.');
end
if k > n
    error('k should not be larger than the number of observations.');
end
if (nargin < 6) || isempty(shuffle)
    shuffle = true;
end
if (nargin < 7) || isempty(nCross)
    nCross = 1;
else
    if (nCross > 1) && (~shuffle)
        error('nCross>1 but shuffle=false');
    end
end
if (nargin < 8)
    weights = [];
else
    if (~isempty(weights)) && ...
       ((size(weights,1) ~= n) || (size(weights,2) ~= 1))
        error('weights vector is of wrong size.');
    end
end
if (nargin < 9) || isempty(testWithWeights)
    testWithWeights = true;
end
if (nargin < 10) || isempty(verbose)
    verbose = true;
end

MSE = zeros(cNum,k*nCross);
% Repetition of Cross-Validation nCross times
for iCross = 1 : nCross
    to = iCross * k;
    range = (to - k + 1) : to;
    MSE(:,range) = doCV(X, Y, trainParams, cTry, k, shuffle, weights, testWithWeights, verbose, n, d, dy, cNum);
end
MSE = mean(MSE,2); % mean across all folds
[~, ind] = min(MSE);
cBest = cTry(ind);
results(:,1) = cTry;
results(:,2) = MSE;
return

function MSE = doCV(X, Y, trainParams, cTry, k, shuffle, weights, testWithWeights, verbose, n, d, dy, cNum)

if shuffle
    ind = randperm(n); % shuffle the data
else
    ind = 1 : n;
end

% divide the data into k subsets (for compatibility with Octave, not using special Matlab functions)
minsize = floor(n / k);
sizes = repmat(minsize, k, 1);
remainder = n - minsize * k;
if remainder > 0
    sizes(1:remainder) = minsize + 1;
end
offsets = ones(k, 1);
for i = 2 : k
    offsets(i) = offsets(i-1) + sizes(i-1);
end

if isempty(weights)
    w = [];
end
if isempty(weights) || (~testWithWeights)
    wtst = [];
end

% perform training and testing k times
MSE = zeros(cNum,k);
for i = 1 : k
    if verbose
        disp(['Fold #' num2str(i)]);
    end
    Xtr = zeros(n-sizes(k-i+1), d);
    Ytr = zeros(n-sizes(k-i+1), dy);
    if ~isempty(weights)
        w = zeros(n-sizes(k-i+1), 1);
    end
    currsize = 0;
    for j = 1 : k
        if k-i+1 ~= j
            idxtrain = ind(offsets(j):offsets(j)+sizes(j)-1);
            Xtr(currsize+1 : currsize+1+sizes(j)-1, :) = X(idxtrain, :);
            Ytr(currsize+1 : currsize+1+sizes(j)-1, :) = Y(idxtrain, :);
            if ~isempty(weights)
                w(currsize+1 : currsize+1+sizes(j)-1, 1) = weights(idxtrain, 1);
            end
            currsize = currsize + sizes(j);
        end
    end
    idxtst = ind(offsets(k-i+1):offsets(k-i+1)+sizes(k-i+1)-1);
    Xtst = X(idxtst, :);
    Ytst = Y(idxtst, :);
    if (~isempty(weights)) && testWithWeights
        wtst = weights(idxtst, :);
    end
    trainParams.prune = false;
    pretendLinear = trainParams.cubic && (trainParams.cubicFastLevel >= 2);
    if pretendLinear
        % for doCubicFastLevel>=2 we will pretend that this is not cubic
        % modelling so that aresbuild does not return a model that is
        % forced to be cubic already before the turned-off backward phase
        trainParams.cubic = false;
    end
    if trainParams.terminateWhenInfGCV
        trainParams.c = min(cTry); % so that forward phase doesn't terminate too soon
    end
    modelUnpruned = aresbuild(Xtr, Ytr, trainParams, w, true, [], [], false); % unpruned model
    trainParams.prune = true;
    if pretendLinear
        trainParams.cubic = true;
    end
    for j = 1 : cNum
        trainParams.c = cTry(j);
        model = aresbuild(Xtr, Ytr, trainParams, w, false, modelUnpruned, [], false); % pruned model
        res = arestest(model, Xtst, Ytst, wtst);
        if dy == 1
            MSE(j,i) = res.MSE;
        else
            MSE(j,i) = mean(res.MSE);
        end
    end
end

return
