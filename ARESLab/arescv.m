function [resultsTotal, resultsFolds, resultsPruning] = arescv(X, Y, ...
    trainParams, k, shuffle, nCross, weights, testWithWeights, evalPruning, verbose)
% arescv
% Tests ARES performance using k-fold Cross-Validation.
% The function has additional built-in capabilities for finding the "best"
% number of basis functions for the final ARES model (maxFinalFuncs for
% function aresparams). See example of usage in user's manual for details.
%
% Call:
%   [resultsTotal, resultsFolds, resultsPruning] = arescv(X, Y, ...
%       trainParams, k, shuffle, nCross, weights, testWithWeights, ...
%       evalPruning, verbose)
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
%   k             : Value of k for k-fold Cross-Validation. The typical
%                   values are 5 or 10. For Leave-One-Out Cross-Validation
%                   set k equal to n. (default value = 10)
%   shuffle       : Whether to shuffle the order of observations before
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
%   evalPruning   : Whether to evaluate all the candidate models of the
%                   pruning phase. If set to true, the output argument
%                   resultsPruning contains the results. See example of
%                   usage in user's manual. (default value = false)
%   verbose       : Whether to output additional information to console.
%                   (default value = true)
%
% Output:
%   resultsTotal  : A structure of Cross-Validation results. The results
%                   are averaged across Cross-Validation folds and, in case
%                   of multi-response data, also across multiple models.
%   resultsFolds  : A structure of vectors or matrices (in case of multi-
%                   response data) of results for each Cross-Validation
%                   fold. Columns correspond to Cross-Validation folds.
%                   Rows correspond to models.
%   Both structures have the following fields:
%     MAE         : Mean Absolute Error.
%     MSE         : Mean Squared Error.
%     RMSE        : Root Mean Squared Error.
%     RRMSE       : Relative Root Mean Squared Error. Not reported for
%                   Leave-One-Out Cross-Validation.
%     R2          : Coefficient of Determination. Not reported for
%                   Leave-One-Out Cross-Validation.
%     nBasis      : Number of basis functions in model (including the
%                   intercept term).
%     nVars       : Number of input variables included in model.
%     maxDeg      : Highest degree of variable interactions in model.
%   resultsPruning: Available only if evalPruning = true. See example of
%                   usage in the user's manual Section 3.3. The structure
%                   has the following fields:
%     GCV         : A matrix of GCV values for best candidate models of
%                   each size at each Cross-Validation fold. The number of
%                   rows is equal to k*nCross. Column index corresponds to
%                   the number of basis functions in a model.
%     meanGCV     : A vector of mean GCV values for each model size across
%                   all Cross-Validation folds.
%     nBasisGCV   : The number of basis functions (including the intercept
%                   term) for which the mean GCV is minimum.
%     MSEoof      : A matrix of out-of-fold MSE values for best candidate
%                   models of each size at each Cross-Validation fold. The
%                   number of rows for this matrix is equal to k*nCross.
%                   Column index corresponds to the number of basis
%                   functions in a model.
%     meanMSEoof  : A vector of mean out-of-fold MSE values for each model
%                   size across all Cross-Validation folds.
%     nBasisMSEoof : The number of basis functions (including the intercept
%                   term) for which the mean out-of-fold MSE is minimum.
%     R2GCV       : A matrix of R2GCV (R2 estimated by GCV in training
%                   data) values for best candidate models of each size at
%                   each Cross-Validation fold. The number of rows is equal
%                   to k*nCross. Column index corresponds to the number of
%                   basis functions in a model.
%     meanR2GCV   : A vector of mean R2GCV values for each model size
%                   across all Cross-Validation folds.
%     nBasisR2GCV : The number of basis functions (including the intercept
%                   term) for which the mean R2GCV is maximum.
%     R2oof       : A matrix of out-of-fold R2 values for best candidate
%                   models of each size at each Cross-Validation fold. The
%                   number of rows for this matrix is equal to k*nCross.
%                   Column index corresponds to the number of basis
%                   functions in a model.
%     meanR2oof   : A vector of mean out-of-fold R2 values for each model
%                   size across all Cross-Validation folds.
%     nBasisR2oof : The number of basis functions (including the intercept
%                   term) for which the mean out-of-fold R2 is maximum.

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

if (nargin < 4) || isempty(k)
    k = 10;
end
if k < 2
    error('k should not be smaller than 2.');
end
if k > n
    error('k should not be larger than the number of observations.');
end
if (nargin < 5) || isempty(shuffle)
    shuffle = true;
end
if (nargin < 6) || isempty(nCross)
    nCross = 1;
else
    if (nCross > 1) && (~shuffle)
        error('nCross>1 but shuffle=false');
    end
end
if (nargin < 7)
    weights = [];
else
    if (~isempty(weights)) && ...
       ((size(weights,1) ~= n) || (size(weights,2) ~= 1))
        error('weights vector is of wrong size.');
    end
end
if (nargin < 8) || isempty(testWithWeights)
    testWithWeights = true;
end
if (nargin < 9) || isempty(evalPruning)
    evalPruning = false;
else
    if evalPruning && (~trainParams.prune)
        error('evalPruning=true but trainParams.prune=false');
    end
end
if (nargin < 10) || isempty(verbose)
    verbose = true;
end

resultsFolds.MAE = Inf(k*nCross,dy);
resultsFolds.MSE = Inf(k*nCross,dy);
resultsFolds.RMSE = Inf(k*nCross,dy);
resultsFolds.RRMSE = Inf(k*nCross,dy);
resultsFolds.R2 = -Inf(k*nCross,dy);
resultsFolds.nBasis = NaN(k*nCross,dy);
resultsFolds.nVars = zeros(k*nCross,dy);
resultsFolds.maxDeg = zeros(k*nCross,dy);
if evalPruning
    resultsPruning.GCV = NaN(k*nCross,1);
    resultsPruning.MSEoof = NaN(k*nCross,1);
    resultsPruning.meanGCV = NaN;
    resultsPruning.nBasisGCV = NaN;
    resultsPruning.meanMSEoof = NaN;
    resultsPruning.nBasisMSEoof = NaN;
    
    resultsPruning.R2GCV = NaN(k*nCross,1);
    resultsPruning.R2oof = NaN(k*nCross,1);
    resultsPruning.meanR2GCV = NaN;
    resultsPruning.nBasisR2GCV = NaN;
    resultsPruning.meanR2oof = NaN;
    resultsPruning.nBasisR2oof = NaN;
else
    resultsPruning = [];
end

% Repetition of Cross-Validation nCross times
for iCross = 1 : nCross
    [MAE, MSE, RMSE, RRMSE, R2, nBasis, nVars, maxDeg, evalResults] = ...
        doCV(X, Y, trainParams, k, shuffle, weights, testWithWeights, evalPruning, verbose, n, d, dy);
    to = iCross * k;
    range = (to - k + 1) : to;
    resultsFolds.MAE(range,:) = MAE;
    resultsFolds.MSE(range,:) = MSE;
    resultsFolds.RMSE(range,:) = RMSE;
    resultsFolds.RRMSE(range,:) = RRMSE;
    resultsFolds.R2(range,:) = R2;
    resultsFolds.nBasis(range,:) = nBasis;
    resultsFolds.nVars(range,:) = nVars;
    resultsFolds.maxDeg(range,:) = maxDeg;
    if evalPruning
        % Accomodate new data, even if bigger than expected
        sizeOld = size(resultsPruning.GCV,2);
        sizeNew = size(evalResults.GCV,2);
        if sizeOld < sizeNew
            add = NaN(size(resultsPruning.GCV,1),sizeNew-sizeOld);
            resultsPruning.GCV = [resultsPruning.GCV add];
            resultsPruning.MSEoof = [resultsPruning.MSEoof add];
            resultsPruning.R2GCV = [resultsPruning.R2GCV add];
            resultsPruning.R2oof = [resultsPruning.R2oof add];
        end
        resultsPruning.GCV(range,1:sizeNew) = evalResults.GCV;
        resultsPruning.MSEoof(range,1:sizeNew) = evalResults.MSEoof;
        resultsPruning.R2GCV(range,1:sizeNew) = evalResults.R2GCV;
        resultsPruning.R2oof(range,1:sizeNew) = evalResults.R2oof;
    end
end

% first mean across models, then mean across folds
resultsTotal.MAE = mean(mean(resultsFolds.MAE,2),1);
resultsTotal.MSE = mean(mean(resultsFolds.MSE,2),1);
resultsTotal.RMSE = mean(mean(resultsFolds.RMSE,2),1);
resultsTotal.RRMSE = mean(mean(resultsFolds.RRMSE,2),1);
resultsTotal.R2 = mean(mean(resultsFolds.R2,2),1);
resultsTotal.nBasis = mean(mean(resultsFolds.nBasis,2),1);
resultsTotal.nVars = mean(mean(resultsFolds.nVars,2),1);
resultsTotal.maxDeg = mean(mean(resultsFolds.maxDeg,2),1);
% mean across folds
if evalPruning
    resultsPruning.meanGCV = mean(resultsPruning.GCV,1);
    [~, resultsPruning.nBasisGCV] = min(resultsPruning.meanGCV);
    resultsPruning.meanMSEoof = mean(resultsPruning.MSEoof,1);
    [~, resultsPruning.nBasisMSEoof] = min(resultsPruning.meanMSEoof);
    % For columns with some NaNs, calculate mean if at least half of the values are not NaN
    nansSum = sum(isnan(resultsPruning.MSEoof),1);
    nansIdx = find((nansSum > 0) & (nansSum <= floor(size(resultsPruning.MSEoof,1) / 2)));
    for i = nansIdx
    	where = isnan(resultsPruning.MSEoof(:,i));
    	resultsPruning.meanMSEoof(i) = mean(resultsPruning.MSEoof(~where,i),1);
    end
    
    resultsPruning.meanR2GCV = mean(resultsPruning.R2GCV,1);
    [~, resultsPruning.nBasisR2GCV] = max(resultsPruning.meanR2GCV);
    resultsPruning.meanR2oof = mean(resultsPruning.R2oof,1);
    [~, resultsPruning.nBasisR2oof] = max(resultsPruning.meanR2oof);
    % For columns with some NaNs, calculate mean if at least half of the values are not NaN
    nansSum = sum(isnan(resultsPruning.R2oof),1);
    nansIdx = find((nansSum > 0) & (nansSum <= floor(size(resultsPruning.R2oof,1) / 2)));
    for i = nansIdx
    	where = isnan(resultsPruning.R2oof(:,i));
    	resultsPruning.meanR2oof(i) = mean(resultsPruning.R2oof(~where,i),1);
    end
end
return

function [MAE, MSE, RMSE, RRMSE, R2, nBasis, nVars, maxDeg, evalResults] = ...
    doCV(X, Y, trainParams, k, shuffle, weights, testWithWeights, evalPruning, verbose, n, d, dy)

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

MAE = Inf(k,dy);
MSE = Inf(k,dy);
RMSE = Inf(k,dy);
RRMSE = Inf(k,dy);
R2 = -Inf(k,dy);
nBasis = NaN(k,dy);
nVars = zeros(k,dy);
maxDeg = zeros(k,dy);

if evalPruning
    evalResults.GCV = NaN(k,1);
    evalResults.MSEoof = NaN(k,1);
    evalResults.R2GCV = NaN(k,1);
    evalResults.R2oof = NaN(k,1);
else
    dataEval = [];
    evalResults = [];
end

% perform training and testing k times
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
            Xtr(currsize+1:currsize+1+sizes(j)-1, :) = X(idxtrain, :);
            Ytr(currsize+1:currsize+1+sizes(j)-1, :) = Y(idxtrain, :);
            if ~isempty(weights)
                w(currsize+1:currsize+1+sizes(j)-1, 1) = weights(idxtrain, 1);
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
    if evalPruning
        dataEval.X = Xtst;
        dataEval.Y = Ytst;
        dataEval.weights = wtst;
    end
    [model, ~, res] = aresbuild(Xtr, Ytr, trainParams, w, false, [], dataEval, verbose);
    if evalPruning
        % Accomodate new data, even if bigger than expected
        sizeOld = size(evalResults.GCV,2);
        sizeNew = size(res.GCV,1);
        if sizeOld < sizeNew
            add = NaN(size(evalResults.GCV,1),sizeNew-sizeOld);
            evalResults.GCV = [evalResults.GCV add];
            evalResults.MSEoof = [evalResults.MSEoof add];
            evalResults.R2GCV = [evalResults.R2GCV add];
            evalResults.R2oof = [evalResults.R2oof add];
        end
        evalResults.GCV(i,1:sizeNew) = res.GCV;
        evalResults.MSEoof(i,1:sizeNew) = res.MSEtest;
        evalResults.R2GCV(i,1:sizeNew) = res.R2GCV;
        evalResults.R2oof(i,1:sizeNew) = res.R2test;
    end
    res = arestest(model, Xtst, Ytst, wtst);
    MAE(i,:) = res.MAE;
    MSE(i,:) = res.MSE;
    RMSE(i,:) = res.RMSE;
    RRMSE(i,:) = res.RRMSE;
    R2(i,:) = res.R2;
    if dy == 1
        modelCheck = model;
    else
        modelCheck = model{1}; % analyze only one model because all are identical
    end
    nBasis(i,:) = length(modelCheck.coefs);
    if ~isempty(modelCheck.knotdims)
        vars = [];
        deg = 0;
        for j = 1 : length(modelCheck.knotdims)
            vars = union(vars, modelCheck.knotdims{j});
            if length(modelCheck.knotdims{j}) > deg
                deg = length(modelCheck.knotdims{j});
            end
        end
        nVars(i,:) = length(vars);
        maxDeg(i,:) = deg;
    end
end
return
