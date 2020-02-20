function varImp = aresimp(model, Xtr, Ytr, resultsEval, weights)
% aresimp
% Performs input variable importance assessment and reports the results.
% For details, see remarks on aresimp in user's manual.
% For multi-response modelling, supply one submodel at a time.
%
% Call:
%   varImp = aresimp(model, Xtr, Ytr, resultsEval, weights)
%
% The first three input arguments are required.
%
% Input:
%   model         : ARES model.
%   Xtr, Ytr      : Training data observations. The same data that was used
%                   when the model was built.
%   resultsEval   : resultsEval from function aresbuild. Do not use this
%                   argument if model was modified by any function other
%                   than aresbuild (i.e., aresdel or aresanovareduce).
%   weights       : A vector of weights for observations. The same weights
%                   that were used when the model was built.
%
% Output:
%   varImp        : A matrix of estimated variable importance. Rows
%                   correspond to input variables, columns correspond to
%                   criterion used. If argument resultsEval is not supplied
%                   or the model was not pruned, then 2nd, 3rd, and 4th
%                   columns are NaN.

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
    resultsEval = [];
end
if (nargin < 5)
    weights = [];
else
    if (~isempty(weights)) && ...
       ((size(weights,1) ~= n) || (size(weights,2) ~= 1))
        error('weights vector is of wrong size.');
    end
end

nVars = length(model.minX);
nBasisExI = length(model.knotdims); % excluding intercept

fprintf('Estimated input variable importance:\n');
if isempty(resultsEval)
    fprintf('(columns nSubsets, subsRSS, and subsGCV not available with empty argument resultsEval)\n');
end
fprintf('Variable    delGCV');
if ~isempty(resultsEval)
    fprintf('      nSubsets       subsRSS       subsGCV');
end
fprintf('\n');

varImp = NaN(nVars,4);

% The first criterion
for v = 1 : nVars
    funcsToDel = [];
    for i = 1 : nBasisExI
        if any(model.knotdims{i} == v)
            funcsToDel = [funcsToDel i];
        end
    end
    if ~isempty(funcsToDel)
        modelReduced = aresdel(model, funcsToDel, Xtr, Ytr, weights);
        varImp(v,1) = sqrt(modelReduced.GCV) - sqrt(model.GCV);
    else
        varImp(v,1) = 0;
    end
end
if max(varImp(:,1)) ~= 0
    varImp(:,1) = varImp(:,1) ./ max(varImp(:,1)) * 100; % scaling
end

% The other three criteria
if ~isempty(resultsEval)
    finalSize = length(model.knotdims);
    varImp(:,2) = sum(resultsEval.usedVars(2:finalSize,:),1);
    
    reduction = resultsEval.MSE(1:(finalSize-1)) - resultsEval.MSE(2:finalSize);
    for v = 1 : nVars
        varImp(v,3) = sum(reduction(resultsEval.usedVars(2:finalSize,v)));
    end
    if max(varImp(:,3)) ~= 0
        varImp(:,3) = varImp(:,3) ./ max(varImp(:,3)) * 100; % scaling
    end
    
    reduction = resultsEval.GCV(1:(finalSize-1)) - resultsEval.GCV(2:finalSize);
    for v = 1 : nVars
        varImp(v,4) = sum(reduction(resultsEval.usedVars(2:finalSize,v)));
    end
    if max(varImp(:,4)) ~= 0
        varImp(:,4) = varImp(:,4) ./ max(varImp(:,4)) * 100; % scaling
    end
end

% Printing results
for v = 1 : nVars
    counterStr = num2str(v);
    counterStr = [counterStr repmat(' ', 1, 4-length(counterStr))];
    fprintf(counterStr);
    fprintf('%14.3f', varImp(v,1));
    if ~isempty(resultsEval)
        fprintf('%14d%14.3f%14.3f', varImp(v,2), varImp(v,3), varImp(v,4));
        if ~any(resultsEval.usedVars(2:finalSize,v))
            fprintf('        unused');
        end
    end
    fprintf('\n');
end

return
