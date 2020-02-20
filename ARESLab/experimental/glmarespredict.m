function [Yq, Yraw] = glmarespredict(glm, Xq, glmParams)
% glmarespredict
% This is an experimental feature for ARESLab.
% Predicts response values for the given query points Xq using ARES GLM model.
% Currently, the function is tested only with logit and probit.
%
% Call:
%   [Yq, Yraw] = glmarespredict(glm, Xq, glmParams)
%
% Input:
%   glm           : A single ARES GLM model or a cell array of ARES GLM
%                   models for multiclass classification.
%   Xq            : A matrix of query data points.
%   glmParams     : Optional. Cell array of name/value pairs to pass to
%                   function predict or glmval (argument varargin). Default
%                   value = {}.
%
% Output:
%   Yq            : Predicted response values for each Xq row. For logit /
%                   probit models, Yq contains predicted class labels.
%   Yraw          : Raw logit/probit response values, calculated by each
%                   separate model. Available only for logit/probit models.

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

if nargin < 2
    error('Not enough input arguments.');
end
if (nargin < 3) || isempty(glmParams)
    glmParams = {};
end
if iscell(Xq)
    error('Xq should not be cell array.');
end
if ~isfloat(Xq)
    error('Xq data type should be double.');
end

numModels = length(glm);
if ((numModels == 1) && (length(glm.minX) ~= size(Xq,2))) || ...
   ((numModels > 1) && (length(glm{1}.minX) ~= size(Xq,2)))
    error('The number of columns in Xq is different from the number when the model was built.');
end

% Getting the link type
if numModels == 1
    idx = find(strcmpi(glm.glmParams, 'link'));
    glmLink = glm.glmParams{idx+1};
else
    idx = find(strcmpi(glm{1}.glmParams, 'link'));
    glmLink = glm{1}.glmParams{idx+1};
end
linkLogitOrProbit = any(strcmpi(glmLink, {'logit' 'probit'}));

if numModels == 1
    [~, BX] = arespredict(glm, Xq); % get basis matrix from ARES model
    Yq = doPredict(glm, BX, glmLink, glmParams); % predict using GLM
    if linkLogitOrProbit
        Yraw = Yq;
        Yq = glm.glmClassLabels((Yraw >= 0.5) + 1); % replace with class labels
    else
        Yraw = [];
    end
else
    % this is for multiclass logit/probit only
    [~, BX] = arespredict(glm{1}, Xq); % get basis matrix from ARES model
    Yraw = NaN(size(Xq,1),numModels);
    for k = 1 : numModels
        Yraw(:,k) = doPredict(glm{k}, BX, glmLink, glmParams); % predict using GLM
    end
    [~, Yq] = max(Yraw, [], 2);
    Yq = glm{1}.glmClassLabels(Yq); % replace with class labels
end
return

function Yq = doPredict(glm, X, glmLink, glmParams)
% function for prediction using GLM
if strcmpi(glm.glmReduce, 'off')
    %Yq = glmval(glm.glmStats.beta, X(:,2:end), glmLink, glmvalParams{:}); % for the older implementation of GLM
    Yq = predict(glm.glmModel, X(:,2:end), glmParams{:});
elseif strcmpi(glm.glmReduce, 'stepwise')
    Yq = predict(glm.glmModel, X(:,2:end), glmParams{:});
else
    %idx = glm.glmLassoFitInfo.IndexMinDeviance; % minimum deviance
    idx = glm.glmLassoFitInfo.Index1SE; % minimum deviance plus one standard deviation
    Yq = glmval([glm.glmLassoFitInfo.Intercept(idx);glm.glmLassoB(:,idx)], X(:,2:end), glmLink, glmParams{:});
end
return
