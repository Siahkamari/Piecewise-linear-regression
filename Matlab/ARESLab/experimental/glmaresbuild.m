function glm = glmaresbuild(Xtr, Ytr, paramsOrModel, reduce, glmDistr, glmParams, verbose)
% glmaresbuild
% This is an experimental feature for ARESLab.
% Builds a Generalized Linear Model (GLM). By default, builds logit
% classification model (binary or multiclass).
% The algorithm: First builds an ARES model as usual, then invokes GLM on
% the ARES basis matrix model.X, and, optionally, reduces the GLM model
% using either the stepwise algorithm or lasso / elastic net / ridge
% regularization. Use the reduction techniques to deal with the sometimes
% occurring erratic behaviour, especially due to deletable ARES basis
% functions (symptoms include warning messages about perfect separability
% and reaching iteration limits).
% Currently, the function is tested only with logit and probit.
%
% Call:
%	glm = glmaresbuild(Xtr, Ytr, paramsOrModel, reduce, glmDistr, glmParams, verbose)
%
% All the input arguments, except the first three, are optional. Empty
% values are also accepted (the corresponding defaults will be used).
%
% Input:
%   Xtr, Ytr      : Training data. Xtr is a matrix with rows corresponding
%                   to observations and columns corresponding to input
%                   variables. Xtr type must be double. Ytr is a column
%                   vector of response values. Ytr type must be double,
%                   logical (internally converted to double), or cell array
%                   of strings (class labels). Categorical variables in Xtr
%                   with more than two categories must be replaced with
%                   synthetic binary variables before using glmaresbuild
%                   (or any other ARESLab function), for example using
%                   function dummyvar.
%   paramsOrModel : Either ARES parameters structure made using aresparams
%                   / aresparams2 or an already built ARES model (with
%                   keepX enabled).
%   reduce        : Set to 'off' (default), 'stepwise', or 'regularize'.
%                   The respective GLM-fitting functions called internally
%                   are fitglm, stepwiseglm, and lassoglm. See descriptions
%                   of those functions to understand arguments glmDistr and
%                   glmParams below.
%   glmDistr      : Distribution. Parameter to pass to the GLM-fitting
%                   function's argument Distribution (if reduce is 'off' or
%                   'stepwise') or argument distr (if reduce is
%                   'regularize'). Default value = 'binomial'.
%   glmParams     : Cell array of name/value pairs to pass to the
%                   GLM-fitting function (argument varargin). Default
%                   values for the parameters are set by the corresponding
%                   GLM-fitting functions, except that the following
%                   defaults are set by glmaresbuild:
%                   If reduce = 'off', then glmParams = {'link', 'logit',
%                   'Distribution', glmDistr}, meaning that a logit model
%                   is built.
%                   If reduce = 'stepwise', then glmParams = {'link',
%                   'logit', 'Distribution', glmDistr, 'Criterion', 'aic',
%                   'Lower', 'constant', 'Upper', 'linear'}, meaning that a
%                   logit model is built and then reduced using backward
%                   stepwise algorithm and AIC criterion.
%                   If reduce = 'regularize', then glmParams = {'link',
%                   'logit', 'CV', 10, 'NumLambda', 50, 'LambdaRatio',
%                   0.001}, meaning that a logit model is built and then
%                   reduced using lasso for which the value for lambda is
%                   chosen using 10-fold Cross-Validation from a list of 50
%                   values. If this option is too slow, it can be made
%                   faster, e.g., by lowering the number of Cross-
%                   Validation folds and/or NumLambda.
%                   Note that Cross-Validation employs random number
%                   generator for which you can set seed before calling
%                   glmaresbuild.
%   verbose       : Whether to output additional information to console
%                   (default value = true).
%
% Output:
%   glm           : A single ARES GLM model or a cell array of ARES GLM
%                   models for multiclass classification (one for each
%                   class). The structures keep all their ARES model fields
%                   and have additional fields starting with "glm".
%                   GLM coefficients are stored in glm.glmModel or
%                   glm.glmLassoB. Note that all other ARESLab functions
%                   without the prefix "glm" in their names are ignoring
%                   the new "glm" fields, therefore all their results are
%                   for vanilla ARES models, not GLM.
%
% Remarks:
% 1. For logit/probit modelling of multiclass data, internally, the
%    response with three or more levels is converted into the same number
%    of indicator columns of 1s and 0s. Then a multi-response ARES model is
%    build and, finally, a GLM model is built for each ARES model
%    separately.
% 2. glmaresbuild does not work with Octave or Matlab without the
%    Statistics and Machine Learning Toolbox (specifically, functions
%    fitglm, stepwiseglm, lassoglm, glmval, and predict). This can be
%    partly remedied, though, using the Econometrics Toolbox by James P.
%    LeSage (http://www.spatial-econometrics.com). Although, his logit
%    implementation seems to have a bug in calculation of yhat, residuals,
%    and sige.

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
if (nargin < 4) || isempty(reduce)
    reduce = 'off';
else
    if ~any(strcmpi(reduce, {'off' 'stepwise' 'regularize'}))
        error('Accepted values for argument reduce are ''off'', ''stepwise'', and ''regularize''.');
    end
end
if (nargin < 5) || isempty(glmDistr)
    glmDistr = 'binomial';
else
    if (nargin < 6) || isempty(glmParams) || ~any(strcmpi(glmParams, 'link'))
        % error, because otherwise we'll not always be sure which link to use
        error('If you set glmDistr, you also must set link type in glmParams.');
    end
end
if (nargin < 6) || isempty(glmParams)
    % setting logit link
    glmParams = {'link', 'logit'};
    isLinkLogitOrProbit = true;
else
    % setting logit link, if no link chosen
    if ~any(strcmpi(glmParams, 'link'))
        glmParams = [glmParams {'link', 'logit'}];
        isLinkLogitOrProbit = true;
    else
        isLinkLogitOrProbit = any(strcmpi(glmParams, 'logit')) || any(strcmpi(glmParams, 'probit'));
    end
end
if (nargin < 7) || isempty(verbose)
    verbose = true;
end

if isempty(Xtr) || isempty(Ytr)
    error('Training data is empty.');
end
if iscell(Xtr)
    error('Xtr should not be cell array.');
end
if islogical(Ytr)
    Ytr = double(Ytr);
elseif iscell(Ytr)
    if (~isLinkLogitOrProbit) || (~strcmpi(glmDistr, 'binomial'))
        error('Ytr may contain class labels as strings only for logit and probit.');
    end
elseif ~isfloat(Ytr)
    error('Ytr data type should be double, logical (internally converted to double), or cell array of strings.');
end
if ~isfloat(Xtr)
    error('Xtr data type should be double.');
end
if size(Ytr,2) ~= 1
    error('Ytr must have one column.');
end
if size(Ytr,1) ~= size(Xtr,1)
    error('The number of rows in Xtr and Ytr should be equal.');
end

% Settting additional defaults
if strcmpi(reduce, 'off')
    glmParams = [glmParams {'Distribution', glmDistr}];
elseif strcmpi(reduce, 'stepwise')
    glmParams = [glmParams {'Distribution', glmDistr}];
    if ~any(strcmpi(glmParams, 'Criterion'))
        glmParams = [glmParams {'Criterion', 'aic'}];
    end
    if ~any(strcmpi(glmParams, 'Lower'))
        glmParams = [glmParams {'Lower', 'constant'}];
    end
    if ~any(strcmpi(glmParams, 'Upper'))
        glmParams = [glmParams {'Upper', 'linear'}];
    end
    if (~verbose) && (~any(strcmpi(glmParams, 'Verbose')))
        glmParams = [glmParams {'Verbose', 0}];
    end
else
    if ~any(strcmpi(glmParams, 'CV'))
        glmParams = [glmParams {'CV', 10}];
    end
    if ~any(strcmpi(glmParams, 'NumLambda'))
        glmParams = [glmParams {'NumLambda', 50}];
    end
    if ~any(strcmpi(glmParams, 'LambdaRatio'))
        glmParams = [glmParams {'LambdaRatio', 0.001}]; % 0.0001 can make the function crazy slow
    end
end

if isLinkLogitOrProbit
    % we'll be doing classification
    classLabels = unique(Ytr); % find all the classes
    numClasses = numel(classLabels);
    if (numClasses > size(Ytr,1) / 2) || (numClasses > 30)
        error('Suspiciously large number of classes. Is it by mistake?');
    end
    if numClasses > 2
        % the response is multiclass
        % let's make indicator columns
        Ytmp = NaN(size(Ytr));
        if iscell(Ytr)
            for k = 1 : numClasses
                Ytmp(strcmpi(Ytr, classLabels(k))) = k;
            end
        else
            for k = 1 : numClasses
                Ytmp(Ytr == classLabels(k)) = k;
            end
        end
        Ytr = dummyvar(Ytmp);
        numModels = numClasses;
    else
        % the response is binary
        if iscell(Ytr)
            Ytr = strcmpi(Ytr, classLabels(2));
        else
            Ytr = Ytr == classLabels(2);
        end
        numModels = 1;
    end
else
    numModels = 1;
end

if (numel(paramsOrModel) == 1) && ~isfield(paramsOrModel, 'coefs')
    % build ARES model(s) if parameters are provided
    paramsOrModel = aresbuild(Xtr, Ytr, paramsOrModel, [], true, [], [], verbose);
else
    if ~isfield(paramsOrModel, 'X')
        error('Field paramsOrModel.X is missing.');
    end
end

% Setting names for basis functions
if strcmpi(reduce, 'off') || strcmpi(reduce, 'stepwise')
    if numModels == 1
        varNames = strcat('BF', strread(num2str(1:length(paramsOrModel.knotdims)),'%s')');
    else
        varNames = strcat('BF', strread(num2str(1:length(paramsOrModel{1}.knotdims)),'%s')');
    end
    glmParams = [glmParams {'VarNames', [varNames 'y']}];
end

if numModels == 1
    if verbose, disp('Building ARES GLM model...'); end;
    glm = doFit(Ytr, paramsOrModel, reduce, glmDistr, glmParams, verbose);
    if isLinkLogitOrProbit
        glm.glmClassLabels = classLabels;
    end
else
    % this is for multiclass logit/probit models only
    glm = cell(numModels,1);
    for k = 1 : numModels
        if verbose, disp(['Building ARES GLM model #' int2str(k) '...']); end;
        glm{k} = doFit(Ytr(:,k), paramsOrModel{k}, reduce, glmDistr, glmParams, verbose);
        glm{k}.glmClassLabels = classLabels;
    end
end
if verbose, disp('Done.'); end;
return

function model = doFit(Ytr, model, reduce, glmDistr, glmParams, verbose)
% function for fitting GLM
model.glmReduce = reduce;
model.glmDistr = glmDistr;
model.glmParams = glmParams;
if strcmpi(reduce, 'off')
    %[~, model.glmDev, model.glmStats] = glmfit(model.X(:,2:end), Ytr, glmDistr, glmParams{:}); % for the older implementation of GLM
    model.glmModel = fitglm(model.X(:,2:end), Ytr, 'linear', glmParams{:});
elseif strcmpi(reduce, 'stepwise')
    % turning off some warnings as they can create massive spam
    origWarningState = warning;
    warning('off', 'stats:glmfit:IterationLimit');
    warning('off', 'stats:glmfit:PerfectSeparation');
    model.glmModel = stepwiseglm(model.X(:,2:end), Ytr, 'linear', glmParams{:});
    warning(origWarningState);
    % Info
    if verbose
        disp(['Basis functions used in the final GLM model:' ...
            cell2mat(strcat({' '}, model.glmModel.Coefficients.Properties.RowNames'))]);
    end
else
    [model.glmLassoB, model.glmLassoFitInfo] = lassoglm(model.X(:,2:end), Ytr, glmDistr, glmParams{:});
    % Info
    if verbose
        %idx = glm.glmLassoFitInfo.IndexMinDeviance; % minimum deviance
        idx = model.glmLassoFitInfo.Index1SE; % minimum deviance plus one standard deviation
        idx = find(model.glmLassoB(:,idx) ~= 0);
        str = 'Basis functions used in the final GLM model: (Intercept)';
        if ~isempty(idx)
            str = [str cell2mat(strcat(' BF', strread(num2str(idx'),'%s')'))];
        end
        disp(str);
    end
end
return
