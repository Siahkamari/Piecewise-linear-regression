function [model, time, resultsEval] = aresbuild(Xtr, Ytr, trainParams, ...
    weights, keepX, modelOld, dataEval, verbose)
% aresbuild
% Builds a regression model using the Multivariate Adaptive Regression
% Splines method.
%
% Call:
%   [model, time, resultsEval] = aresbuild(Xtr, Ytr, trainParams, ...
%       weights, keepX, modelOld, dataEval, verbose)
%
% All the input arguments, except the first two, are optional. Empty values
% are also accepted (the corresponding defaults will be used).
%
% Input:
%   Xtr, Ytr      : Xtr is a matrix with rows corresponding to
%                   observations and columns corresponding to input
%                   variables. Ytr is either a column vector of response
%                   values or, for multi-response data, a matrix with
%                   columns corresponding to response variables. The
%                   structure of the output of this function changes
%                   depending on whether Ytr is a vector or a matrix (see
%                   below).
%                   Xtr type must be double. Ytr type must be double or
%                   logical (internally converted to double). Categorical
%                   variables in Xtr with more than two categories must be
%                   replaced with synthetic binary variables before using
%                   aresbuild (or any other ARESLab function), for example
%                   using function dummyvar.
%                   For multi-response data, each model will have the same
%                   set of basis functions but different coefficients. The
%                   models are built and pruned as usual but with the
%                   Residual Sum of Squares and GCVs summed across all
%                   responses. Since all the models are optimized
%                   simultaneously, the results for each model won't be as
%                   good as building the models independently. However, the
%                   combined model may be better in other senses, depending
%                   on what you are trying to achieve. For example, it
%                   could be useful to select the set of basis functions
%                   that is best across all responses.
%                   It is recommended to pre-scale Xtr values to [0,1]
%                   (Friedman, 1991a). This is because widely different
%                   locations and scales for the input variables can cause
%                   instabilities that could affect the quality of the
%                   final model. The MARS method is (except for
%                   numerics) invariant to the locations and scales of the
%                   input variables. It is therefore reasonable to perform
%                   a transformation that causes resulting locations and
%                   scales to be most favourable from the point of view of
%                   numeric stability (Friedman, 1991a).
%                   For multi-response modelling, it is recommended to
%                   pre-scale Ytr values so that each response variable
%                   gets the appropriate weight during model building. A
%                   variable with higher variance will influence the
%                   results more than a variable with lower variance
%                   (Milborrow, 2016).
%   trainParams   : A structure of training parameters for the algorithm.
%                   If not provided, default values will be used (see
%                   function aresparams for details).
%   weights       : A vector of observation weights. The length of the
%                   vector must be the same as the number of observations
%                   in Xtr and Ytr. The weights must be nonnegative.
%   keepX         : Set to true to retain basis matrix model.X (see
%                   description of model.X). For multi-response modelling,
%                   the matrix will be replicated for each model. (default
%                   value = false)
%   modelOld      : If an already built ARES model is provided (whether
%                   pruned or not), no forward phase will be done. Instead
%                   the provided model will be taken directly to the
%                   backward phase and pruned. This is useful for fast
%                   tuning of parameters of the backward phase (c, cubic,
%                   maxFinalFuncs). Note that this is also a much faster
%                   way of changing a piecewise-linear model into a
%                   piecewise-cubic model or vice versa instead of building
%                   a new model from scratch. This argument is also used by
%                   function arescvc for fast selection of the "best" value
%                   for penalty c using Cross-Validation.
%   dataEval      : A structure containing test data in fields X, Y, and,
%                   optionally, weights. Used for getting evaluations for
%                   the best candidate models of each size in the backward
%                   pruning phase. For example, arescv uses it to help
%                   choosing a good value for the number of basis functions
%                   using Cross-Validation (see example of usage in user's
%                   manual Section 3.3). Results are saved in fields R2test
%                   and MSEtest of output argument resultsEval.
%   verbose       : Whether to output additional information to console
%                   (default value = true).
%
% Output:
%   model         : A single ARES model for single-response Ytr or a cell
%                   array of ARES models for multi-response Ytr. A
%                   structure defining one model has the following fields:
%     coefs       : Coefficients vector of the regression model (first, for
%                   the intercept term, and then for all the rest of basis
%                   functions). Because of the coefficient for the
%                   intercept term, this vector is one row longer than the
%                   others.
%     knotdims	  : Cell array of indices of used input variables for knots
%                   in each basis function.
%     knotsites	  : Cell array of knot sites for each knot and used input
%                   variable in each basis function. knotdims and knotsites
%                   together contain all the information for locating the
%                   knots. If a variable entered a basis function linearly
%                   (i.e., without hinge function), the knot site for that
%                   variable is set to minX.
%     knotdirs	  : Cell array of directions (-1 or 1) of the hinge
%                   functions for each used input variable in each basis
%                   function. If a variable entered a basis function
%                   linearly (i.e., without hinge function), the direction
%                   for that variable is set to 2.
%     parents     : Vector of indices of direct parents for each basis
%                   function (0 if there is no direct parent).
%     trainParams : A structure of training parameters for the algorithm.
%                   The values are updated if chosen automatically. Except
%                   useMinSpan, because in automatic mode it is calculated
%                   for each parent basis function separately.
%     MSE         : Mean Squared Error of the model in the training data.
%     GCV         : Generalized Cross-Validation of the model in the
%                   training data set. The value may also be Inf if model�s
%                   effective number of parameters is larger than or equal
%                   to the number of observations in the training data.
%     t1, t2      : Matrix of sites for the additional side knots on the
%                   left and on the right of the central knot.
%     minX, maxX  : Vectors defining the ranges of the input variables
%                   determined from the training data.
%     isBinary    : A vector indicating binary input variables. Determined
%                   automatically by counting unique values for each
%                   variable in training data. Therefore a variable can
%                   also be taken as binary by mistake if the data for some
%                   reason included only two values for the variable. Note
%                   that whether a variable is binary does not influence
%                   building of the model. This vector is further used in
%                   other functions to simplify printed equations.
%     X           : Basis matrix. Contains values of basis functions
%                   applied to Xtr. The number of columns in X is equal to
%                   the number of rows in coefs, i.e., the first column is
%                   for the intercept (all ones) and all the other columns
%                   correspond to the basis functions defined by knotdims,
%                   knotsites, knotdirs, t1, and t2. Each row corresponds
%                   to a row in Xtr. Multiplying X by coefs gives ARES
%                   prediction for Ytr. This variable is available only if
%                   argument keepX is set to true.
%   time          : Algorithm execution time (in seconds).
%   resultsEval   : Model evaluation results from the backward pruning
%                   phase. Fields R2test and MSEtest are available only if
%                   input argument dataEval is not empty. The structure has
%                   the following fields:
%     MSE         : MSE (Mean Squared Error) in training data for the best
%                   candidate model of each size.
%     R2          : R2 (Coefficient of Determination) in training data for
%                   the best candidate model of each size.
%     GCV         : GCV (Generalized Cross-Validation) in training data for
%                   the best candidate model of each size. Contains Inf
%                   values for models with effective number of parameters
%                   larger than the number of observations in training
%                   data.
%     R2GCV       : R2 estimated by GCV in training data for the best
%                   candidate model of each size. Contains -Inf values for
%                   models with effective number of parameters larger than
%                   the number of observations in training data.
%     R2test      : R2 in dataEval test data for the best candidate model
%                   of each size.
%     MSEtest     : MSE in dataEval test data for the best candidate model
%                   of each size.
%                   Note that if trainParams.cubic = true, values of these
%                   fields are calculated using piecewise-linear models if
%                   trainParams.cubicFastLevel = 2 and piecewise-cubic
%                   models if trainParams.cubicFastLevel < 2.
%     usedVars    : Logical matrix showing which input variables were used
%                   in the best candidate model of each size.

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

% Citing the ARESLab toolbox:
% Jekabsons G., ARESLab: Adaptive Regression Splines toolbox for Matlab/
% Octave, 2016, available at http://www.cs.rtu.lv/jekabsons/

% Last update: May 15, 2016

if nargin < 2
    error('Not enough input arguments.');
end

if isempty(Xtr) || isempty(Ytr)
    error('Training data is empty.');
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
[ny, dy] = size(Ytr); % number of observations and number of output variables
if ny ~= n
    error('The number of rows in Xtr and Ytr should be equal.');
end
if (ny == 1) && (dy > 1)
    warning('Ytr has one row but more than one column.');
end

if (nargin < 3) || isempty(trainParams)
    trainParams = aresparams();
end
if (trainParams.cubic) && (trainParams.selfInteractions > 1)
    trainParams.selfInteractions = 1;
    warning('trainParams.selfInteractions value changed to 1 due to piecewise-cubic setting.');
end
if trainParams.maxInteractions < 0
    trainParams.maxInteractions = d; % for maximal interactivity (except if selfInteractions are used)
end
if trainParams.cubic
    doCubicFastLevel = trainParams.cubicFastLevel;
    if trainParams.cubicFastLevel > 0
        trainParams.cubic = false; % let's turn it off until the backward phase or the final model
    end
else
    doCubicFastLevel = -1; % no piecewise-cubic modelling
end
if (trainParams.endSpanAdjust < 1)
    trainParams.endSpanAdjust = 1;
    warning('trainParams.endSpanAdjust value too small. Changed to 1.');
end

if (trainParams.fastK < 3)
    trainParams.fastK = 3;
    warning('trainParams.fastK value too small. Changed to 3.');
end
if (trainParams.fastBeta < 0)
    trainParams.fastBeta = 0;
    warning('trainParams.fastBeta value too small. Changed to 0.');
end
if (trainParams.fastH < 1)
    trainParams.fastH = 1;
    warning('trainParams.fastH value too small. Changed to 1.');
end

if trainParams.useMinSpan == 0
    trainParams.useMinSpan = 1; % 1 and 0 is the same here (no minspan)
end

if (nargin < 4)
    weights = [];
else
    if (~isempty(weights)) && ...
       ((size(weights,1) ~= n) || (size(weights,2) ~= 1))
        error('weights vector is of wrong size.');
    end
    sumWeights = sum(weights);
end
if (nargin < 5) || isempty(keepX)
    keepX = false;
end
if nargin < 6
    modelOld = [];
else
    if ~isempty(modelOld)
        if (dy > 1) && (length(modelOld) ~= dy)
            error('modelOld should contain as many models as there are columns in Ytr.');
        end
        if ((dy == 1) && (length(modelOld.minX) ~= d)) || ...
           ((dy > 1) && (length(modelOld{1}.minX) ~= d))
            error('The number of columns in Xtr is different from the number when the modelOld was built.');
        end
    end
end
if (nargin < 7)
    dataEval = [];
end
if (nargin < 8) || isempty(verbose)
    verbose = true;
end

requestingResultsEval = nargout >= 3;
resultsEval = [];

if ~isempty(trainParams.noInteract)
    if ~isempty(trainParams.yesInteract)
        error('trainParams.noInteract and trainParams.yesInteract cannot be used at the same time.');
    end
    if size(trainParams.noInteract,2) ~= 2
        error('trainParams.noInteract should have two columns.');
    end
    trainParams.noInteract = [trainParams.noInteract; trainParams.noInteract(:,2) trainParams.noInteract(:,1)];
    trainParams.noInteract = unique(trainParams.noInteract, 'rows');
elseif ~isempty(trainParams.yesInteract)
    if size(trainParams.yesInteract,2) ~= 2
        error('trainParams.yesInteract should have two columns.');
    end
    trainParams.yesInteract = [trainParams.yesInteract; trainParams.yesInteract(:,2) trainParams.yesInteract(:,1)];
    trainParams.yesInteract = unique(trainParams.yesInteract, 'rows');
end

if trainParams.maxFuncs >= 0
    maxIters = floor((trainParams.maxFuncs - 1) / 2); % because basis functions are added two at a time
else
    trainParams.maxFuncs = min(200, max(20, 2 * d)) + 1; % default number of basis functions
    maxIters = floor(trainParams.maxFuncs / 2); % because basis functions are added two at a time
    if verbose, fprintf('Setting trainParams.maxFuncs to %d\n', trainParams.maxFuncs); end
end

if (trainParams.c < 0)
    if trainParams.maxInteractions > 1
        trainParams.c = 3;
    else
        trainParams.c = 2; % penalty coefficient for additive modelling
    end
    if verbose, fprintf('Setting trainParams.c to %d\n', trainParams.c); end
end

if ~any([0 1 2] == trainParams.allowLinear)
    error('Expected values for trainParams.allowLinear are 0, 1 or 2.');
end

if verbose, fprintf('Building ARES model...\n'); end
ttt = tic;

if isempty(weights)
    YtrMean = mean(Ytr,1);
else
    YtrMean = zeros(1,dy);
    for k = 1 : dy
        YtrMean(k) = sum(Ytr(:,k) .* weights) / sumWeights; % Ytr weighted mean
    end
end
YtrVarN = zeros(1,dy);
if isempty(weights)
    for k = 1 : dy
        YtrVarN(k) = sum((Ytr(:,k) - YtrMean(k)) .^ 2); % Ytr variance * n (i.e., null SSE)
    end
    YtrVar = YtrVarN / n; % Ytr variance (i.e., null MSE)
else
    for k = 1 : dy
        YtrVarN(k) = sum(((Ytr(:,k) - YtrMean(k)) .^ 2) .* weights); % (i.e., null SSE)
    end
    YtrVar = YtrVarN / sumWeights; % Ytr weighted variance (i.e., null MSE)
end

minX = min(Xtr);
maxX = max(Xtr);
if isempty(modelOld)
    isBinary = [];
else
    if dy == 1
        isBinary = modelOld.isBinary;
    else
        isBinary = modelOld{1}.isBinary;
    end
end

if trainParams.useEndSpan < 0
    trainParams.useEndSpan = getEndSpan(d); % automatic
end

if dy > 1
    modelsY = cell(dy, 1);
end

if isempty(modelOld)
    X = ones(n,1);
    err = 1; % normalized error for the constant model
    model.MSE = Inf;
    model.GCV = Inf;
    if dy == 1
        model.coefs = YtrMean;
    else
        model.coefs = YtrMean(1);
        for k = 1 : dy
            modelsY{k}.coefs = YtrMean(k);
            modelsY{k}.MSE = Inf;
            modelsY{k}.GCV = Inf;
        end
    end
    model.knotdims = {};
    model.knotsites = {};
    model.knotdirs = {};
    model.parents = [];
    model.trainParams = [];
else
    % modelOld is the initial model
    if dy == 1
        model = modelOld;
    else
        model = modelOld{1};
        for k = 1 : dy
            modelsY{k}.coefs = modelOld{k}.coefs;
            modelsY{k}.MSE = modelOld{k}.MSE;
            modelsY{k}.GCV = modelOld{k}.GCV;
        end
    end
end

origWarningState = warning;
if exist('OCTAVE_VERSION', 'builtin')
    warning('off', 'Octave:nearly-singular-matrix');
    warning('off', 'Octave:singular-matrix');
else
    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
end

if trainParams.useEndSpan * 2 >= n
    warning('trainParams.useEndSpan * 2 >= n');
    if isempty(modelOld)
        isBinary = false(1,d);
        if dy == 1
            model.MSE = YtrVar;
            model.GCV = gcv(length(model.coefs), model.MSE, n, trainParams.c);
        else
            model.MSE = 0;
            model.GCV = 0;
            for k = 1 : dy
                modelsY{k}.MSE = YtrVar(k);
                modelsY{k}.GCV = gcv(length(model.coefs), modelsY{k}.MSE, n, trainParams.c);
                model.MSE = model.MSE + modelsY{k}.MSE;
                model.GCV = model.GCV + modelsY{k}.GCV;
            end
        end
        if trainParams.cubic
            model.t1 = [];
            model.t2 = [];
        end
    end
else

    % FORWARD PHASE

    if isempty(modelOld) % no forward phase when modelOld is used
        
        if verbose && (maxIters < 1)
            fprintf('Forward phase  .');
        end
        
        % create sorted lists of observations for knot placements
        [sortedXtr, sortedXtrInd] = sort(Xtr);
        % check whether any input variable could be binary
        isBinary = false(1,d);
        for i = 1 : d
            %isBinary(i) = all((sortedXtr(:,i) == sortedXtr(1,i)) | (sortedXtr(:,i) == sortedXtr(end,i)));
            isBinary(i) = numel(unique(sortedXtr(:,i))) == 2;
        end
        % is endSpan is used, throw away observations at the ends of the intervals
        if trainParams.useEndSpan ~= 0
            sortedXtr = sortedXtr(1+trainParams.useEndSpan:end-trainParams.useEndSpan,:);
            sortedXtrInd = sortedXtrInd(1+trainParams.useEndSpan:end-trainParams.useEndSpan,:);
        end
        
        if verbose
            sizeInfGCV = NaN;
            countKnots = zeros(1,d);
        end
        
        if trainParams.cubic
            tmp_t1 = [];
            tmp_t2 = [];
        end
        basisFunctionList = {}; % will contain candidate basis functions
        numNewFuncs = 0; % how many basis functions added in the last iteration
        if (trainParams.newVarPenalty > 0)
            dimsInModel = []; % lists all input variables the model uses
        end
        
        if (trainParams.fastK < Inf)
            % we could preallocate space here but benchmarking shows that
            % there is not much to be gained from this
            fastParent = {};
            fastMinErrAdj = [];
            fastIterComputedErr = [];
            fastNumBasis = 0;
        end
        
        % the main loop of the forward phase
        for currIter = 1 : maxIters
            % create list of all possible daughter basis functions
            [basisFunctionList, idxStart1, idxEnd1, idxStart2, idxEnd2] = ...
                    createList(basisFunctionList, Xtr, sortedXtr, sortedXtrInd, ...
                               n, d, model, numNewFuncs, trainParams, minX, maxX);
            
            % stop the forward phase if basisFunctionList is empty
            if isempty(basisFunctionList)
                if trainParams.cubic
                    t1 = tmp_t1;
                    t2 = tmp_t2;
                end
                if verbose
                    if (currIter == 1)
                        fprintf('Forward phase  .');
                    end
                    fprintf('\nTermination condition is met: no more basis functions to add.');
                end
                break;
            end
            
            % count the number of knot locations for the basis functions
            if verbose && (currIter == 1)
                for i = 1 : size(basisFunctionList,2)
                    countKnots(1,basisFunctionList{1,i}) = countKnots(1,basisFunctionList{1,i}) + 1;
                end
                fprintf('Approx number of available knot locations (controlled by useMinSpan and useEndSpan):%s\n', ...
                    sprintf(' x%d:%d', [1:length(countKnots); countKnots]));
                fprintf('Forward phase  .');
            end
            
            % preallocate space
            % (we could allocate less but this is simpler)
            tmpErr = inf(1,size(basisFunctionList,2));
            if dy == 1
                tmpCoefs = inf(length(model.coefs)+2, size(basisFunctionList,2));
            else
                tmpErrVar = zeros(1,size(basisFunctionList,2));
                tmpCoefs = cell(dy, 1);
                for k = 1 : dy
                    tmpCoefs{k} = inf(length(model.coefs)+2, size(basisFunctionList,2));
                end
            end
            if trainParams.cubic
                Xtmp = zeros(n,size(X,2)+2);
            else
                Xtmp = [X zeros(n,2)];
            end
            
            if (trainParams.fastK < Inf) % using the Fast MARS algorithm
                
                % initialize info about the two new parent basis functions
                if ~isempty(basisFunctionList)
                    if (idxStart1 > 0)
                        fastNumBasis = fastNumBasis + 1;
                        fastParent{fastNumBasis}.idxStart = idxStart1;
                        fastParent{fastNumBasis}.idxEnd = idxEnd1;
                        fastMinErrAdj(fastNumBasis) = -Inf;
                        fastIterComputedErr(fastNumBasis) = currIter;
                        fastParent{fastNumBasis}.iterComputedAllDims = -Inf;
                    end
                    if (idxStart2 > 0)
                        fastNumBasis = fastNumBasis + 1;
                        fastParent{fastNumBasis}.idxStart = idxStart2;
                        fastParent{fastNumBasis}.idxEnd = idxEnd2;
                        fastMinErrAdj(fastNumBasis) = -Inf;
                        fastIterComputedErr(fastNumBasis) = currIter;
                        fastParent{fastNumBasis}.iterComputedAllDims = -Inf;
                    end
                end
                
                % make a selection of parent basis functions to try
                lenR = length(fastParent);
                if lenR > trainParams.fastK
                    % ranking and prioritizing
                    [~, idxR] = sort(fastMinErrAdj, 2, 'descend'); % sort by err
                    idxR(idxR) = 1:lenR; % get ranks in the original order
                    idxInf = fastMinErrAdj == -Inf; % special case
                    idxR(idxInf) = idxR(idxInf) + 1E6; % so that -Inf is never outprioritized
                    priority = idxR + trainParams.fastBeta * (repmat(currIter, 1, lenR) - fastIterComputedErr); % calculate priorities
                    [~, idxP] = sort(priority, 2); % sort (with equal priorities, most recent will be first)
                    indToCalc = idxP(end:-1:end-trainParams.fastK+1); % indices of parents to try
                else
                    indToCalc = 1 : lenR;
                end
                fastIterComputedErr(indToCalc) = currIter;
                
                bestParent = -1;
                bestErrAdj = Inf;
                % try the selected parent basis functions
                for ir = indToCalc
                    % let's see if only a preselected dimension should be checked
                    checkPreselectedDimOnly = currIter - fastParent{ir}.iterComputedAllDims < trainParams.fastH;
                    if (~checkPreselectedDimOnly)
                        fastParent{ir}.iterComputedAllDims = currIter;
                    end
                    idxStart = fastParent{ir}.idxStart;
                    idxEnd = fastParent{ir}.idxEnd;
                    % try all the reflected pairs (daughter basis functions) in the list
                    for i = idxStart : idxEnd
                        if  checkPreselectedDimOnly && (basisFunctionList{1,i}(end) ~= fastParent{ir}.bestDim)
                            %tmpErr(i) = Inf; % unnecessary because the array is already initialized with Inf
                            continue;
                        end
                        if trainParams.cubic
                            [t1, t2, diff] = findsideknots(model, basisFunctionList{1,i}, basisFunctionList{2,i}, ...
                                            d, minX, maxX, tmp_t1, tmp_t2);
                            Xtmp(:,1:end-2) = X;
                            % update basis functions with the updated side knots
                            for j = diff
                                Xtmp(:,j+1) = createbasisfunction(Xtr, Xtmp, model.knotdims{j}, model.knotsites{j}, ...
                                              model.knotdirs{j}, model.parents(j), minX, maxX, t1(j,:), t2(j,:));
                            end
                            % New basis function
                            dirs = basisFunctionList{3,i};
                            Xtmp(:,end-1) = createbasisfunction(Xtr, Xtmp, basisFunctionList{1,i}, basisFunctionList{2,i}, ...
                                            dirs, basisFunctionList{4,i}, minX, maxX, t1(end,:), t2(end,:));
                            ok1 = ~isnan(Xtmp(1,end-1));
                            % Reflected partner
                            dirs(end) = -dirs(end);
                            Xtmp(:,end) = createbasisfunction(Xtr, Xtmp, basisFunctionList{1,i}, basisFunctionList{2,i}, ...
                                          dirs, basisFunctionList{4,i}, minX, maxX, t1(end,:), t2(end,:));
                            ok2 = ~isnan(Xtmp(1,end));
                        else
                            % New basis function
                            dirs = basisFunctionList{3,i};
                            Xtmp(:,end-1) = createbasisfunction(Xtr, Xtmp, basisFunctionList{1,i}, ...
                                            basisFunctionList{2,i}, dirs, basisFunctionList{4,i}, minX, maxX);
                            ok1 = ~isnan(Xtmp(1,end-1));
                            % Reflected partner
                            dirs(end) = -dirs(end);
                            Xtmp(:,end) = createbasisfunction(Xtr, Xtmp, basisFunctionList{1,i}, ...
                                          basisFunctionList{2,i}, dirs, basisFunctionList{4,i}, minX, maxX);
                            ok2 = ~isnan(Xtmp(1,end));
                        end
                        if ok1 && ok2 % both basis functions created
                            if dy == 1
                                [tmpCoefs(:,i), tmpErr(i)] = lreg(Xtmp, Ytr, weights);
                            else
                                tmpErr(i) = 0;
                                tmpErrVar(i,:) = 0;
                                for k = 1 : dy
                                    [tmpCoefs{k}(:,i), sse] = lreg(Xtmp, Ytr(:,k), weights);
                                    tmpErr(i) = tmpErr(i) + sse;
                                    tmpErrVar(i) = tmpErrVar(i) + sse / YtrVarN(k);
                                end
                                %tmpErr(i) = tmpErr(i) / dy;
                            end
                        elseif ok1 || ok2 % one of the basis functions not created
                            if dy == 1
                                if (ok1)
                                    [coefs, tmpErr(i)] = lreg(Xtmp(:, 1:end-1), Ytr, weights);
                                else
                                    [coefs, tmpErr(i)] = lreg([Xtmp(:, 1:end-2) Xtmp(:, end)], Ytr, weights);
                                end
                                tmpCoefs(:,i) = [coefs; NaN];
                            else
                                tmpErr(i) = 0;
                                tmpErrVar(i,:) = 0;
                                for k = 1 : dy
                                    if (ok1)
                                        [coefs, sse] = lreg(Xtmp(:, 1:end-1), Ytr(:,k), weights);
                                    else
                                        [coefs, sse] = lreg([Xtmp(:, 1:end-2) Xtmp(:, end)], Ytr(:,k), weights);
                                    end
                                    tmpCoefs{k}(:,i) = [coefs; NaN];
                                    tmpErr(i) = tmpErr(i) + sse;
                                    tmpErrVar(i) = tmpErrVar(i) + sse / YtrVarN(k);
                                end
                                %tmpErr(i) = tmpErr(i) / dy;
                            end
                        %else % no basis function created
                        %    tmpErr(i) = Inf; % unnecessary because the array is already initialized as Inf
                        end
                    end
                    
                    % find the best pair of daughter basis functions from this parent
                    isNewVarPenalty = ((trainParams.newVarPenalty > 0) && (currIter > 1));
                    isPreferLinear = trainParams.allowLinear == 2;
                    if isNewVarPenalty || isPreferLinear
                        adjust = ones(1, idxEnd - idxStart + 1);
                    end
                    if isNewVarPenalty
                        % score adjustment due to new variable entering the model
                        for i = idxStart : idxEnd
                            if (~isempty(setdiff(basisFunctionList{1,i}, dimsInModel)))
                                adjust(i - idxStart + 1) = 1 + trainParams.newVarPenalty;
                            end
                        end
                    end
                    if isPreferLinear
                        % score adjustment due to preference of variables entering linearly
                        gcvCorrection = gcv(length(model.coefs) + 2, 1, n, trainParams.c);
                        if isfinite(gcvCorrection)
                            gcvCorrection = 1 / gcvCorrection * gcv(length(model.coefs) + 1, 1, n, trainParams.c);
                            for i = idxStart : idxEnd
                                if basisFunctionList{3,i}(end) == 2
                                    adjust(i - idxStart + 1) = adjust(i - idxStart + 1) * gcvCorrection;
                                end
                            end
                        end
                    end
                    if isNewVarPenalty || isPreferLinear
                        % the adjusted values are used for selection only,
                        % they are not used in any further calculations
                        [newErrAdj, ind] = min(tmpErr(idxStart:idxEnd) .* adjust);
                    else
                        [newErrAdj, ind] = min(tmpErr(idxStart:idxEnd));
                    end
                    ind = ind + idxStart - 1;
                    if dy == 1
                        newErr = tmpErr(ind);
                    else
                        newErr = tmpErrVar(ind) / dy;
                    end
                    fastParent{ir}.bestDim = basisFunctionList{1,ind}(end);
                    fastParent{ir}.bestIdxInList = ind;
                    fastParent{ir}.bestErr = newErr;
                    fastMinErrAdj(ir) = newErrAdj;
                    % continue the search for best daughters from all parents
                    if bestErrAdj > newErrAdj
                        bestErrAdj = newErrAdj;
                        bestParent = ir;
                    end
                end
                % we have found the best pair of daughter basis functions
                ind = fastParent{bestParent}.bestIdxInList;
                if dy == 1
                    newErr = fastParent{bestParent}.bestErr / YtrVarN;
                else
                	newErr = fastParent{bestParent}.bestErr; 
                end
                % reset the parent so that full optimization is done in next iteration
                fastMinErrAdj(bestParent) = -Inf;
                %fastParent{bestParent}.bestDim = -1;
                fastParent{bestParent}.iterComputedAllDims = -Inf;
                
            else % else of "if (trainParams.fastK < Inf)"
                
                % try all the reflected pairs in the list
                for i = 1 : size(basisFunctionList,2)
                    if trainParams.cubic
                        [t1, t2, diff] = findsideknots(model, basisFunctionList{1,i}, basisFunctionList{2,i}, ...
                                        d, minX, maxX, tmp_t1, tmp_t2);
                        Xtmp(:,1:end-2) = X;
                        % update basis functions with the updated side knots
                        for j = diff
                            Xtmp(:,j+1) = createbasisfunction(Xtr, Xtmp, model.knotdims{j}, model.knotsites{j}, ...
                                          model.knotdirs{j}, model.parents(j), minX, maxX, t1(j,:), t2(j,:));
                        end
                        % New basis function
                        dirs = basisFunctionList{3,i};
                        Xtmp(:,end-1) = createbasisfunction(Xtr, Xtmp, basisFunctionList{1,i}, basisFunctionList{2,i}, ...
                                        dirs, basisFunctionList{4,i}, minX, maxX, t1(end,:), t2(end,:));
                        ok1 = ~isnan(Xtmp(1,end-1));
                        % Reflected partner
                        dirs(end) = -dirs(end);
                        Xtmp(:,end) = createbasisfunction(Xtr, Xtmp, basisFunctionList{1,i}, basisFunctionList{2,i}, ...
                                      dirs, basisFunctionList{4,i}, minX, maxX, t1(end,:), t2(end,:));
                        ok2 = ~isnan(Xtmp(1,end));
                    else
                        % New basis function
                        dirs = basisFunctionList{3,i};
                        Xtmp(:,end-1) = createbasisfunction(Xtr, Xtmp, basisFunctionList{1,i}, ...
                                        basisFunctionList{2,i}, dirs, basisFunctionList{4,i}, minX, maxX);
                        ok1 = ~isnan(Xtmp(1,end-1));
                        % Reflected partner
                        dirs(end) = -dirs(end);
                        Xtmp(:,end) = createbasisfunction(Xtr, Xtmp, basisFunctionList{1,i}, ...
                                      basisFunctionList{2,i}, dirs, basisFunctionList{4,i}, minX, maxX);
                        ok2 = ~isnan(Xtmp(1,end));
                    end
                    if ok1 && ok2 % both basis functions created
                        if dy == 1
                        	[tmpCoefs(:,i), tmpErr(i)] = lreg(Xtmp, Ytr, weights);
                        else
                            tmpErr(i) = 0;
                            tmpErrVar(i,:) = 0;
                            for k = 1 : dy
                                [tmpCoefs{k}(:,i), sse] = lreg(Xtmp, Ytr(:,k), weights);
                                tmpErr(i) = tmpErr(i) + sse;
                                tmpErrVar(i) = tmpErrVar(i) + sse / YtrVarN(k);
                            end
                            %tmpErr(i) = tmpErr(i) / dy;
                        end
                    elseif ok1 || ok2 % one of the basis functions not created
                        if dy == 1
                            if (ok1)
                                [coefs, tmpErr(i)] = lreg(Xtmp(:, 1:end-1), Ytr, weights);
                            else
                                [coefs, tmpErr(i)] = lreg([Xtmp(:, 1:end-2) Xtmp(:, end)], Ytr, weights);
                            end
                            tmpCoefs(:,i) = [coefs; NaN];
                        else
                            tmpErr(i) = 0;
                            tmpErrVar(i,:) = 0;
                            for k = 1 : dy
                                if (ok1)
                                    [coefs, sse] = lreg(Xtmp(:, 1:end-1), Ytr(:,k), weights);
                                else
                                    [coefs, sse] = lreg([Xtmp(:, 1:end-2) Xtmp(:, end)], Ytr(:,k), weights);
                                end
                                tmpCoefs{k}(:,i) = [coefs; NaN];
                                tmpErr(i) = tmpErr(i) + sse;
                                tmpErrVar(i) = tmpErrVar(i) + sse / YtrVarN(k);
                            end
                            %tmpErr(i) = tmpErr(i) / dy;
                        end
                    %else % no basis function created
                    %    tmpErr(i) = Inf; % unnecessary because the array is already initialized as Inf
                    end
                end
                
                % find the best pair of daughter basis functions
                isNewVarPenalty = (trainParams.newVarPenalty > 0) && (currIter > 1);
                isPreferLinear = trainParams.allowLinear == 2;
                if isNewVarPenalty || isPreferLinear
                    adjust = ones(1,size(basisFunctionList,2));
                end
                if isNewVarPenalty
                    % score adjustment due to new variable entering the model
                    for i = 1 : size(basisFunctionList,2)
                        if (~isempty(setdiff(basisFunctionList{1,i}, dimsInModel)))
                            adjust(i) = 1 + trainParams.newVarPenalty;
                        end
                    end
                end
                if isPreferLinear
                    % score adjustment due to preference of variables entering linearly
                    gcvCorrection = gcv(length(model.coefs) + 2, 1, n, trainParams.c);
                    if isfinite(gcvCorrection)
                        gcvCorrection = 1 / gcvCorrection * gcv(length(model.coefs) + 1, 1, n, trainParams.c);
                        for i = 1 : size(basisFunctionList,2)
                            if basisFunctionList{3,i}(end) == 2
                                adjust(i) = adjust(i) * gcvCorrection;
                            end
                        end
                    end
                end
                if isNewVarPenalty || isPreferLinear
                    % the adjusted values are used for selection only,
                    % they are not used in any further calculations
                    [~, ind] = min(tmpErr .* adjust);
                else
                    [~, ind] = min(tmpErr);
                end
                if dy == 1
                    newErr = tmpErr(ind) / YtrVarN;
                else
                    newErr = tmpErrVar(ind) / dy;
                end
                
            end % end of "if (trainParams.fastK < Inf)"
            
            % stop the forward phase if no correct model was created or
            % if the decrease in error is below the threshold
            if isnan(newErr) || (err - newErr < trainParams.threshold)
                if trainParams.cubic
                    t1 = tmp_t1;
                    t2 = tmp_t2;
                end
                if verbose
                    if isnan(newErr)
                        fprintf('\nTermination condition is met: more complex models could not be created.');
                    else
                        fprintf('\nTermination condition is met: R2 improvement is below threshold.');
                    end
                end
                break;
            end
            
            if trainParams.cubic
                [t1, t2, diff] = findsideknots(model, basisFunctionList{1,ind}, basisFunctionList{2,ind}, ...
                              d, minX, maxX, tmp_t1, tmp_t2);
                % update basis functions with the updated side knots
                for j = diff
                    X(:,j+1) = createbasisfunction(Xtr, X, model.knotdims{j}, model.knotsites{j}, ...
                               model.knotdirs{j}, model.parents(j), minX, maxX, t1(j,:), t2(j,:));
                end
                % Add the new basis function
                dirs = basisFunctionList{3, ind};
                Xn = createbasisfunction(Xtr, X, basisFunctionList{1,ind}, basisFunctionList{2,ind}, ...
                     dirs, basisFunctionList{4,ind}, minX, maxX, t1(end,:), t2(end,:));
                if isnan(Xn(1)), Xn = []; end
                % Add the reflected partner
                dirs(end) = -dirs(end);
                Xn2 = createbasisfunction(Xtr, X, basisFunctionList{1,ind}, basisFunctionList{2,ind}, ...
                      dirs, basisFunctionList{4,ind}, minX, maxX, t1(end,:), t2(end,:));
                if isnan(Xn2(1)), Xn2 = []; end
                X = [X Xn Xn2];
                if ~isempty(Xn) && ~isempty(Xn2) % both basis functions are created
                    t1(end+1,:) = t1(end,:);
                    t2(end+1,:) = t2(end,:);
                end
            else
                dirs = basisFunctionList{3, ind};
                % Add the new basis function
                Xn = createbasisfunction(Xtr, X, basisFunctionList{1,ind}, ...
                     basisFunctionList{2,ind}, dirs, basisFunctionList{4,ind}, minX, maxX);
                if isnan(Xn(1)), Xn = []; end
                % Add the reflected partner
                dirs(end) = -dirs(end);
                Xn2 = createbasisfunction(Xtr, X, basisFunctionList{1,ind}, ...
                      basisFunctionList{2,ind}, dirs, basisFunctionList{4,ind}, minX, maxX);
                if isnan(Xn2(1)), Xn2 = []; end
                X = [X Xn Xn2];
            end
            
            if dy == 1
                model.coefs = tmpCoefs(:,ind);
            else
                model.coefs = tmpCoefs{1}(:,ind);
                for k = 1 : dy
                    modelsY{k}.coefs = tmpCoefs{k}(:,ind);
                end
            end
            
            % add the basis functions to the model
            numNewFuncs = 0;
            dirs = basisFunctionList{3, ind};
            if ~isempty(Xn)
                model.knotdims{end+1,1} = basisFunctionList{1, ind};
                model.knotsites{end+1,1} = basisFunctionList{2, ind};
                model.knotdirs{end+1,1} = dirs;
                model.parents(end+1,1) = basisFunctionList{4, ind};
                numNewFuncs = numNewFuncs + 1;
                if (trainParams.newVarPenalty > 0)
                    dimsInModel = union(dimsInModel, basisFunctionList{1, ind});
                end
            else
                model.coefs(end) = [];
                if dy > 1
                    for k = 1 : dy
                        modelsY{k}.coefs(end) = [];
                    end
                end
            end
            if ~isempty(Xn2)
                dirs(end) = -dirs(end);
                model.knotdims{end+1,1} = basisFunctionList{1, ind};
                model.knotsites{end+1,1} = basisFunctionList{2, ind};
                model.knotdirs{end+1,1} = dirs;
                model.parents(end+1,1) = basisFunctionList{4, ind};
                numNewFuncs = numNewFuncs + 1;
                if (trainParams.newVarPenalty > 0)
                    dimsInModel = union(dimsInModel, basisFunctionList{1, ind});
                end
            else
                model.coefs(end) = [];
                if dy > 1
                    for k = 1 : dy
                        modelsY{k}.coefs(end) = [];
                    end
                end
            end
            
            if verbose
                if getENP(length(model.coefs), trainParams.c) >= n
                    fprintf('xx'); % indicates that GCV would be Inf
                    if isnan(sizeInfGCV)
                        sizeInfGCV = currIter * 2 + 1;
                    end
                else
                    fprintf('..');
                end
            end
            
            % stop the forward phase if newErr is too small or if the
            % number of model's basis functions (including intercept term)
            % in the next iteration is expected to be > n
            err = newErr;
            if newErr < trainParams.threshold
                if verbose
                    fprintf('\nTermination condition is met: R2 >= 1 - threshold.');
                end
                break;
            end
            if length(model.coefs) + 2 > n
                if verbose
                    fprintf('\nTermination condition is met: number of basis functions reached the number of observations in data.');
                end
                break;
            end
            if trainParams.terminateWhenInfGCV && (getENP(length(model.coefs), trainParams.c) >= n)
                if verbose
                    fprintf('\nTermination condition is met: effective number of parameters reached the number of observations in data (GCV would be Inf).');
                end
                break;
            end
            
            if trainParams.cubic
                tmp_t1 = t1;
                tmp_t2 = t2;
            end
            basisFunctionList(:,ind) = [];
            
            if (trainParams.fastK < Inf)
                % update references according to the index of parent basis function
                for ir = 1 : length(fastParent)
                    if fastParent{ir}.idxStart > ind
                        fastParent{ir}.idxStart = fastParent{ir}.idxStart - 1;
                    end
                    if fastParent{ir}.idxEnd >= ind
                        fastParent{ir}.idxEnd = fastParent{ir}.idxEnd - 1;
                    end
                    if fastParent{ir}.bestIdxInList > ind
                        fastParent{ir}.bestIdxInList = fastParent{ir}.bestIdxInList - 1;
                    end
                end
                for ir = 1 : length(fastParent)
                    if fastParent{ir}.idxStart > fastParent{ir}.idxEnd
                        fastParent(ir) = [];
                        fastMinErrAdj(ir) = [];
                        fastIterComputedErr(ir) = [];
                        fastNumBasis = fastNumBasis - 1;
                        break;
                    end
                end
            end
            
        end % end of the main loop
        
        if verbose, fprintf('\n'); end
        
        if verbose && (~isnan(sizeInfGCV)) && (length(model.coefs) >= 5)
            percent = (length(model.coefs) - sizeInfGCV) / length(model.coefs);
            if percent >= 0.2
                warning('The last %d%% iterations have models with GCV = Inf. Depending on your data, you might want to try either enabling terminateWhenInfGCV, lowering maxFuncs, or lowering c.', round(percent * 100));
            end
        end
        
    end % end of "isempty(modelOld)"
    
    if isempty(modelOld)
        if verbose && trainParams.prune
            fprintf('Number of basis functions in the model after forward phase: %d\n', length(model.coefs));
        end
        
        if (doCubicFastLevel == 1) || ...
           ((doCubicFastLevel >= 2) && (~trainParams.prune)) % for this level, if there is no pruning, we force calculations for cubic
            % turn the cubic modelling on
            trainParams.cubic = true;
            [t1, t2] = findsideknots(model, [], [], d, minX, maxX, [], []);
            % update all the basis functions
            for i = 1 : length(model.knotdims)
                X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                           model.knotdirs{i}, model.parents(i), minX, maxX, t1(i,:), t2(i,:));
            end
            if dy == 1
                [model.coefs, model.MSE] = lreg(X, Ytr, weights);
                if isempty(weights)
                    model.MSE = model.MSE / n;
                else
                    model.MSE = model.MSE / sumWeights;
                end
            else
                for k = 1 : dy
                    [model.coefs, modelsY{k}.MSE] = lreg(X, Ytr(:,k), weights);
                    modelsY{k}.coefs = model.coefs;
                    if isempty(weights)
                        modelsY{k}.MSE = modelsY{k}.MSE / n;
                    else
                        modelsY{k}.MSE = modelsY{k}.MSE / sumWeights;
                    end
                end
            end
        else
            if dy == 1
                %if isempty(weights)
                %    model.MSE = sum((Ytr-X*model.coefs).^2) / n;
                %else
                %    model.MSE = sum((Ytr-X*model.coefs).^2.*weights) / sumWeights;
                %end
                model.MSE = err * YtrVar; % faster. works because err is sse/YtrVarN
            else
                for k = 1 : dy
                    if isempty(weights)
                        modelsY{k}.MSE = sum((Ytr(:,k)-X*modelsY{k}.coefs).^2) / n;
                    else
                        modelsY{k}.MSE = sum((Ytr(:,k)-X*modelsY{k}.coefs).^2.*weights) / sumWeights;
                    end
                end
            end
        end
        if dy == 1
            model.GCV = gcv(length(model.coefs), model.MSE, n, trainParams.c);
        else
            model.GCV = 0;
            for k = 1 : dy
                modelsY{k}.GCV = gcv(length(model.coefs), modelsY{k}.MSE, n, trainParams.c);
                model.GCV = model.GCV + modelsY{k}.GCV;
            end
        end
        if trainParams.cubic
            model.t1 = t1;
            model.t2 = t2;
        end
    end
    
    % BACKWARD PHASE
    
    if trainParams.prune
        
        if verbose, fprintf('Backward phase .'); end
        
        if ~isempty(modelOld) % create basis functions from scratch when modelOld is used
            if (doCubicFastLevel == -1) || (doCubicFastLevel >= 2) % either no cubic or not yet cubic
                if (dy == 1) && isfield(modelOld, 'X')
                    X = modelOld.X;
                elseif (dy > 1) && isfield(modelOld{1}, 'X')
                    X = modelOld{1}.X;
                else
                    % create all basis functions (piecewise-linear) from scratch
                    X = ones(n,length(model.knotdims)+1);
                    for i = 1 : length(model.knotdims)
                        X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                                   model.knotdirs{i}, model.parents(i), minX, maxX);
                    end
                    if dy == 1
                        [model.coefs, model.MSE] = lreg(X, Ytr, weights);
                        if isempty(weights)
                            model.MSE = model.MSE / n;
                        else
                            model.MSE = model.MSE / sumWeights;
                        end
                    else
                        for k = 1 : dy
                            [model.coefs, modelsY{k}.MSE] = lreg(X, Ytr(:,k), weights);
                            modelsY{k}.coefs = model.coefs;
                            if isempty(weights)
                                modelsY{k}.MSE = modelsY{k}.MSE / n;
                            else
                                modelsY{k}.MSE = modelsY{k}.MSE / sumWeights;
                            end
                        end
                    end
                end
            else % cubic modelling when doCubicFastLevel is set to 0 or 1
                trainParams.cubic = true; % set to true once again because the value in modelOld is lost
                t1 = model.t1;
                t2 = model.t2;
                if (dy == 1) && isfield(modelOld, 'X')
                    X = modelOld.X;
                elseif (dy > 1) && isfield(modelOld{1}, 'X')
                    X = modelOld{1}.X;
                else
                    % create all basis functions (piecewise-cubic) from scratch
                    X = ones(n,length(model.knotdims)+1);
                    for i = 1 : length(model.knotdims)
                        X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                                   model.knotdirs{i}, model.parents(i), minX, maxX, t1(i,:), t2(i,:));
                    end
                end
            end
            % recalculate GCV in case c has changed (e.g., if aresbuild is called from arescvc) or GCV was not yet calculated
            if dy == 1
                model.GCV = gcv(length(model.coefs), model.MSE, n, trainParams.c);
            else
                model.GCV = 0;
                for k = 1 : dy
                    modelsY{k}.GCV = gcv(length(model.coefs), modelsY{k}.MSE, n, trainParams.c);
                    model.GCV = model.GCV + modelsY{k}.GCV;
                end
            end
        end
        
        models = {model};
        if dy == 1
            mses = model.MSE;
        else
            modelsYAll = {modelsY};
        end
        gcvs = model.GCV; % for multi-response data, this is a sum of GCVs
        if (dy > 1) && requestingResultsEval
            msesAll = zeros(1,dy);
            for k = 1 : dy
                msesAll(1,k) = modelsY{k}.MSE;
            end
            gcvsAll = zeros(1,dy);
            for k = 1 : dy
                gcvsAll(1,k) = modelsY{k}.GCV;
            end
        end
        
        
        if requestingResultsEval
            resultsEval.usedVars = false(length(model.knotdims)+1,d);
            resultsEval.usedVars(length(model.knotdims)+1,:) = getUsedVariables(model.knotdims, d);
        end
        
        % the main loop of the backward phase
        for j = 1 : length(model.knotdims)
            tmpErr = zeros(1, length(model.knotdims));
            if dy == 1
                tmpCoefs = inf(length(model.coefs)-1, length(model.knotdims));
            else
                tmpGCV = inf(1, length(model.knotdims));
                if requestingResultsEval
                    tmpMSEAll = inf(length(model.knotdims), dy);
                    tmpGCVAll = inf(length(model.knotdims), dy);
                end
                tmpCoefs = cell(dy, 1);
                for k = 1 : dy
                    tmpCoefs{k} = inf(length(model.coefs)-1, length(model.knotdims));
                end
            end
            
            % try to delete basis functions one at a time
            for jj = 1 : length(model.knotdims)
                Xtmp = X;
                Xtmp(:,jj+1) = [];
                if trainParams.cubic
                    % create a temporary model without the basis function
                    tmp_t1 = t1;
                    tmp_t1(jj,:) = [];
                    tmp_t2 = t2;
                    tmp_t2(jj,:) = [];
                    tmp_model.knotdims = model.knotdims;
                    tmp_model.knotdims(jj) = [];
                    tmp_model.knotsites = model.knotsites;
                    tmp_model.knotsites(jj) = [];
                    tmp_model.knotdirs = model.knotdirs;
                    tmp_model.knotdirs(jj) = [];
                    tmp_model.parents = model.parents;
                    tmp_model.parents(jj) = [];
                    tmp_model.parents = updateParents(tmp_model.parents, jj);
                    [tmp_t1, tmp_t2, diff] = findsideknots(tmp_model, [], [], d, minX, maxX, tmp_t1, tmp_t2);
                    % update basis functions that have their side knots moved
                    for i = diff
                        Xtmp(:,i+1) = createbasisfunction(Xtr, Xtmp, tmp_model.knotdims{i}, tmp_model.knotsites{i}, ...
                                      tmp_model.knotdirs{i}, tmp_model.parents(i), minX, maxX, tmp_t1(i,:), tmp_t2(i,:));
                    end
                end
                if dy == 1
                    [coefs, tmpErr(jj)] = lreg(Xtmp, Ytr, weights);
                    tmpCoefs(:,jj) = coefs;
                else
                    tmpGCV(jj) = 0;
                    for k = 1 : dy
                        [coefs, err] = lreg(Xtmp, Ytr(:,k), weights);
                        tmpErr(jj) = tmpErr(jj) + err;
                        tmpCoefs{k}(:,jj) = coefs;
                        if isempty(weights)
                            err = err / n;
                        else
                            err = err / sumWeights;
                        end
                        if requestingResultsEval
                            tmpMSEAll(jj,k) = err;
                        end
                        err = gcv(length(model.coefs) - 1, err, n, trainParams.c); % "-1" is because we delete one basis function
                        tmpGCV(jj) = tmpGCV(jj) + err;
                        if requestingResultsEval
                            tmpGCVAll(jj,k) = err;
                        end
                    end
                end
            end
            
            [~, ind] = min(tmpErr); % find the best modification
            X(:,ind+1) = [];
            if dy == 1
                model.coefs = tmpCoefs(:,ind);
            else
                model.coefs = tmpCoefs{1}(:,ind);
                for k = 1 : dy
                    modelsY{k}.coefs = tmpCoefs{k}(:,ind);
                end
            end
            model.knotdims(ind) = [];
            model.knotsites(ind) = [];
            model.knotdirs(ind) = [];
            model.parents(ind) = [];
            model.parents = updateParents(model.parents, ind);
            
            if requestingResultsEval
                resultsEval.usedVars(length(model.knotdims)+1,:) = getUsedVariables(model.knotdims, d);
            end
            
            if trainParams.cubic
                t1(ind,:) = [];
                t2(ind,:) = [];
                [t1, t2, diff] = findsideknots(model, [], [], d, minX, maxX, t1, t2);
                % update basis functions that have their side knots moved
                for i = diff
                    X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                               model.knotdirs{i}, model.parents(i), minX, maxX, t1(i,:), t2(i,:));
                end
                model.t1 = t1;
                model.t2 = t2;
            end
            
            models{end+1} = model;
            if dy == 1
                if isempty(weights)
                    mses(end+1) = tmpErr(ind) / n;
                else
                    mses(end+1) = tmpErr(ind) / sumWeights;
                end
                gcvs(end+1) = gcv(length(model.coefs), mses(end), n, trainParams.c);
            else
                modelsYAll{end+1} = modelsY;
                gcvs(end+1) = tmpGCV(ind);
                if requestingResultsEval
                    msesAll(end+1,1:dy) = tmpMSEAll(ind,:);
                    gcvsAll(end+1,1:dy) = tmpGCVAll(ind,:);
                end
            end
            
            if verbose, fprintf('.'); end
        end % end of the main loop
        
        % now choose the best pruned model
        if trainParams.maxFinalFuncs <= 1
            g = gcvs(end);
            ind = length(gcvs);
        else
            if dy == 1
                [sgcvs, sind] = sort(gcvs);
                gcvNull = gcv(1, YtrVar, n, 0);
                if trainParams.maxFinalFuncs >= length(models{1}.coefs) % if even the biggest model is not bigger than maxFinalFuncs
                    g = sgcvs(1); % in the sorted list, 1st is the best
                    ind = sind(1);
                else
                    [g, ind] = min(gcvs(end-trainParams.maxFinalFuncs+1:end));
                    ind = ind + length(gcvs) - trainParams.maxFinalFuncs;
                end
                % select smaller model, if it is only negligibly worse
                gBest = g;
                indBest = ind;
                for i = 2 : numel(sgcvs)
                    if (sgcvs(i) / gcvNull - 1e-10 <= g / gcvNull)
                        if (sind(i) > ind)
                            gBest = sgcvs(i);
                            indBest = sind(i);
                        end
                    else
                        break;
                    end
                end
                g = gBest;
                ind = indBest;
            else
                if trainParams.maxFinalFuncs >= length(models{1}.coefs) % if even the biggest model is not bigger than maxFinalFuncs
                    [g, ind] = min(gcvs);
                else
                    [g, ind] = min(gcvs(end-trainParams.maxFinalFuncs+1:end));
                    ind = ind + length(gcvs) - trainParams.maxFinalFuncs;
                end
            end
        end
        model = models{ind};
        if dy > 1
            modelsY = modelsYAll{ind};
        end
        
        if requestingResultsEval
            availableDataEval = ~isempty(dataEval);
            if availableDataEval
                nDataEval = size(dataEval.Y,1);
                dataEvalHasWeights = isfield(dataEval, 'weights');
                if dataEvalHasWeights
                    if isempty(dataEval.weights)
                        dataEvalHasWeights = false;
                    else
                        sumDataEvalWeights = sum(dataEval.weights);
                    end
                end
            end
            if dy == 1
                if availableDataEval
                    % Calculate MSEtest in the test data for each model size
                    resultsEval.MSEtest = zeros(length(models),1);
                    for j = 1 : length(models)
                        models{j}.trainParams = trainParams;
                        models{j}.minX = minX;
                        models{j}.maxX = maxX;
                        if ~dataEvalHasWeights
                            resultsEval.MSEtest(j) = ...
                                sum((dataEval.Y - arespredict(models{j}, dataEval.X)).^2) / nDataEval;
                        else
                            resultsEval.MSEtest(j) = ...
                                sum((dataEval.Y - arespredict(models{j}, dataEval.X)).^2.*dataEval.weights) / sumDataEvalWeights;
                        end
                    end
                end
                % Save GCV and R2GCV for each model size
                resultsEval.MSE = mses';
                resultsEval.R2 = 1 - resultsEval.MSE / YtrVar;
                resultsEval.GCV = gcvs';
                resultsEval.R2GCV = 1 - resultsEval.GCV / gcv(1, YtrVar, n, 0);
            else
                % Calculate MSEtest in the test data for each model size
                if availableDataEval
                    resultsEval.MSEtest = zeros(length(models),dy);
                    for j = 1 : length(models)
                        models{j}.trainParams = trainParams;
                        models{j}.minX = minX;
                        models{j}.maxX = maxX;
                        modelsYTestTmp = modelsYAll{j};
                        for k = 1 : dy
                            models{j}.coefs = modelsYTestTmp{k}.coefs;
                            if ~dataEvalHasWeights
                                resultsEval.MSEtest(j,k) = ...
                                    sum((dataEval.Y(:,k) - arespredict(models{j}, dataEval.X)).^2) / nDataEval;
                            else
                                resultsEval.MSEtest(j,k) = ...
                                    sum((dataEval.Y(:,k) - arespredict(models{j}, dataEval.X)).^2.*dataEval.weights) / sumDataEvalWeights;
                            end
                        end
                    end
                end
                % Save GCV and R2GCV for each model size
                resultsEval.MSE = msesAll;
                resultsEval.R2 = 1 - resultsEval.MSE ./ repmat(YtrVar, size(resultsEval.MSE,1), 1);
                resultsEval.GCV = gcvsAll;
                GCVnull = zeros(size(resultsEval.GCV,1), dy);
                for k = 1 : dy
                    GCVnull(:,k) = gcv(1, YtrVar(k), n, 0);
                end
                resultsEval.R2GCV = 1 - resultsEval.GCV ./ GCVnull;
                % mean accross models
                resultsEval.MSE = mean(resultsEval.MSE,2);
                resultsEval.R2 = mean(resultsEval.R2,2);
                resultsEval.GCV = mean(resultsEval.GCV,2);
                resultsEval.R2GCV = mean(resultsEval.R2GCV,2);
            end
            % Flip so that model size is ascending
            resultsEval.MSE = flip(resultsEval.MSE);
            resultsEval.R2 = flip(resultsEval.R2);
            resultsEval.GCV = flip(resultsEval.GCV);
            %resultsEval.GCV(isinf(resultsEval.GCV)) = NaN;
            resultsEval.R2GCV = flip(resultsEval.R2GCV);
            %resultsEval.R2GCV(isinf(resultsEval.R2GCV)) = NaN;
            if availableDataEval
                % Calculate R2test
                resultsEval.R2test = nan(size(resultsEval.MSEtest));
                if ~dataEvalHasWeights
                    dataEvalYMean = mean(dataEval.Y,1);
                    for k = 1 : dy
                        dataEvalYtrVar = sum((dataEval.Y(:,k) - dataEvalYMean(k)) .^ 2) / nDataEval;
                        if (dataEvalYtrVar > eps)
                            resultsEval.R2test(:,k) = 1 - resultsEval.MSEtest(:,k) / dataEvalYtrVar;
                        else
                            resultsEval.R2test(:) = NaN;
                            break;
                        end
                    end
                else
                    for k = 1 : dy
                        dataEvalYMean = sum(dataEval.Y(:,k) .* dataEval.weights) / sumDataEvalWeights;
                        dataEvalYtrVar = sum(((dataEval.Y(:,k) - dataEvalYMean) .^ 2) .* dataEval.weights) / sumDataEvalWeights;
                        if (dataEvalYtrVar > eps)
                            resultsEval.R2test(:,k) = 1 - resultsEval.MSEtest(:,k) / dataEvalYtrVar;
                        else
                            resultsEval.R2test(:) = NaN;
                            break;
                        end
                    end
                end
                % Mean across models
                resultsEval.MSEtest = mean(resultsEval.MSEtest,2);
                resultsEval.R2test = mean(resultsEval.R2test,2);
                % Flip so that model size is ascending
                resultsEval.MSEtest = flip(resultsEval.MSEtest);
                resultsEval.R2test = flip(resultsEval.R2test);
            end
        end
        
        if doCubicFastLevel >= 2
            % turn the cubic modelling on
            trainParams.cubic = true;
            [t1, t2] = findsideknots(model, [], [], d, minX, maxX, [], []);
            % update all the basis functions
            X = ones(n,length(model.coefs));
            for i = 1 : length(model.knotdims)
                X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                           model.knotdirs{i}, model.parents(i), minX, maxX, t1(i,:), t2(i,:));
            end
            model.t1 = t1;
            model.t2 = t2;
            if dy == 1
                [model.coefs, model.MSE] = lreg(X, Ytr, weights);
                if isempty(weights)
                    model.MSE = model.MSE / n;
                else
                    model.MSE = model.MSE / sumWeights;
                end
                model.GCV = gcv(length(model.coefs), model.MSE, n, trainParams.c);
            else
                for k = 1 : dy
                    [modelsY{k}.coefs, modelsY{k}.MSE] = lreg(X, Ytr(:,k), weights);
                    if isempty(weights)
                        modelsY{k}.MSE = modelsY{k}.MSE / n;
                    else
                        modelsY{k}.MSE = modelsY{k}.MSE / sumWeights;
                    end
                    modelsY{k}.GCV = gcv(length(model.coefs), modelsY{k}.MSE, n, trainParams.c);
                end
            end
        else
            if dy == 1
                model.MSE = mses(ind);
                model.GCV = g;
            end
            if keepX || (dy > 1)
                % recreate all basis functions from scratch just for keepX
                % or to be able to recalculate MSE and GCV for multiple Ys
                X = ones(n,length(model.coefs));
                if trainParams.cubic
                    for i = 1 : length(model.knotdims)
                        X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                                   model.knotdirs{i}, model.parents(i), minX, maxX, model.t1(i,:), model.t2(i,:));
                    end
                else
                    for i = 1 : length(model.knotdims)
                        X(:,i+1) = createbasisfunction(Xtr, X, model.knotdims{i}, model.knotsites{i}, ...
                                   model.knotdirs{i}, model.parents(i), minX, maxX);
                    end
                end
            end
            if dy > 1
                for k = 1 : dy
                    if isempty(weights)
                        modelsY{k}.MSE = sum((Ytr(:,k)-X*modelsY{k}.coefs).^2) / n;
                    else
                        modelsY{k}.MSE = sum((Ytr(:,k)-X*modelsY{k}.coefs).^2.*weights) / sumWeights;
                    end
                    modelsY{k}.GCV = gcv(length(model.coefs), modelsY{k}.MSE, n, trainParams.c);
                end
            end
        end
        
        if verbose, fprintf('\n'); end
        
    end % end of "trainParams.prune"
    
end % end of "if useEndSpan*2 >= n"

warning(origWarningState);

model.trainParams = trainParams;
model.minX = minX;
model.maxX = maxX;
model.isBinary = isBinary;

if keepX
    model.X = X;
end

time = toc(ttt);
if verbose
    fprintf('Number of basis functions in the final model: %d\n', length(model.coefs));
    fprintf('Total effective number of parameters: %0.1f\n', getENP(length(model.coefs), model.trainParams.c));
    maxDeg = 0;
    vars = [];
    if ~isempty(model.knotdims)
        for i = 1 : length(model.knotdims)
            vars = union(vars, model.knotdims{i});
            if length(model.knotdims{i}) > maxDeg
                maxDeg = length(model.knotdims{i});
            end
        end
    end
    fprintf('Highest degree of interactions: %d\n', maxDeg);
    if ~isempty(vars)
        listStr = '';
        for i = 1:length(vars)
            listStr = [listStr 'x' int2str(vars(i))];
            if i < length(vars)
                listStr = [listStr ', '];
            end
        end
        fprintf('Number of input variables in the model: %d (%s)\n', length(vars), listStr);
    else
        fprintf('Number of input variables in the model: 0\n');
    end
    fprintf('Execution time: %0.2f seconds\n', time);
end

if dy > 1
    for k = 1 : dy
        coefs = modelsY{k}.coefs;
        MSE = modelsY{k}.MSE;
        GCV = modelsY{k}.GCV;
        modelsY{k} = model;
        modelsY{k}.coefs = coefs;
        modelsY{k}.MSE = MSE;
        modelsY{k}.GCV = GCV;
    end
    model = modelsY;
end

return

%==========================================================================

function enp = getENP(nBasis, c)
% Calculates model's effective number of parameters
enp = nBasis + c * (nBasis - 1) / 2;
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

function parents = updateParents(parents, deletedInd)
% Updates direct parent indices after deletion of a basis function.
parents(parents == deletedInd) = 0;
tmp = parents > deletedInd;
parents(tmp) = parents(tmp) - 1;
return

function [basisFunctionList, idxStart1, idxEnd1, idxStart2, idxEnd2] = ...
    createList(basisFunctionList_old, Xtr, sortedXtr, sortedXtrInd, n, d, ...
    model, numNewFuncs, trainParams, minX, maxX)
% Takes the old list of basis functions and adds new ones according to the
% current model. If the old list is empty, adds all basis functions without
% interactions. If it's non-empty, adds only basis functions with
% interactions which result from the last numNewFuncs basis functions
% (typically 2, sometimes 1).
% For knot placement (and for calculation of minSpan) only those potential
% knot sites are considered for which the parent basis function is larger
% than zero.
% Input argument Xtr should contain all the training observations;
% sortedXtr should contain all the Xtr observations, except first and last
% endSpan points, sorted in each column separately; sortedXtrInd should
% contain indices of the sorted observations.

% Create no-ineraction basis functions

if (isempty(basisFunctionList_old)) && (numNewFuncs == 0)
    if trainParams.useMinSpan ~= 1
        % make a list of knot sites allowed by minSpan
        minSpan = getMinSpan(d, n, trainParams.useMinSpan);
        % symmetrically center the knot sites
        nAvail = n - trainParams.useEndSpan * 2;
        if nAvail > minSpan % is there space for more than one knot?
            nDiv = floor(nAvail / minSpan);
            if nDiv == nAvail / minSpan
                offset = floor(minSpan / 2);
            else
                offset = floor((nAvail - nDiv * minSpan) / 2);
            end
        else
            offset = floor(nAvail / 2); % if space for only one knot, put it in center
        end
        allowed = mod(-offset:nAvail-1-offset, minSpan) == 0;
    end
    basisFunctionList = cell(4,0);
    counter = 0;
    for di = 1 : d % for each dimension
        forceLinear = any(trainParams.forceLinear == di);
        if (trainParams.allowLinear > 0) || forceLinear
            counter = counter + 1;
            basisFunctionList{1, counter} = di; % knotdims
            basisFunctionList{2, counter} = minX(di);  % knotsites
            basisFunctionList{3, counter} = 2;  % knotdirs
            basisFunctionList{4, counter} = 0;  % parent
            if forceLinear
                continue;
            end
            ignoreEdges = true; % because var entering linearly is (almost) like putting a knot on the edge
        else
            ignoreEdges = false;
        end
        if trainParams.useMinSpan == 1
            allowedSites = unique(sortedXtr(:,di))';
        else
            allowedSites = unique(sortedXtr(allowed,di))';
        end
        if ignoreEdges && ~isempty(allowedSites)
            allowedSites = allowedSites((1+(allowedSites(1) == minX(di))):(end-(allowedSites(end) == maxX(di))));
        end
        % add new unique basis functions to the list (either all or except
        % those which do not fall on the allowed knot sites)
        for x = allowedSites
            % we could use cell array of structs here but benchmarking
            % showed that this is faster
            counter = counter + 1;
            basisFunctionList{1, counter} = di; % knotdims
            basisFunctionList{2, counter} = x;  % knotsites
            basisFunctionList{3, counter} = 1;  % knotdirs
            basisFunctionList{4, counter} = 0;  % parent
        end
    end
    if (counter > 0)
        idxStart1 = 1;
        idxEnd1 = counter;
    else
        idxStart1 = -1;
        idxEnd1 = -1;
    end
    idxStart2 = -1;
    idxEnd2 = -1;
    return
end

if (trainParams.maxInteractions < 2) || (numNewFuncs < 1)
    basisFunctionList = basisFunctionList_old;
    idxStart1 = -1;
    idxEnd1 = -1;
    idxStart2 = -1;
    idxEnd2 = -1;
    return
end

% Create basis functions with interactions

% for basis functions with interactions we can make endSpan wider
if (trainParams.endSpanAdjust > 1)
    endSpanToUse = round(trainParams.useEndSpan * trainParams.endSpanAdjust);
    if endSpanToUse * 2 >= n
        endSpanToUse = floor(n / 2) - 1; % force always at least one knot
    end
else
    endSpanToUse = trainParams.useEndSpan;
end

basisFunctionList = basisFunctionList_old;
sizeOld = size(basisFunctionList_old,2);
counter = sizeOld;
start = length(model.knotdims) - (numNewFuncs-1);

idxStart1 = counter + 1;
idxEnd1 = -1;
idxStart2 = -1;

% loop through one or two last basis functions already in the model
for j = start : length(model.knotdims)
    if (j > start)
        if (counter >= idxStart1)
            idxEnd1 = counter;
        else
            idxStart1 = -1;
            idxEnd1 = -1;
        end
        idxStart2 = counter + 1;
    end
    if length(model.knotdims{j}) < trainParams.maxInteractions
        allowedDims = 1 : d;
        if trainParams.selfInteractions <= 1
            % will not consider already used dimensions
            allowedDims = setdiff(allowedDims, model.knotdims{j});
        else
            for i = 1 : d
                if length(find(model.knotdims{j} == i)) >= trainParams.selfInteractions
                    allowedDims = setdiff(allowedDims, i);
                end
            end
        end
        if isempty(allowedDims)
            continue
        end
        if (~isempty(trainParams.noInteract)) || (~isempty(trainParams.yesInteract))
            if ~isempty(trainParams.noInteract)
                allowed = true(1,d);
                for i = model.knotdims{j}
                    allowed(trainParams.noInteract(trainParams.noInteract(:,1)==i,2)) = false;
                end
            else
                allowed = false(1,d);
                for i = model.knotdims{j}
                    allowed(trainParams.yesInteract(trainParams.yesInteract(:,1)==i,2)) = true;
                end
            end
            allowedDims = setdiff(allowedDims, find(~allowed));
            if isempty(allowedDims)
                continue
            end
        end

        if trainParams.useMinSpan ~= 1

            % make a list of knot sites allowed by minSpan
            nonzero = listNonZero(Xtr, model.knotdims{j}, model.knotsites{j}, model.knotdirs{j});
            minSpan = getMinSpan(d, length(find(nonzero)), trainParams.useMinSpan);
            if ~isfinite(minSpan)
                continue
            end
            % symmetrically center the knot sites
            nAvail = n - endSpanToUse * 2;
            if nAvail > minSpan % is there space for more than one knot?
                nDiv = floor(nAvail / minSpan);
                if nDiv == nAvail / minSpan
                    offset = floor(minSpan / 2);
                else
                    offset = floor((nAvail - nDiv * minSpan) / 2);
                end
            else
                offset = floor(nAvail / 2); % if space for only one knot, put it in center
            end
            allowed = mod(-offset:nAvail-1-offset, minSpan) == 0;
            
            for di = allowedDims % for each dimension
                forceLinear = any(trainParams.forceLinear == di);
                if (trainParams.allowLinear > 0) || forceLinear
                    counter = counter + 1;
                    basisFunctionList{1, counter} = [model.knotdims{j} di]; % knotdims
                    basisFunctionList{2, counter} = [model.knotsites{j} minX(di)];  % knotsites
                    basisFunctionList{3, counter} = [model.knotdirs{j} 2];  % knotdirs
                    basisFunctionList{4, counter} = j;  % parent
                    if forceLinear
                        continue;
                    end
                    ignoreEdges = true; % because var entering linearly is (almost) like putting a knot on the edge
                else
                    ignoreEdges = false;
                end
                % add new unique basis functions to the list (all except
                % those which do not fall on the allowed knot sites and
                % except those for which the parent basis function is zero
                % on the knot site)
                if (trainParams.endSpanAdjust > 1)
                    add = endSpanToUse - trainParams.useEndSpan;
                    allowed = allowed & nonzero(sortedXtrInd(1+add:end-add,di));
                else
                    allowed = allowed & nonzero(sortedXtrInd(:,di));
                end
                allowedSites = unique(sortedXtr(allowed,di))';
                if ignoreEdges && ~isempty(allowedSites)
                    allowedSites = allowedSites((1+(allowedSites(1) == minX(di))):(end-(allowedSites(end) == maxX(di))));
                end
                for x = allowedSites
                    counter = counter + 1;
                    basisFunctionList{1, counter} = [model.knotdims{j} di];
                    basisFunctionList{2, counter} = [model.knotsites{j} x];
                    basisFunctionList{3, counter} = [model.knotdirs{j} 1];
                    basisFunctionList{4, counter} = j;
                end
            end

        else

            nonzero = listNonZero(Xtr, model.knotdims{j}, model.knotsites{j}, model.knotdirs{j});
            for di = allowedDims % for each dimension
                forceLinear = any(trainParams.forceLinear == di);
                if (trainParams.allowLinear > 0) || forceLinear
                    counter = counter + 1;
                    basisFunctionList{1, counter} = [model.knotdims{j} di]; % knotdims
                    basisFunctionList{2, counter} = [model.knotsites{j} minX(di)];  % knotsites
                    basisFunctionList{3, counter} = [model.knotdirs{j} 2];  % knotdirs
                    basisFunctionList{4, counter} = j;  % parent
                    if forceLinear
                        continue;
                    end
                    ignoreEdges = true; % because var entering linearly is (almost) like putting a knot on the edge
                else
                    ignoreEdges = false;
                end
                % add new unique basis functions to the list (all except
                % those for which the parent basis function is zero on
                % the knot site)
                if (trainParams.endSpanAdjust > 1)
                    add = endSpanToUse - trainParams.useEndSpan;
                    allowedSites = unique(sortedXtr(nonzero(sortedXtrInd(1+add:end-add,di)),di))';
                else
                    allowedSites = unique(sortedXtr(nonzero(sortedXtrInd(:,di)),di))';
                end
                if ignoreEdges && ~isempty(allowedSites)
                    allowedSites = allowedSites((1+(allowedSites(1) == minX(di))):(end-(allowedSites(end) == maxX(di))));
                end
                for x = allowedSites
                    counter = counter + 1;
                    basisFunctionList{1, counter} = [model.knotdims{j} di];
                    basisFunctionList{2, counter} = [model.knotsites{j} x];
                    basisFunctionList{3, counter} = [model.knotdirs{j} 1];
                    basisFunctionList{4, counter} = j;
                end
            end

        end

    end
end
if (sizeOld == size(basisFunctionList,2)) % size didn't increase
    idxStart1 = -1;
    idxEnd1 = -1;
    idxStart2 = -1;
    idxEnd2 = -1;
else
    if (idxStart1 > 0) && (idxEnd1 <= 0) % possible only if there is no second parent
        idxEnd1 = counter;
        idxEnd2 = -1;
    elseif (idxStart2 > 0) && (counter >= idxStart2)
        idxEnd2 = counter;
    else
        idxStart2 = -1;
        idxEnd2 = -1;
    end
end
return

function nonzero = listNonZero(Xtr, knotdims, knotsites, knotdirs)
% Lists nonzero (according to the parent basis function) sites where knots
% may be placed. (Line 5 of Algorithm 2 in Friedman, 1991a)
nonzero = true(1,size(Xtr,1));
for j = 1 : size(Xtr,1)
    for i = 1 : length(knotdims)
        z = Xtr(j,knotdims(i)) - knotsites(i);
        if ((z >= 0) && (knotdirs(i) < 0)) || ...
           ((z <= 0) && (knotdirs(i) > 0))
            nonzero(j) = false;
            break;
        end
    end
end
return

function vars = getUsedVariables(knotdims, d)
% Outputs a logical vector with true for variables that are included in the knotdims
vars = false(1,d);
if ~isempty(knotdims)
    list = [];
    for i = 1 : length(knotdims)
        list = union(list, knotdims{i});
    end
    vars(list) = true;
end
return

function s = getEndSpan(d)
% Calculation of endSpan so that potential knot sites that are too close to
% the ends of data intervals are not considered.
%s = floor(3 - log2(0.05/d));
s = floor(7.32193 + log(d) / 0.69315); % precomputed version
if s < 1, s = 1; end
return

function s = getMinSpan(d, nz, param)
% Calculation of minSpan so that only those potential knot sites are
% considered which are at least minSpan apart. This increases resistance to
% runs of correlated noise.
% nz is the number of potential knot sites where the parent basis function
% is nonzero.
if nz == 0
    s = Inf;
else
    if param < 0 % automatic
        %s = floor(-log2(-log(1-0.05)/(d*nz)) / 2.5);
        s = floor((2.9702 + log(d*nz)) / 1.7329); % precomputed version
    else
        s = param;
        if s > nz
            s = Inf;
        end
    end
    if s < 1, s = 1; end
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
