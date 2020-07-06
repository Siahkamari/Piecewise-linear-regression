function trainParams = aresparams(maxFuncs, c, cubic, cubicFastLevel, ...
    selfInteractions, maxInteractions, threshold, prune, fastK, fastBeta, ...
    fastH, useMinSpan, useEndSpan, maxFinalFuncs, endSpanAdjust, ...
    newVarPenalty, terminateWhenInfGCV, yesInteract, noInteract, ...
    allowLinear, forceLinear)
% aresparams
% Creates configuration for building ARES models. The output structure is
% for further use with aresbuild, arescv, and arescvc functions.
%
% Call:
%   trainParams = aresparams(maxFuncs, c, cubic, cubicFastLevel, ...
%       selfInteractions, maxInteractions, threshold, prune, fastK, ...
%       fastBeta, fastH, useMinSpan, useEndSpan, maxFinalFuncs, ...
%       endSpanAdjust, newVarPenalty, terminateWhenInfGCV, yesInteract, ...
%       noInteract, allowLinear, forceLinear)
%
% All the input arguments of this function are optional. Empty values are
% also accepted (the corresponding defaults will be used).
% Parameters prune and maxFinalFuncs are used in the backward pruning
% phase. Parameters c and cubic may be used in both phases depending on
% terminateWhenInfGCV, forceLinear, and cubicFastLevel. All other
% parameters are used in the forward phase only.
% For many applications, it can be expected that the most attention should
% be paid to the following parameters: maxFuncs, maxInteractions, cubic, c,
% and maxFinalFuncs. It is quite possible that the default values for
% maxFuncs and maxInteractions will be far from optimal for your data.
% But note that, if you are prepared to use Cross-Validation, choosing a
% good value for maxFinalFuncs can sometimes release you from being too
% pedantic about parameters maxFuncs and c, because you can set large
% enough maxFuncs and not too large c and follow the example in Section 3.3
% in user's manual.
% If you have the necessary domain knowledge, it is recommended to also set
% yesInteract, noInteract, allowLinear, and forceLinear.
%
% Input:
%   maxFuncs      : The maximum number of basis functions included in model
%                   in the forward building phase (before pruning in the
%                   backward phase). Includes the intercept term. The
%                   recommended value for this parameter is about two times
%                   the expected number of basis functions in the final
%                   model (Friedman, 1991a). Note that the algorithm may
%                   also not reach this number if some other termination
%                   condition happens first (see remarks on function
%                   aresbuild in user's manual). The default value for
%                   maxFuncs is -1 in which case it is calculated
%                   automatically using formula min(200, max(20, 2d)) + 1,
%                   where d is the number of input variables (Milborrow,
%                   2016). This is fairly arbitrary but can be useful for
%                   first experiments.
%                   To enforce an upper bound on the final model size, use
%                   maxFinalFuncs instead. This is because the forward
%                   phase can see only one basis function ahead while the
%                   backward pruning phase can choose any of the built
%                   basis functions to include in the final model.
%   c             : Generalized Cross-Validation (GCV) penalty per knot.
%                   Larger values for c will lead to fewer knots (i.e., the
%                   final model will have fewer basis functions). A value
%                   of 0 penalizes only terms, not knots (can be useful,
%                   e.g., with lots of data, low or no noise, and highly
%                   structured underlying function of the data). Generally,
%                   the choice of the value for c should greatly depend on
%                   size of the dataset, how structured is the underlying
%                   function, and how high is the noise level, and mildly
%                   depend on the thoroughness of the optimization
%                   procedure, i.e., on the parameters maxFuncs,
%                   maxInteractions, and useMinSpan (Friedman, 1991a).
%                   Simulation studies suggest values for c in the range of
%                   about 2 to 4 (Friedman, 1991a). The default value for
%                   this parameter is -1 in which case c is chosen
%                   automatically using the following rule: if
%                   maxInteractions = 1 (additive modelling) c = 2,
%                   otherwise c = 3. These are the values recommended in
%                   Friedman, 1991a.
%   cubic         : Whether to use piecewise-cubic (true) or piecewise-
%                   linear (false) type of modelling. In general, it is
%                   expected that the piecewise-cubic modelling will give
%                   better predictive performance for smoother and less
%                   noisy data. (default value = true)
%   cubicFastLevel : aresbuild implements three levels of piecewise-cubic
%                   modelling. In level 0, cubic modelling for each
%                   candidate model is done in both phases of the method
%                   (slow). In level 1, cubic modelling is done only in the
%                   backward phase (much faster). In level 2, cubic
%                   modelling is done after both phases, only for the final
%                   model (fastest). The default level is 2 (and it
%                   corresponds to the recommendations in Friedman, 1991a).
%                   Levels 0 and 1 may bring extra accuracy in the
%                   situations when, e.g., the underlying function of the
%                   data has sharp thresholds that for piecewise-cubic
%                   modelling require knot placements different that those
%                   required for piecewise-linear modelling.
%   selfInteractions : This is experimental feature. The maximum degree of
%                   self interactions for any input variable. It can be set
%                   larger than 1 only for piecewise-linear modelling. The
%                   default, and recommended, value = 1, no self
%                   interactions.
%   maxInteractions : The maximum degree of interactions between input
%                   variables. Set to 1 (default) for additive modelling
%                   (i.e., no interactions). For maximal interactivity
%                   between the variables, set the parameter to
%                   d * selfInteractions, where d is the number of input
%                   variables – this way the modelling procedure will have
%                   the most freedom building a complex model. Set to -1,
%                   so that aresbuild sets it automatically equal to d
%                   (maximal interactivity when self interactions are not
%                   used).
%   threshold     : One of the stopping criteria for the forward phase (see
%                   remarks section of function aresbuild in user's manual
%                   for details). Default value = 1e-4. For noise-free
%                   data, the value may be lowered (e.g., to 1e-6) but
%                   setting it to 0 can cause numerical issues and
%                   instability.
%   prune         : Whether to perform model pruning (the backward phase).
%                   (default value = true)
%   fastK         : Parameter (integer) for Fast MARS algorithm (Friedman,
%                   1993, Section 3.0). Maximum number of parent basis
%                   functions considered at each step of the forward phase.
%                   Typical values for fastK are 20, 10, 5 (default value =
%                   Inf, i.e., no Fast MARS). With lower fastK values
%                   model building is faster at the expense of some
%                   accuracy. Good starting values for exploratory work
%                   are fastK = 20, fastBeta = 1, fastH = 5 (Friedman,
%                   1993). Friedman in his paper concluded that changing
%                   the values of fastK and fastH can have big effect on
%                   training computation times but predictive performance
%                   is largely unaffected over a wide range of their values
%                   (Friedman, 1993).
%   fastBeta      : Artificial ageing factor for Fast MARS algorithm
%                   (Friedman, 1993, Section 3.1). Typical value for
%                   fastBeta is 1 (default value = 0, i.e., no artificial
%                   ageing). The parameter is ignored if fastK = Inf.
%   fastH         : Parameter (integer) for Fast MARS algorithm (Friedman,
%                   1993, Section 4.0). Number of iterations till next full
%                   optimization over all input variables for each parent
%                   basis function. Larger values make the search faster.
%                   Typical values for fastH are 1, 5, 10 (default value =
%                   1, i.e., full optimization in every iteration).
%                   Computational reduction associated with increasing
%                   fastH is most pronounced for data sets with many input
%                   variables and when large fastK is used. There seems to
%                   be little gain in increasing fastH beyond 5 (Friedman,
%                   1993). The parameter is ignored if fastK = Inf.
%   useMinSpan    : In order to lower the local variance of the estimates,
%                   a minimum span is imposed that makes the method
%                   resistant to runs of positive or negative error values
%                   between knots (by taking every useMinSpan-th
%                   observation for knot placement) (Friedman, 1991a).
%                   Setting useMinSpan to -1 (default), enables automatic
%                   mode (see remarks in user's manual). Setting useMinSpan
%                   to 0 or 1, disables the protection so that all the
%                   observations are considered for knot placement (except,
%                   see useEndSpan). Setting useMinSpan to > 1, enables
%                   manual tuning of the value. Disabling or lowering
%                   useMinSpan may allow creating a model which is more
%                   responsive to local variations in the data (can be
%                   especially useful if the number of data observations is
%                   small and noise is low) however this can also lead to
%                   overfitting. For further information and examples of
%                   usage, see Section 3.4 in user's manual.
%   useEndSpan    : In order to lower the local variance of the estimates
%                   near the ends of data intervals, a minimum span is
%                   imposed that makes the method resistant to runs of
%                   positive or negative error values between extreme knot
%                   locations and the corresponding ends of data intervals
%                   (by not allowing to place knots too near to the ends of
%                   the intervals) (Friedman, 1991a). Setting useEndSpan to
%                   -1 (default), enables automatic mode that chooses value
%                   for this parameter depending on the number of input
%                   variables (but never lower than 7). Setting useEndSpan
%                   to 0, disables the protection so that all the
%                   observations are considered for knot placement (except,
%                   see useMinSpan). Setting useEndSpan to > 1, enables
%                   manual tuning of the value. Disabling or lowering
%                   useEndSpan may allow creating a model which is more
%                   responsive to local variations near the edges of the
%                   data (can be especially useful if the number of data
%                   observations is small and noise is low) however this
%                   can also lead to overfitting. For further information
%                   and examples of usage, see remarks for aresparams and
%                   Section 3.4 in user's manual.
%   maxFinalFuncs : Maximum number of basis functions (including the
%                   intercept term) in the final pruned model (default
%                   value = Inf). Use this (rather than the maxFuncs
%                   parameter) to enforce an upper bound on the final model
%                   size. See Section 3.3 in user's manual for an example
%                   on how to choose value for this parameter using Cross-
%                   Validation.
%   endSpanAdjust : For basis functions with variable interactions,
%                   useEndSpan gets multiplied by this value. This reduces
%                   probability of getting overfitted interaction terms
%                   supported by just a few observations on the boundaries
%                   of data intervals. Still, at least one knot will always
%                   be allowed in the middle, even if endSpanAdjust would
%                   prohibit it. Useful values range from 1 to 10. (default
%                   value = 1, i.e., no adjustment)
%   newVarPenalty : Penalty for adding a new variable to a model in the
%                   forward phase. This is the gamma parameter of Eq. 74 in
%                   the original paper (Friedman, 1991a). The higher is the
%                   penalty, the more reluctant will be the forward phase
%                   to add a new variable to the model - it will rather try
%                   to use variables already in the model. This can be
%                   useful when some of the variables are highly collinear.
%                   As a result, the final model may be easier to interpret
%                   although usually the built models also will have worse
%                   predictive performance. Useful non-zero values
%                   typically range from 0.01 to 0.2 (Milborrow, 2016).
%                   (default value = 0, i.e., no penalty)
%   terminateWhenInfGCV : Whether to check termination condition,
%                   terminating forward phase when the effective number of
%                   parameters of a model reaches the number of
%                   observations in training data, i.e., when GCV for
%                   such large models would be Inf (see remarks section in
%                   description of function aresbuild on GCV in user's
%                   manual). In such cases it could be pointless to
%                   continue because larger models wouldn't be considered
%                   as candidates for final model anyway. On the other
%                   hand, some of the added basis functions could still
%                   turn out to be useful for inclusion in final model.
%                   Note that the effective number of parameters is not the
%                   same as the number of regression coefficients, except
%                   when c = 0 (in which case enabling terminateWhenInfGCV
%                   has no additional effect). (default value = false)
%   yesInteract   : A matrix indicating pairs of input variables that are
%                   allowed to interact with each other in the ARES model.
%                   The matrix must have two columns. Each row is a pair of
%                   indices for the input variables. Default value = [].
%                   Cannot be used together with noInteract.
%   noInteract    : A matrix indicating pairs of input variables that are
%                   not allowed to interact with each other in the ARES
%                   model. The matrix must have two columns. Each row is a
%                   pair of indices for the input variables. Default value
%                   = []. Cannot be used together with yesInteract.
%   allowLinear   : Whether to allow input variables to enter basis
%                   functions linearly, i.e., without hinge functions.
%                   Such basis functions are added to the model one at a
%                   time, as opposed to basis functions with new hinges
%                   that are added two at a time - one for each hinge of
%                   the reflected pair. Set allowLinear to 0 (default), to
%                   disallow variables entering linearly, i.e., consider
%                   hinge functions only (except see forceLinear). Set to
%                   1, to allow, and treat error reduction associated with
%                   adding such basis function the same way as for a pair
%                   of basis functions with new hinges. Set to 2, to prefer
%                   variables entering basis functions linearly. This is
%                   done by calculating error reduction of such basis
%                   functions using GCV (instead of sum of squared error),
%                   resulting in preference of adding a single basis
%                   function instead of two even when this produces
%                   slightly smaller error reduction. Note that the R
%                   package earth (Milborrow, 2016) has options "0" and "1"
%                   while py-earth (Rudy, 2016) has options "0" and "2".
%   forceLinear   : A vector of indices of input variables that should be
%                   forced to enter the model only linearly, i.e., without
%                   hinge functions. This overrides allowLinear for the
%                   listed variables. Note that forceLinear does not say
%                   that a variable must enter the model; only that if it
%                   enters, it enters linearly. Also note that it has
%                   nothing to do with whether a variable is allowed to
%                   interact with other variables. Default value = [].
%
% Output:
%   trainParams   : A structure of parameters for further use with
%                   aresbuild, arescv, and arescvc functions containing the
%                   provided values (or defaults, if not provided).

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

if (nargin < 1) || isempty(maxFuncs)
    trainParams.maxFuncs = -1; % automatic
else
    if ischar(maxFuncs)
        error('Bad argument value.');
    end
    trainParams.maxFuncs = maxFuncs;
end

if (nargin < 2) || isempty(c)
    trainParams.c = -1; % automatic
else
    trainParams.c = c;
end

if (nargin < 3) || isempty(cubic)
    trainParams.cubic = true;
else
    trainParams.cubic = cubic;
end

if (nargin < 4) || isempty(cubicFastLevel)
    trainParams.cubicFastLevel = 2;
else
    trainParams.cubicFastLevel = cubicFastLevel;
end

if (nargin < 5) || isempty(selfInteractions)
    trainParams.selfInteractions = 1;
else
    trainParams.selfInteractions = selfInteractions;
end

if (nargin < 6) || isempty(maxInteractions)
    trainParams.maxInteractions = 1; % applicable maximum is d * trainParams.selfInteractions
else
    trainParams.maxInteractions = maxInteractions;
end

if (nargin < 7) || isempty(threshold)
    trainParams.threshold = 1e-4;
else
    trainParams.threshold = threshold;
end

if (nargin < 8) || isempty(prune)
    trainParams.prune = true;
else
    trainParams.prune = prune;
end

if (nargin < 9) || isempty(fastK)
    trainParams.fastK = Inf;
else
    trainParams.fastK = fastK;
end

if (nargin < 10) || isempty(fastBeta)
    trainParams.fastBeta = 0;
else
    trainParams.fastBeta = fastBeta;
end

if (nargin < 11) || isempty(fastH)
    trainParams.fastH = 1;
else
    trainParams.fastH = fastH;
end

if (nargin < 12) || isempty(useMinSpan)
    trainParams.useMinSpan = -1; % automatic
else
    if useMinSpan == 0
        trainParams.useMinSpan = 1; % 1 and 0 is the same here (no minspan)
    else
        trainParams.useMinSpan = useMinSpan;
    end
end

if (nargin < 13) || isempty(useEndSpan)
    trainParams.useEndSpan = -1; % automatic
else
    trainParams.useEndSpan = useEndSpan;
end

if (nargin < 14) || isempty(maxFinalFuncs)
    trainParams.maxFinalFuncs = Inf;
else
    trainParams.maxFinalFuncs = maxFinalFuncs;
end

if (nargin < 15) || isempty(endSpanAdjust)
    trainParams.endSpanAdjust = 1;
else
    trainParams.endSpanAdjust = endSpanAdjust;
end

if (nargin < 16) || isempty(newVarPenalty)
    trainParams.newVarPenalty = 0;
else
    trainParams.newVarPenalty = newVarPenalty;
end

if (nargin < 17) || isempty(terminateWhenInfGCV)
    trainParams.terminateWhenInfGCV = false;
else
    trainParams.terminateWhenInfGCV = terminateWhenInfGCV;
end

if (nargin < 18) || isempty(yesInteract)
    trainParams.yesInteract = [];
else
    trainParams.yesInteract = yesInteract;
end

if (nargin < 19) || isempty(noInteract)
    trainParams.noInteract = [];
else
    trainParams.noInteract = noInteract;
end

if (nargin < 20) || isempty(allowLinear)
    trainParams.allowLinear = 0;
else
    trainParams.allowLinear = allowLinear;
end

if (nargin < 21) || isempty(forceLinear)
    trainParams.forceLinear = [];
else
    trainParams.forceLinear = forceLinear;
end

return
