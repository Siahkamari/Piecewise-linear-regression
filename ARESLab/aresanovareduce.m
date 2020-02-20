function [model, usedBasis] = aresanovareduce(model, varsToStay, exact)
% aresanovareduce
% Deletes all the basis functions from ARES model (without recalculating
% model's coefficients and relocating additional knots of piecewise-cubic
% models) in which at least one used variable is not in the given list of
% allowed variables. This can be used to perform ANOVA decomposition as
% well as for investigation of individual and joint contributions of
% variables in the model, i.e., the reduced model can then be plotted to
% visualize the contributions.
% For multi-response modelling, supply one submodel at a time.
%
% Call:
%   [model, usedBasis] = aresanovareduce(model, varsToStay, exact)
%
% Input:
%   model         : ARES model.
%   varsToStay    : A vector of indices for input variables to stay in the
%                   model. The size of the vector should be between one and
%                   the total number of input variables.
%   exact         : Set this to true to get a model with only those basis
%                   functions where the exact combination of variables is
%                   present (default value = false). This is used from
%                   function aresanova.
%
% Output:
%   model         : Reduced ARES model.
%   usedBasis     : Vector of original indices for basis functions still in
%                   use.

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
if isempty(varsToStay)
    error('varsToStay is empty.');
end
if length(model) > 1
    error('This function works with single-response models only. You can supply one submodel at a time.');
else
    if iscell(model)
        model = model{1};
    end
end
if (nargin < 3) || isempty(exact)
    exact = false;
end

nBasis = length(model.knotdims);
notvars = setdiff(1:length(model.minX), varsToStay);
stay = true(1,nBasis);

for i = 1 : nBasis
    if exact && (length(unique(model.knotdims{i})) ~= length(varsToStay))
        stay(i) = false;
        continue;
    end
    for j = 1 : length(notvars)
        if any(model.knotdims{i} == notvars(j))
            stay(i) = false;
            break;
        end
    end
end

usedBasis = find(stay);

for i = nBasis : -1 : 1 % deleting in reverse order
    if ~stay(i)
        model.coefs(i+1) = [];
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
end

return

function parents = updateParents(parents, deletedIdx)
% Updates direct parent indices after deletion of a basis function.
parents(parents == deletedIdx) = 0;
tmp = parents > deletedIdx;
parents(tmp) = parents(tmp) - 1;
return
