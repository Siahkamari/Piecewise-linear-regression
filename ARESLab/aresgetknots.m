function knots = aresgetknots(model, variable)
% aresgetknots
% Gets all knot locations of an ARES model for the specified input
% variable. A knot is added to the list only if the variable entered a
% basis function non-linearly, i.e., using a hinge function.
%
% Call:
%   knots = aresgetknots(model, variable)
%
% For datasets with one input variable, only the first input argument is
% used. For datasets with more than one input variable, both input
% arguments are required.
%
% Input:
%   model         : ARES model or, for multi-response modelling, a cell
%                   array of ARES models.
%   variable      : Index of the input variable.
%
% Output:
%   knots         : Column vector of knot locations.

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

% Last update: May 2, 2016

if (nargin < 1)
    error('Not enough input arguments.');
end
if length(model) > 1
    model = model{1};
end
if (nargin < 2) && (length(model.minX) > 1)
    error('Not enough input arguments.');
end
if length(model.minX) == 1
    variable = 1;
else
    if length(variable) ~= 1
        error('Input argument variable should be scalar.');
    end
end

knots = [];
nBasis = length(model.knotsites);
if nBasis > 0
    for i = 1 : nBasis
        which = (model.knotdims{i} == variable) & (model.knotdirs{i} ~= 2);
        knots = union(knots, model.knotsites{i}(which));
    end
    knots = knots(:);
end
return
