function func = getbfstr(model, bfidx, p, binarySimple, expandParentBF, varNames, cubicAsLinear, dimidx)
% Forms printable equation for the specified basis function.

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

if (nargin < 8) || isempty(dimidx)
    if (~expandParentBF) && (model.parents(bfidx) > 0)
        func = ['BF' num2str(model.parents(bfidx)) ' *'];
        start = length(model.knotdims{bfidx});
    else
        func = '';
        start = 1;
    end
    dimidx = start : length(model.knotdims{bfidx});
else
    func = '';
end
for j = dimidx
    if ~isempty(func)
        func = [func ' '];
    end
    if model.knotdirs{bfidx}(j) == 2 % here the hinge function is actually a linear function "/"
        if isempty(varNames)
            func = [func 'x' num2str(model.knotdims{bfidx}(j),p)];
        else
            func = [func varNames{model.knotdims{bfidx}(j)}];
        end
    elseif model.knotdirs{bfidx}(j) > 0 % here the hinge function looks like "_/"
        if model.knotsites{bfidx}(j) > 0
            m = '';
        else
            m = '+';
        end
        if binarySimple && model.isBinary(model.knotdims{bfidx}(j))
            if (length(model.knotdims{bfidx}) > 1) && (model.knotsites{bfidx}(j) ~= 0)
                func = [func '('];
            end
            if isempty(varNames)
                func = [func 'x' num2str(model.knotdims{bfidx}(j),p)];
            else
                func = [func varNames{model.knotdims{bfidx}(j)}];
            end
            if model.knotsites{bfidx}(j) ~= 0
                func = [func ' ' m num2str(-model.knotsites{bfidx}(j),p)];
            end
            if (length(model.knotdims{bfidx}) > 1) && (model.knotsites{bfidx}(j) ~= 0)
                func = [func ')'];
            end
        else
            if (~model.trainParams.cubic) || cubicAsLinear || ...
               ((model.knotdirs{bfidx}(j) > 0) && (model.knotsites{bfidx}(j) <= model.minX(model.knotdims{bfidx}(j)))) || ...
               ((model.knotdirs{bfidx}(j) < 0) && (model.knotsites{bfidx}(j) >= model.maxX(model.knotdims{bfidx}(j))))
                if isempty(varNames)
                    func = [func 'max(0, x' num2str(model.knotdims{bfidx}(j),p) ' ' m num2str(-model.knotsites{bfidx}(j),p) ')'];
                else
                    func = [func 'max(0, ' varNames{model.knotdims{bfidx}(j)} ' ' m num2str(-model.knotsites{bfidx}(j),p) ')'];
                end
            else
                if isempty(varNames)
                    func = [func 'C(x' num2str(model.knotdims{bfidx}(j),p)];
                else
                    func = [func 'C(' varNames{model.knotdims{bfidx}(j)}];
                end
                func = [func '|+1,' ...
                        num2str(model.t1(bfidx,model.knotdims{bfidx}(j)),p) ',' ...
                        num2str(model.knotsites{bfidx}(j),p) ',' ...
                        num2str(model.t2(bfidx,model.knotdims{bfidx}(j)),p) ')'];
            end
        end
    else % here the hinge function looks like "\_"
        if binarySimple && model.isBinary(model.knotdims{bfidx}(j))
            if (length(model.knotdims{bfidx}) > 1) && (model.knotsites{bfidx}(j) ~= 0)
                func = [func '('];
            end
            if model.knotsites{bfidx}(j) ~= 0
                func = [func num2str(model.knotsites{bfidx}(j),p) ' '];
            end
            if isempty(varNames)
                func = [func '-x' num2str(model.knotdims{bfidx}(j),p)];
            else
                func = [func '-' varNames{model.knotdims{bfidx}(j)}];
            end
            if (length(model.knotdims{bfidx}) > 1) && (model.knotsites{bfidx}(j) ~= 0)
                func = [func ')'];
            end
        else
            if (~model.trainParams.cubic) || cubicAsLinear || ...
               ((model.knotdirs{bfidx}(j) > 0) && (model.knotsites{bfidx}(j) <= model.minX(model.knotdims{bfidx}(j)))) || ...
               ((model.knotdirs{bfidx}(j) < 0) && (model.knotsites{bfidx}(j) >= model.maxX(model.knotdims{bfidx}(j))))
                if isempty(varNames)
                    func = [func 'max(0,' num2str(model.knotsites{bfidx}(j),p) ' -x' num2str(model.knotdims{bfidx}(j),p) ')'];
                else
                    func = [func 'max(0, ' num2str(model.knotsites{bfidx}(j),p) ' -' varNames{model.knotdims{bfidx}(j)} ')'];
                end
            else
                if isempty(varNames)
                    func = [func 'C(x' num2str(model.knotdims{bfidx}(j),p)];
                else
                    func = [func 'C(' varNames{model.knotdims{bfidx}(j)}];
                end
                func = [func '|-1,' ...
                        num2str(model.t1(bfidx,model.knotdims{bfidx}(j)),p) ',' ...
                        num2str(model.knotsites{bfidx}(j),p) ',' ...
                        num2str(model.t2(bfidx,model.knotdims{bfidx}(j)),p) ')'];
            end
        end
    end
    if j < length(model.knotdims{bfidx})
        func = [func ' *'];
    end
end
return
