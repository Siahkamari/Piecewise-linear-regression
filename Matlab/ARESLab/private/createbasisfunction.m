function Xn = createbasisfunction(X, Xtmp, knotdims, knotsites, knotdirs, parent, minX, maxX, t1, t2)
% Creates a list of response values of a defined basis function for either
% a piecewise-linear or piecewise-cubic model.

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

n = size(X,1);

% if the basis function is always zero (it's on the edge), return NaN
% (we check only the last element because all others are already checked)
if ((knotdirs(end) > 0) && (knotsites(end) >= maxX(knotdims(end)))) || ...
   ((knotdirs(end) < 0) && (knotsites(end) <= minX(knotdims(end))))
    Xn = NaN(n,1);
    return
end

if nargin < 10 % piecewise-linear
    
    if parent > 0
        % If the basis function has a direct parent in the model, use it to speed-up the calculations
        if knotdirs(end) == 2
            Xn = Xtmp(:,parent+1) .* X(:,knotdims(end)); % variable enters linearly
        else
            if knotdirs(end) > 0
                z = X(:,knotdims(end)) - knotsites(end);
            else
                z = knotsites(end) - X(:,knotdims(end));
            end
            Xn = Xtmp(:,parent+1) .* max(0,z);
        end
    else
        if knotdirs(1) == 2
            Xn = X(:,knotdims(1)); % variable enters linearly
        else
            if knotdirs(1) > 0
                z = X(:,knotdims(1)) - knotsites(1);
            else
                z = knotsites(1) - X(:,knotdims(1));
            end
            Xn = max(0,z);
        end
        len = length(knotdims);
        for i = 2 : len
            if knotdirs(i) == 2
                Xn = Xn .* X(:,knotdims(i)); % variable enters linearly
            else
                if knotdirs(i) > 0
                    z = X(:,knotdims(i)) - knotsites(i);
                else
                    z = knotsites(i) - X(:,knotdims(i));
                end
                Xn = Xn .* max(0,z);
            end
        end
    end
    
else % piecewise-cubic
    
    if parent > 0
        % If the basis function has a direct parent in the model, use it to speed-up the calculations
        Xn = Xtmp(:,parent+1);
        start = length(knotdims);
    else
        start = 1;
    end
    
    Xx = zeros(n,1);
    for i = start : length(knotdims)
        if knotdirs(i) == 2
            Xx(:) = X(:,knotdims(i)); % variable enters linearly
        elseif (knotdirs(i) > 0) && (knotsites(i) <= minX(knotdims(i)))
            % if the knot is on the very edge, treat the basis function as piecewise-linear
            Xx(:) = max(0, X(:,knotdims(i)) - knotsites(i));
        elseif (knotdirs(i) < 0) && (knotsites(i) >= maxX(knotdims(i)))
            % if the knot is on the very edge, treat the basis function as piecewise-linear
            Xx(:) = max(0, knotsites(i) - X(:,knotdims(i)));
        else
            tt1 = t1(knotdims(i));
            tt2 = t2(knotdims(i));
            if knotdirs(i) > 0
                % p = (2*tt2 + tt1 - 3*knotsites(i)) / (tt2 - tt1)^2; % unoptimized
                % r = (2*knotsites(i) - tt2 - tt1) / (tt2 - tt1)^3; % unoptimized
                tt2_tt1 = tt2 - tt1;
                tt2_tt1_sq = tt2_tt1 * tt2_tt1;
                p = (2*tt2 + tt1 - 3*knotsites(i)) / tt2_tt1_sq;
                r = (2*knotsites(i) - tt2 - tt1) / (tt2_tt1_sq*tt2_tt1);
                for j = 1 : n
                    tmpx = X(j,knotdims(i));
                    if tmpx <= tt1
                        Xx(j) = 0;
                    elseif tmpx < tt2
                        % Xx(j) = p*(tmpx-tt1)^2 + r*(tmpx-tt1)^3; % unoptimized
                        tmpx_tt1 = tmpx - tt1;
                        tmpx_tt1_sq = tmpx_tt1 * tmpx_tt1;
                        Xx(j) = p*tmpx_tt1_sq + r*tmpx_tt1_sq*tmpx_tt1;
                    else
                        Xx(j) = tmpx - knotsites(i);
                    end
                end
            else
                % p = (3*knotsites(i) - 2*tt1 - tt2) / (tt1 - tt2)^2; % unoptimized
                % r = (tt1 + tt2 - 2*knotsites(i)) / (tt1 - tt2)^3; % unoptimized
                tt1_tt2 = tt1 - tt2;
                tt1_tt2_sq = tt1_tt2 * tt1_tt2;
                p = (3*knotsites(i) - 2*tt1 - tt2) / tt1_tt2_sq;
                r = (tt1 + tt2 - 2*knotsites(i)) / (tt1_tt2_sq*tt1_tt2);
                for j = 1 : n
                    tmpx = X(j,knotdims(i));
                    if tmpx <= tt1
                        Xx(j) = knotsites(i) - tmpx;
                    elseif tmpx < tt2
                        % Xx(j) = p*(tmpx-tt2)^2 + r*(tmpx-tt2)^3; % unoptimized
                        tmpx_tt2 = tmpx - tt2;
                        tmpx_tt2_sq = tmpx_tt2 * tmpx_tt2;
                        Xx(j) = p*tmpx_tt2_sq + r*tmpx_tt2_sq*tmpx_tt2;
                    else
                        Xx(j) = 0;
                    end
                end
            end
        end
        if i > 1
            Xn = Xn .* Xx;
        else
            Xn = Xx;
        end
    end

end
return
