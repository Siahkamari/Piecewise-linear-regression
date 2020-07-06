function [t1, t2, diff] = findsideknots(model, knotdim, knotsite, d, minX, maxX, old_t1, old_t2)
% Recomputes side knot placements for all the basis functions in the model
% (in either the knotdim dimension or all the dimensions) while taking into
% account that a new basis function is to be added and its side knots must
% not disrupt the integrity of the whole side knot placement. The function
% is used only for piecewise-cubic modelling.
% The function assumes that there are no self-interactions in any basis
% function.
% t1 is a matrix of knot sites for the knots on the left of the central
% knot. t2 is a matrix of knot sites for the knots on the right of the
% central knot. diff is a list of basis functions for which at least one
% knot has moved.

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

% Last update: April 25, 2016

if isempty(model.knotdims)
    if isempty(knotdim)
        t1 = [];
        t2 = [];
    else
        t1 = inf(1,d);
        t2 = inf(1,d);
        for di = 1 : d
            which = find(knotdim == di);
            if ~isempty(which)
                t1(di) = (minX(di) + knotsite(which)) / 2;
                t2(di) = (knotsite(which) + maxX(di)) / 2;
            end
        end
    end
    diff = [];
else
    if ~isempty(knotdim)
        t1 = old_t1;
        t1(end+1,:) = Inf;
        t2 = old_t2;
        t2(end+1,:) = Inf;
        dims = knotdim; % take dimension numbers from the new knot
    else
        t1 = inf(length(model.knotdims),d);
        t2 = inf(length(model.knotdims),d);
        dims = 1 : d; % take all dimensions
    end
    % "idx" = indices of functions which are in the di dimension
    % "knot" = knotsites of those functions in the di dimension
    % for each dimension, make a list of central knots and place side knots
    for di = dims
        % find all used knotsites in the di dimension
        idx = zeros(length(model.knotdims)+1,1);
        knot = idx;
        count = 0;
        % list existing knots of this dimension
        for i = 1 : length(model.knotdims)
            which = model.knotdims{i} == di;
            if any(which)
                count = count + 1;
                idx(count) = i;
                knot(count) = model.knotsites{i}(which);
            end
        end
        % add the new knot to the list
        if ~isempty(knotdim)
            which = knotdim == di;
            if any(which)
                count = count + 1;
                % here idx(count) is already 0 so we don't set it
                knot(count) = knotsite(which);
            end
        end
        idx = idx(1:count);
        knot = knot(1:count);
        % sort the arrays by knot place
        [knot, ind2] = sort(knot);
        idx = idx(ind2);
        % calculate and store t1, t2
        % (it is possible that two or more functions have the same central
        % knot and so the same side knots)
        i = 1;
        while (i <= length(knot))
            % left side knot t1
            if i == 1
                temp_t1 = (minX(di) + knot(i)) / 2;
            else
                temp_t1 = (knot(i-1) + knot(i)) / 2;
            end
            % store location
            which = find(knot == knot(i));
            for j = 1 : length(which)
                if idx(which(j)) ~= 0
                    t1(idx(which(j)),di) = temp_t1;
                else
                    t1(end,di) = temp_t1;
                end
            end
            % right side knot t2
            if knot(which(end)) == knot(end)
                temp_t2 = (knot(i) + maxX(di)) / 2;
            else
                temp_t2 = (knot(i) + knot(which(end)+1)) / 2;
            end
            % store location
            for j = 1 : length(which)
                if idx(which(j)) ~= 0
                    t2(idx(which(j)),di) = temp_t2;
                else
                    t2(end,di) = temp_t2;
                end
            end
            i = i + length(which);
        end
    end
    % find the difference between old t1,t2 and new t1,t2
    if ~isempty(old_t1)
        if isempty(knotdim)
            diff = any((t1 ~= old_t1) | (t2 ~= old_t2), 2);
        else
            diff = any((t1(1:end-1,:) ~= old_t1) | (t2(1:end-1,:) ~= old_t2), 2);
        end
        diff = find(diff);
        diff = diff';
    else
        diff = [];
    end
end
return
