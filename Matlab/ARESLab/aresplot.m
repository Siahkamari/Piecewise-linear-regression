function fh = aresplot(model, idx, vals, minX, maxX, gridSize, showKnots, varargin)
% aresplot
% Plots ARES model. For datasets with one input variable, plots the model
% together with its knot locations. For datasets with more than one input
% variable, plots 3D surface. If idx is not provided, checks if the model
% uses more than one variable and, if not, plots in 2D even if the dataset
% has more than one input variable.
% For multi-response modelling, supply one submodel at a time.
%
% Call:
%   fh = aresplot(model, idx, vals, minX, maxX, gridSize, showKnots, varargin)
%
% All the input arguments, except the first one, are optional. Empty values
% are also accepted (the corresponding defaults will be used).
%
% Input:
%   model         : ARES model.
%   idx           : Only used when the number of input variables is larger
%                   than two. This is a vector containing two indices for
%                   the two variables values of which are to be varied in
%                   the plot (default value = [1 2]).
%   vals          : Only used when the number of input variables is larger
%                   than two. This is a vector of fixed values for all the
%                   input variables (except that the values for the varied
%                   variables are not used). By default, continuous
%                   variables are fixed at (minX + maxX) / 2 but binary
%                   variables (according to model.isBinary) are fixed at
%                   minX.
%   minX, maxX    : Minimum and maximum values for each input variable
%                   (this is the same type of data as in model.minX and
%                   model.maxX). By default, those values are taken from
%                   model.minX and model.maxX.
%   gridSize      : Grid size for the plot. Default value is 400 for 2D
%                   plots and 50 for 3D plots.
%   showKnots     : Whether to show knots in the plot (default value is
%                   true for data with one input variable and false
%                   otherwise). Showing knot locations in 3D plots is
%                   experimental feature. In a 3D plot, knots for basis
%                   functions without interactions are represented as
%                   planes with white edges while knots for basis functions
%                   with interactions are represented as 90-degrees "broken
%                   planes" with black edges. The directions of the broken
%                   planes depend on directions of hinge functions in the
%                   corresponding basis functions. Planes for each new knot
%                   (or pair of knots) are shown using a new color and a
%                   new vertical offset so that they are easier to
%                   distinguish. Unfortunately, coincident planes flicker;
%                   though it helps seeing that there is more than one
%                   plane.
%                   Note that, for input variables entering linearly (i.e.,
%                   without hinge functions), 2D plots don't show any knots
%                   but 3D plots show knots at minX.
%   varargin      : Name/value pairs of arguments passed to function plot
%                   (for 2D plots) and function surfc (for 3D plots). May
%                   include 'XLim', 'YLim', and 'ZLim' which are separated
%                   and passed to axes.
%
% Input:
%   fh            : Handle to the created figure.

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

if nargin < 1
    error('Not enough input arguments.');
end
if length(model) > 1
    error('This function works with single-response models only. You can supply one submodel at a time.');
else
    if iscell(model)
        model = model{1};
    end
end
if (nargin < 3)
    vals = [];
end
if (nargin < 4) || isempty(minX)
    minX = model.minX;
else
    if length(minX) ~= length(model.minX)
        error('Vector minX is of wrong size.');
    end
end
if (nargin < 5) || isempty(maxX)
    maxX = model.maxX;
else
    if length(maxX) ~= length(model.maxX)
        error('Vector maxX is of wrong size.');
    end
end
if (nargin < 6) || isempty(gridSize)
    gridSize = [];
end
if (nargin < 8) || isempty(varargin)
    varargin = [];
    vararginAxes = [];
else
    cellfind = @(string)(@(cell_contents)(strcmpi(string,cell_contents)));
    vararginAxes = {};
    for str = {'XLim', 'YLim', 'ZLim'}
        i = find(cellfun(cellfind(str), varargin));
        if ~isempty(i)
            vararginAxes = [vararginAxes varargin(i) varargin(i+1)];
            varargin(i+1) = [];
            varargin(i) = [];
        end
    end
end

fh = [];

if (length(model.minX) > 1) && ((nargin < 2) || isempty(idx))
    nBasis = length(model.knotsites);
    force2D = true;
    if nBasis > 0
        variable = [];
        for i = 1 : nBasis
            variable = union(variable, model.knotdims{i});
            if numel(variable) > 1
                force2D = false;
                break;
            end
        end
    else
        variable = 1;
    end
else
    force2D = false;
end

if (length(model.minX) <= 1) || force2D
    if isempty(gridSize)
        gridSize = 400;
    end
    if force2D
        minX = minX(variable);
        maxX = maxX(variable);
    end
    nBasis = length(model.knotsites);
    if nBasis > 0
        if model.trainParams.cubic || (model.trainParams.selfInteractions > 1)
            step = (maxX - minX) / gridSize;
            X = (minX:step:maxX)';
        else
            X = [minX; maxX]; % for piecewise-linear models there are no need in more than this
        end
        knotsX = [];
        % Find all knot locations
        for i = 1 : nBasis
            newKnotX = model.knotsites{i};
            %which = (newKnotX > minX) & (newKnotX < maxX);
            which = (model.knotdirs{i} ~= 2); % not showing knot for variables entering linearly
            knotsX = union(knotsX, newKnotX(which));
        end
        knotsX = knotsX(:);
        X = union(X, knotsX); % adding knot locations to X so that the graph shows them correctly and accurately
    else
        knotsX = [];
        X = [minX; maxX];
    end
    fh = figure;
    if force2D
        Xtmp = zeros(size(X,1), length(model.minX));
        Xtmp(:,variable) = X;
        Y = arespredict(model, Xtmp);
    else
        Y = arespredict(model, X);
    end
    if isempty(varargin)
        plot(X, Y);
    else
        plot(X, Y, varargin{:});
    end
    grid on;
    if force2D
        xlabel(['x' num2str(variable)]);
    else
        xlabel('x');
    end
    ylabel('y');
    if (nargin < 7) || isempty(showKnots)
        showKnots = true;
    end
    if showKnots && (~isempty(knotsX))
        % Add all knots to the plot
        hold on;
        if force2D
            Xtmp = zeros(size(knotsX,1), length(model.minX));
            Xtmp(:,variable) = knotsX;
            knotsY = arespredict(model, Xtmp);
        else
            knotsY = arespredict(model, knotsX);
        end
        plot(knotsX, knotsY, 'ko', 'MarkerSize', 8);
        if ~isempty(vararginAxes)
            set(gca, vararginAxes{:});
        end
        ylim = get(gca, 'ylim');
        for i = 1 : length(knotsX)
            plot([knotsX(i) knotsX(i)], [ylim(1) knotsY(i)], 'k--');
        end
        hold off;
    else
        if ~isempty(vararginAxes)
            set(gca, vararginAxes{:});
        end
    end
    return;
end

if isempty(gridSize)
    gridSize = 50;
end

if (nargin < 2) || isempty(idx)
    ind1 = 1;
    ind2 = 2;
else
    if numel(idx) ~= 2
        error('Vector idx should have exactly two elements.');
    end
    ind1 = idx(1);
    ind2 = idx(2);
end

if isempty(vals)
    vals = (minX + maxX) ./ 2;
    vals(model.isBinary) = minX(model.isBinary);
else
    if length(minX) ~= length(vals)
        error('Vector vals is of wrong size.');
    end
end

% Creating grid
step1 = (maxX(ind1) - minX(ind1)) / gridSize;
step2 = (maxX(ind2) - minX(ind2)) / gridSize;
[X1, X2] = meshgrid(minX(ind1):step1:maxX(ind1), minX(ind2):step2:maxX(ind2));
XX1 = reshape(X1, numel(X1), 1);
XX2 = reshape(X2, numel(X2), 1);

% Fill other columns with the fixed values
X = zeros(size(XX1,1), length(minX));
X(:,ind1) = XX1;
X(:,ind2) = XX2;
for i = 1 : length(minX)
    if (i ~= ind1) && (i ~= ind2)
        X(:,i) = vals(i);
    end
end

% Calculate Y and plot the surface
YY = arespredict(model, X);
Y = reshape(YY, size(X1,1), size(X2,2));
fh = figure;

if ~isempty(vararginAxes)
    % A workaround to always force the contour plot to be at ZLim
    zlimidx = find(strcmpi('ZLim', vararginAxes));
    if ~isempty(zlimidx)
        zlim = vararginAxes{zlimidx + 1};
        hp = plot3(X1(1), X2(1), zlim(1), '.');
        grid on;
        hold on;
    end
end
% Plotting surface
if isempty(varargin)
    surfc(X1, X2, Y);
else
    surfc(X1, X2, Y, varargin{:});
end
% Setting properties for axes
if ~isempty(vararginAxes)
    set(gca, vararginAxes{:});
    if ~isempty(zlimidx)
        hold off;
        set(hp, 'Visible', 'off'); % hiding the workaround
    end
end

xlabel(['x' num2str(ind1)]);
ylabel(['x' num2str(ind2)]);
zlabel('y');

% Visualization of knot locations for 3D plots
if (nargin < 7) || isempty(showKnots)
    showKnots = false;
end
if showKnots
    if model.trainParams.selfInteractions > 1
        disp('Cannot show knots when selfInteractions > 1.');
        return;
    end
    nBasis = length(model.knotsites);
    if nBasis <= 0
        return;
    end
    % First, let's count involved basis functions and check whether they
    % involve any basis functions other than those selected
    countUnique = 0;
    knotdimsOLD = [];
    knotsitesOLD = [];
    for i = 1 : nBasis
        knotdims = model.knotdims{i};
        %if (numel(knotdims) == 1) && (model.knotdirs{i} == 2)
        %    continue; % not showing knots for single variables entering linearly
        %end
        which = (knotdims == ind1) | (knotdims == ind2);
        if any(which) % whether the knots involve one or both of the selected variables
            if any(~which) % whether the knots involve any other variable besides those selected
                disp('Cannot show knots because one or more of the basis functions that involve at least');
                disp('one SELECTED input variable also involve at least one OTHER input variable.');
                disp('You can try deleting those basis functions before plotting.');
                return;
            end
            knotdims = model.knotdims{i};
            knotsites = model.knotsites{i};
            if (numel(knotsites) ~= numel(knotsitesOLD)) || (~all(knotsites == knotsitesOLD)) || ~all(knotdims == knotdimsOLD)
                % incrementing only if the new knots are not identical to
                % the previous ones (because hinges may come in pairs)
                countUnique = countUnique + 1;
            end
            knotdimsOLD = knotdims;
            knotsitesOLD = knotsites;
        end
    end
    if countUnique <= 0
        return;
    end
    hold on;
    xlim = get(gca, 'xlim');
    ylim = get(gca, 'ylim');
    zlim = get(gca, 'zlim');
    hue = (0:(0.9/(countUnique-1)):0.9)'; % a different color for each new plane
    rgb = hsv2rgb([hue ones(size(hue)) ones(size(hue)) * 0.8]);
    countUnique = 0;
    knotdimsOLD = [];
    knotsitesOLD = [];
    for i = 1 : nBasis
        knotdims = model.knotdims{i};
        %if (numel(knotdims) == 1) && (model.knotdirs{i} == 2)
        %    continue; % not showing knots for single variables entering linearly
        %end
        which = (knotdims == ind1) | (knotdims == ind2);
        if any(which) % whether the knots involve one or both of the selected variables
            knotsites = model.knotsites{i};
            if (numel(knotsites) ~= numel(knotsitesOLD)) || (~all(knotsites == knotsitesOLD)) || ~all(knotdims == knotdimsOLD)
                % incrementing only if the new knots are not identical to
                % the previous ones (because hinges may come in pairs)
                countUnique = countUnique + 1;
            end
            knotdimsOLD = knotdims;
            knotsitesOLD = knotsites;
            if numel(knotdims) == 1
                if model.knotdirs{i} == 2 % variable is entering linearly
                    knotsites = minX(model.knotdims{i}); % so that the plane is always on the edge
                end
                % "unbroken" plane
                if knotdims == ind1
                    x = [knotsites knotsites];
                    y = repmat([ylim(1) ylim(2)], 2, 1);
                    z = repmat([zlim(1); zlim(2)], 1, 2) + (zlim(2) - zlim(1)) * 0.012 * countUnique;
                    surf(x, y, z, 'FaceAlpha', 0.5, 'FaceColor', rgb(countUnique,:), 'LineWidth', 2, 'EdgeColor', 'w');
                else
                    x = repmat([xlim(1) xlim(2)], 2, 1);
                    y = [knotsites knotsites];
                    z = repmat([zlim(1); zlim(2)], 1, 2) + (zlim(2) - zlim(1)) * 0.012 * countUnique;
                    surf(x, y, z, 'FaceAlpha', 0.5, 'FaceColor', rgb(countUnique,:), 'LineWidth', 2, 'EdgeColor', 'w');
                end
            else
                % "broken" plane
                knotdirs = model.knotdirs{i};
                idx1 = find(knotdims == ind1, 1);
                idx2 = find(knotdims == ind2, 1);
                
                if model.knotdirs{idx1} == 2 % variable is entering linearly
                    knotsites(idx1) = minX(model.knotdims{idx1}); % so that the plane is always on the edge
                end
                if model.knotdirs{idx2} == 2 % variable is entering linearly
                    knotsites(idx2) = minX(model.knotdims{idx2}); % so that the plane is always on the edge
                end
                
                x = [knotsites(idx1) knotsites(idx1)];
                if knotdirs(idx2) < 0
                    y = repmat([ylim(1) knotsites(idx2)], 2, 1);
                else
                    y = repmat([knotsites(idx2) ylim(2)], 2, 1);
                end
                z = repmat([zlim(1); zlim(2)], 1, 2) + (zlim(2) - zlim(1)) * 0.012 * countUnique;
                surf(x, y, z, 'FaceAlpha', 0.5, 'FaceColor', rgb(countUnique,:), 'LineWidth', 2, 'EdgeColor', 'k');
                
                if knotdirs(idx1) < 0
                    x = repmat([xlim(1) knotsites(idx1)], 2, 1);
                else
                    x = repmat([knotsites(idx1) xlim(2)], 2, 1);
                end
                y = [knotsites(idx2) knotsites(idx2)];
                z = repmat([zlim(1); zlim(2)], 1, 2) + (zlim(2) - zlim(1)) * 0.012 * countUnique;
                surf(x, y, z, 'FaceAlpha', 0.5, 'FaceColor', rgb(countUnique,:), 'LineWidth', 2, 'EdgeColor', 'k');
            end
        end
    end
end
return
