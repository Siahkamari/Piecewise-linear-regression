function eq = areseq(model, precision, varNames, binarySimple, expandParentBF, cubicSmoothing)
% areseq
% Prints equations of ARES model.
% For multi-response modelling, supply one submodel at a time.
%
% Call:
%   eq = areseq(model, precision, varNames, binarySimple, ...
%           expandParentBF, cubicSmoothing)
%
% All the input arguments, except the first one, are optional. Empty values
% are also accepted (the corresponding defaults will be used).
%
% Input:
%   model         : ARES model.
%   precision     : Number of digits in the model coefficients and knot
%                   sites. Default value = 15.
%   varNames      : A cell array of variable names to show instead of the
%                   generic ones.
%   binarySimple  : Whether to simplify basis functions that use binary
%                   input variables (default value = false). Note that
%                   whether a variable is binary is automatically
%                   determined during model building in aresbuild by
%                   counting unique values for each variable in training
%                   data. Therefore a variable can also be taken as binary
%                   by mistake if the data for some reason includes only
%                   two values for the variable. You can correct such
%                   mistakes by editing model.isBinary. Also note that
%                   whether a variable is binary does not influence
%                   building of models. It's just used here to simplify
%                   equations.
%                   The argument has no effect if the model was allowed to
%                   have input variables to enter linearly, because then
%                   all binary variables are handled using linear functions
%                   instead of hinge functions.
%   expandParentBF : A basis function that involves multiplication of two
%                   or more hinge functions can be defined simply as a
%                   multiplication of an already existing basis function
%                   (parent) and a new hinge function. Alternatively, it
%                   can be defined as a multiplication of a number of hinge
%                   functions. Set expandParentBF to false (default) for
%                   the former behaviour and to true for the latter.
%   cubicSmoothing : This is for piecewise-cubic models only. Set to
%                   'short' (default) to show piecewise-cubic basis
%                   functions in their short mathematical form (Equation 34
%                   in Friedman, 1991a). Set to 'full' to show all
%                   computations involved in calculating the response
%                   value. Set to 'hide' to hide cubic smoothing and see
%                   the model as if it would be piecewise-linear. It's
%                   easier to understand the equations if smoothing is
%                   hidden. Note that, while the model then looks like
%                   piecewise-linear, the coefficients are for the actual
%                   piecewise-cubic model.
%
% Output:
%   eq            : A cell array of strings containing equations for
%                   individual basis functions and the main model.

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
if (nargin < 2) || isempty(precision)
    precision = 15;
end
if (nargin >= 3)
    if (~isempty(varNames)) && (length(varNames) ~= length(model.minX))
        error('Wrong number of names in varNames.');
    end
else
    varNames = [];
end
if (nargin < 4) || isempty(binarySimple)
    binarySimple = false;
end
if (nargin < 5) || isempty(expandParentBF)
    expandParentBF = false;
end
if (nargin < 6) || isempty(cubicSmoothing)
    cubicSmoothing = 'short';
else
    if ~strcmpi(cubicSmoothing, {'hide' 'short' 'full'})
        error('Wrong value for cubicSmoothing.');
    end
end

p = ['%.' num2str(precision) 'g'];
eq = {};
isPrint = nargout <= 0;

% compose the individual basis functions
for i = 1 : length(model.knotdims)
    func = ['BF' num2str(i) ' ='];
    if (~model.trainParams.cubic) || ~strcmpi(cubicSmoothing, 'full')
        func = [func ' ' getbfstr(model, i, p, binarySimple, expandParentBF, varNames, strcmpi(cubicSmoothing, 'hide'))];
    else
        if (~expandParentBF) && (model.parents(i) > 0)
            func = [func ' BF' num2str(model.parents(i)) ' *'];
            start = length(model.knotdims{i});
        else
            start = 1;
        end
        for j = start : length(model.knotdims{i})
            % if the knot is on the very edge, treat the basis function as linear
            if (model.knotdirs{i}(j) == 2) || ...
               ((model.knotdirs{i}(j) > 0) && (model.knotsites{i}(j) <= model.minX(model.knotdims{i}(j)))) || ...
               ((model.knotdirs{i}(j) < 0) && (model.knotsites{i}(j) >= model.maxX(model.knotdims{i}(j))))
                func = [func ' ' getbfstr(model, i, p, binarySimple, expandParentBF, varNames, strcmpi(cubicSmoothing, 'hide'), j)];
                continue;
            end
            t = num2str(model.knotsites{i}(j),p);
            t1 = num2str(model.t1(i,model.knotdims{i}(j)),p);
            t2 = num2str(model.t2(i,model.knotdims{i}(j)),p);
            pp = ['p' num2str(i) '_' num2str(j)];
            rr = ['r' num2str(i) '_' num2str(j)];
            if isempty(varNames)
                d = ['x' num2str(model.knotdims{i}(j),p)];
            else
                d = varNames{model.knotdims{i}(j)};
            end
            f = ['f' num2str(i) '_' num2str(j)];
            if model.knotdirs{i}(j) > 0 % here the hinge function looks like "_/"
                iff = ['if (' d ' <= ' t1 ') then ' f ' = 0'];
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                
                iff = ['if (' t1 ' < ' d ' < ' t2 ') then begin'];
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                pPoz = ['  ' pp ' = (2*(' t2 ') + (' t1 ') - 3*(' t ')) / ((' t2 ') - (' t1 '))^2'];
                if isPrint, disp(pPoz); end
                eq{end+1,1} = pPoz;
                rPoz = ['  ' rr ' = (2*(' t ') - (' t2 ') - (' t1 ')) / ((' t2 ') - (' t1 '))^3'];
                if isPrint, disp(rPoz); end
                eq{end+1,1} = rPoz;
                iff = ['  ' f ' = ' pp ' * (' d ' - (' t1 '))^2 + ' rr ' * (' d ' - (' t1 '))^3'];
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                iff = 'end';
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                
                iff = ['if (' d ' >= ' t2 ') then ' f ' = ' d ' - (' t ')'];
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                
                func = [func ' ' f];
            else % here the hinge function looks like "\_"
                iff = ['if (' d ' <= ' t1 ') then ' f ' = -(' d ' - (' t '))'];
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                
                iff = ['if (' t1 ' < ' d ' < ' t2 ') then begin'];
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                pNeg = ['  ' pp ' = (3*(' t ') - 2*(' t1 ') - (' t2 ')) / ((' t1 ') - (' t2 '))^2'];
                if isPrint, disp(pNeg); end
                eq{end+1,1} = pNeg;
                rNeg = ['  ' rr ' = ((' t1 ') + (' t2 ') - 2*(' t ')) / ((' t1 ') - (' t2 '))^3'];
                if isPrint, disp(rNeg); end
                eq{end+1,1} = rNeg;
                iff = ['  ' f ' = ' pp ' * (' d ' - (' t2 '))^2 + ' rr ' * (' d ' - (' t2 '))^3'];
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                iff = 'end';
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                
                iff = ['if (' d ' >= ' t2 ') then ' f ' = 0'];
                if isPrint, disp(iff); end
                eq{end+1,1} = iff;
                
                func = [func ' ' f];
            end
            
            if j < length(model.knotdims{i})
                func = [func ' *'];
            end
        end
    end
    if isPrint, disp(func); end
    eq{end+1,1} = func;
end

% compose the summation
func = ['y = ' num2str(model.coefs(1),p)];
for i = 1 : length(model.knotdims)
    if model.coefs(i+1) >= 0
        func = [func ' +'];
    else
        func = [func ' '];
    end
    func = [func num2str(model.coefs(i+1),p) '*BF' num2str(i)];
end
if isPrint, disp(func); end
eq{end+1,1} = func;

if model.trainParams.cubic
    if strcmpi(cubicSmoothing, 'hide')
        if isPrint, fprintf('\n'); end
        disp('WARNING: Piecewise-cubic spline smoothing hidden.');
    end
end

return
