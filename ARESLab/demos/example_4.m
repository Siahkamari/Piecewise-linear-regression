
% See Section 3.4 in user's manual for details.

% First example

clear; clc;

X = (0:0.05:1)' * pi * 3;
Y = sin(X) + randn(21,1) * 0.1;
Xsin = (0:0.01:1)' * pi * 3;
Ysin = sin(Xsin);

params = aresparams2();
model = aresbuild(X, Y, params);
aresplot(model,[],[],[],[],[],[],'LineWidth',2,'XLim',[0,pi*3],'YLim',[-1.5,1.5]);
hold on;
plot(X, Y, '.', 'MarkerSize', 20); plot(Xsin, Ysin, '-r');

params = aresparams2('useMinSpan', 2, 'useEndSpan', 2);
model = aresbuild(X, Y, params);
aresplot(model,[],[],[],[],[],[],'LineWidth',2,'XLim',[0,pi*3],'YLim',[-1.5,1.5]);
hold on;
plot(X, Y, '.', 'MarkerSize', 20); plot(Xsin, Ysin, '-r');

%%

% Second example

clear; clc;

X = (0:0.05:1)';
Y = [ones(1,7)*3 ones(1,7) ones(1,7)*2]';

params = aresparams2('cubic', false);
model = aresbuild(X, Y, params);
aresplot(model, [], [], [], [], [], [], 'LineWidth', 2, 'YLim', [0.5, 3.5]);
hold on;
plot(X, Y, '.', 'MarkerSize', 20);

params = aresparams2('cubic', false, 'useMinSpan', 2, 'useEndSpan', 2);
model = aresbuild(X, Y, params);
aresplot(model, [], [], [], [], [], [], 'LineWidth', 2, 'YLim', [0.5, 3.5]);
hold on;
plot(X, Y, '.', 'MarkerSize', 20);

params = aresparams2('cubic', false, 'useMinSpan', 1, 'useEndSpan', 6);
model = aresbuild(X, Y, params);
aresplot(model, [], [], [], [], [], [], 'LineWidth', 2, 'YLim', [0.5, 3.5]);
hold on;
plot(X, Y, '.', 'MarkerSize', 20);
