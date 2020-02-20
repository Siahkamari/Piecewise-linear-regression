
% this script prints out Normalized MSE results for regression on real world data-sets 

clear, clc
rng(0);

%% settings
n_run = 5;      % number of experiment averages
dset = 2;       % choice of data set --> see bellow

%% real data
switch dset
    case 1
        load acetylene.mat;
        X = [x1,x2,x3];
    case 2
        load moore.mat
        X = moore(:,2:end);
        y = moore(:,1);
    case 3
        load reaction.mat
        X = reactants;
        y = rate(:,1);
    case 4
        load cereal.mat
        y = Calories;
        X = [Carbo, Cups, Fat, Fiber, Potass, Protein, Shelf, Sodium, Sugars, Type, Vitamins, Weight];
    case 5
        load carsmall.mat
        y = MPG;
        X = [Acceleration, Cylinders, Displacement, Horsepower, Model_Year, Weight];
        X(isnan(X)) = 0;
        y(isnan(y)) = nanmean(y);
    case 6
        load('data/boston.mat')
end

[n, d] = size(X);
X(isnan(X)) = 0;
y(isnan(y)) = 1;

%% initialization
MSE_DC = zeros(n_run, 1);
MSE_MARS = zeros(n_run, 1);
MSE_MLP = zeros(n_run, 1);
MSE_KNN = zeros(n_run, 1);

for run = 1:n_run
    fprintf("\n\nrun %d/%d\n", run, n_run)
    %% train test split
    n_train = floor(4/5*n);
    n_test = n - n_train;
    
    I_train = randsample(1:n, n_train);
    I_test = setdiff(1:n, I_train);
    
    y_train = y(I_train);
    X_train = X(I_train,:);
    y_test = y(I_test);
    X_test = X(I_test,:);
    
    %% D.C
    f_hat = auto_tune_dc_fit(y_train, X_train, "regression");
    y_hat_test = f_hat(X_test);
    MSE_DC(run) = mean((y_hat_test-y_test).^2)/var(y);
    
    %% MLP
    try
        net = auto_tune_mlp(y_train, X_train);
        y_hat_test =  net(X_test')';
        MSE_MLP(run) = mean((y_hat_test-y_test).^2)/var(y);
    catch
        warning('Deep learning toolbox is not installed. no MLP results!!')
    end
    
    %% MARS
    addpath('ARESLab')
    MARS_params = aresparams2('maxFuncs', 101);
    [model, ~, resultsEval] = aresbuild(X_train, y_train, MARS_params);
    results = arestest(model, X_test, y_test);
    MSE_MARS(run) = (results.MSE)/var(y);
    
    %% KNN
    try
        Mdl = auto_tune_KNN(y_train, X_train,"regression");
        y_hat_test = predict(Mdl,X_test);
        MSE_KNN(run) = mean((y_hat_test-y_test).^2)/var(y);
    catch
        warning('Statistics and ML toolbox is not installed. no KNN results!!')
    end
    
end


%% results
zn = 1.96;      % 95 percent interval
fprintf("\n\n %s = %.1f  -/+  %.1f \n","DC ",...
    100*round(mean(MSE_DC),3),100*round(zn*std(MSE_DC)/sqrt(n_run),3));

fprintf("\n\n %s = %.1f  -/+  %.1f \n","MARS ",...
    100*round(mean(MSE_MARS),3),100*round(zn*std(MSE_MARS)/sqrt(n_run),3));

fprintf("\n\n %s = %.1f  -/+  %.1f \n","MLP ",...
    100*round(mean(MSE_MLP),3),100*round(zn*std(MSE_MLP)/sqrt(n_run),3));

fprintf("\n\n %s = %.1f  -/+  %.1f \n","KNN ",...
    100*round(mean(MSE_KNN),3),100*round(zn*std(MSE_KNN)/sqrt(n_run),3));


