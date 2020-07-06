
% this script prints out mis-classification results on real world data-sets

clear, clc
rng(0);

%% settings
n_run = 1;      % number of experiment averages
dset = 1;       % choice of data set --> see bellow

%% real data
switch dset
    case 1
        load('data/iris.mat') % 150x4
    case 2
        load('data/wine.mat') % 178x13
    case 3
        load('data/ecoli.mat') % 337x7
    case 4
        load('data/heart_disease.mat') % 313x13
    case 5
        load('data/balance-scale.mat') % 625x4
    case 6
        load('data/ionosphere.mat')  % 351x34
end

[n,d] = size(X);

if min(y) == 0
    y = y + 1;
end

X(isnan(X)) = 0;
y(isnan(y)) = 1;

%% initialization
error_DC = zeros(n_run, 1);
error_SVM = zeros(n_run, 1);
error_MLP = zeros(n_run, 1);
error_KNN = zeros(n_run, 1);

%% learning
for run = 1:n_run
    fprintf("\n\nrun %d/%d\n", run, n_run)
    
    %% train test split
    n_train = floor(1/2*n);
    n_test = n - n_train;
    
    I_train = randsample(1:n, n_train);
    I_test = setdiff(1:n, I_train);
    
    y_train = y(I_train);
    X_train = X(I_train,:);
    y_test = y(I_test);
    X_test = X(I_test,:);
    
    %% DC
    f_hat = auto_tune_dc_fit(y_train, X_train, "classification");
    
    scores = zeros(n_test, max(y));
    for k=1:max(y)
        scores(:,k) = f_hat{k}(X_test);
    end
    
    [~, y_hat_test] = max(scores, [], 2);
    error_DC(run) = mean(y_hat_test ~= y_test);
    
    %% SVM
    try
        Mdl = fitcecoc(X_train,y_train,'Coding','onevsall');
        
        y_hat_test = predict(Mdl,X_test);
        error_SVM(run) = mean(y_hat_test~=y_test);
    catch
        warning('Statistics and ML toolbox is not installed. no SVM results!!')
    end
    
    %% KNN
    try
        Mdl = auto_tune_KNN(y_train, X_train, "classification");
        y_hat_test = predict(Mdl,X_test);
        error_KNN(run) = mean(y_hat_test~=y_test);
    catch
        warning('Statistics and ML toolbox is not installed. no KNN results!!')
    end
    
    %% MLP
    try
        net = auto_tune_pattern_net(y_train, X_train);
        y_hat_test =  vec2ind(net(X_test'))';
        error_MLP(run) = mean(y_hat_test~=y_test);
    catch
        warning('Deep learning toolbox is not installed. no MLP results!!')
    end
    
end


%% results
zn = 1.96;      % 95 percent interval
fprintf("\n\n %s = %.1f  -/+  %.1f \n","DC ",...
    100*round(mean(error_DC),3),100*round(zn*std(error_DC)/sqrt(n_run),3));

fprintf("\n\n %s = %.1f  -/+  %.1f \n","SVM ",...
    100*round(mean(error_SVM),3),100*round(zn*std(error_SVM)/sqrt(n_run),3));

fprintf("\n\n %s = %.1f  -/+  %.1f \n","MLP ",...
    100*round(mean(error_MLP),3),100*round(zn*std(error_MLP)/sqrt(n_run),3));

fprintf("\n\n %s = %.1f  -/+  %.1f \n","KNN ",...
    100*round(mean(error_KNN),3),100*round(zn*std(error_KNN)/sqrt(n_run),3));


