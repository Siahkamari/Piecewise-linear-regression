
% this script plots Normalized MSE results vs dimmension for regression on
% synthetic dataset

clear, clc
rng(0);

%% settings
n_run = 5;     % number of experiment averages
max_d = 10;     % maximum dimmension of the experiment
n_train = 30;   % number of data points
sigma = 0.25;   % standard deviation of the noise

%% initialization
MSE_DC = zeros(n_run, max_d);
MSE_MARS = zeros(n_run, max_d);
MSE_MLP = zeros(n_run, max_d);
MSE_KNN = zeros(n_run, max_d);

%% learning
for run = 1:n_run
    fprintf("\n\n run %d/%d\n", run, n_run)
    for d = 1:max_d
        fprintf("\ndimmension %d/%d\n", d, max_d)
        %% generating synthetic data
        X_train = randn(n_train, d);
        y_train = regfunction(X_train, sigma);
        
        X_test = randn(10000, d);
        y_test = regfunction(X_test, 0);
        
        %% D.C
        f_hat= auto_tune_dc_fit(y_train, X_train, "regression");
        y_hat_test = f_hat(X_test);
        MSE_DC(run, d) = mean((y_hat_test-y_test).^2)/var(y_test);
        
        %% MLP
        try
        net = auto_tune_mlp(y_train, X_train);
        y_hat_test =  net(X_test')';
        MSE_MLP(run,d) = mean((y_hat_test-y_test).^2)/var(y_test);
        catch
            warning('Deep learning toolbox is not installed. no MLP results!!')
        end
                
        %% MARS
        addpath('ARESLab')
        MARS_params = aresparams2('maxFuncs', 101);
        [model, ~, resultsEval] = aresbuild(X_train, y_train, MARS_params);
        results = arestest(model, X_test, y_test);
        MSE_MARS(run,d) = (results.MSE)/var(y_test);
        
        %% KNN
        try
            Mdl = auto_tune_KNN(y_train, X_train, "regression");
            y_hat_test = predict(Mdl,X_test);
            MSE_KNN(run,d) = mean((y_hat_test-y_test).^2)/var(y_test);
        catch 
            warning('Statistics and ML toolbox is not installed. no KNN results!!')
        end 
    end
end


%% plotting
errorbar(1:max_d,mean(MSE_DC),-1.96*std(MSE_DC)/sqrt(n_run),...
    1.96*std(MSE_DC)/sqrt(n_run),'LineWidth', 2,'MarkerSize', 10); hold on

errorbar(1:max_d,mean(MSE_MARS),-1.96*std(MSE_MARS)/sqrt(n_run),...
    1.96*std(MSE_MARS)/sqrt(n_run),'LineWidth', 2,'MarkerSize', 10);

errorbar(1:max_d,mean(MSE_MLP),-1.96*std(MSE_MLP)/sqrt(n_run),...
    1.96*std(MSE_MLP)/sqrt(n_run),'LineWidth', 2,'MarkerSize', 10);

errorbar(1:max_d,mean(MSE_KNN),-1.96*std(MSE_KNN)/sqrt(n_run),...
    1.96*std(MSE_KNN)/sqrt(n_run),'LineWidth', 2,'MarkerSize', 10);

legend("DC", "MARS", "MLP", "KNN", 'FontSize',15)

ylabel("Normalized MSE", 'FontSize',15)
xlabel("dimension", 'FontSize',15)
xticks(1:2:max_d)

%% regression function
function y = regfunction(X, sigma)
[n, d] = size(X);

y = (sum(X,2)/sqrt(d)).^2 + sin(pi*sum(X,2)/sqrt(d)) ;

y = y + sigma*randn(n,1);
end

