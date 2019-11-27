clear,  clc
% rng(0);

%% Settings
task = "classification";
% task = "regression";

%% Data
n = 300;            % number of data points
dim = 2;            % dimmension of data
n_train = floor(0.7*n);  
n_test = n - n_train;
sigma = 0.2;

X = rand(n, dim);
if  task == "regression"
    y = regfunction(X, sigma);
elseif task == "classification"
    y = sign(regfunction(X, sigma));
end

I_train = randsample(1:n, n_train);
I_test = setdiff(1:n, I_train);

y_train = y(I_train);
X_train = X(I_train,:);
y_test = y(I_test);
X_test = X(I_test,:);

%% D.C.
f_hat = auto_tune_dc_fit(y_train, X_train);

y_hat_train = f_hat(X_train);
y_hat_test = f_hat(X_test);

if task == "regression"
    DC_train_loss = mean((y_hat_train-y_train).^2);
    DC_test_loss = mean((y_hat_test-y_test).^2);
elseif task == "classification"
    DC_train_loss = mean(sign(y_hat_train)~=y_train);
    DC_test_loss = mean(sign(y_hat_test)~=y_test);
end

%% SVM
if task == "regression"
    Mdl = fitrsvm(X_train,y_train);
elseif task == "classification"
    Mdl = fitcsvm(X_train,y_train);
end

y_hat_train = predict(Mdl,X_train);
y_hat_test = predict(Mdl,X_test);

if task == "regression"
    SVM_train_loss = mean((y_hat_train-y_train).^2);
    SVM_test_loss = mean((y_hat_test-y_test).^2);
elseif task == "classification"
    SVM_train_loss = mean(sign(y_hat_train)~=y_train);
    SVM_test_loss = mean(sign(y_hat_test)~=y_test);
end

%% Kernel SVM
if task == "regression"
    Mdl = fitrkernel(X_train,y_train);
elseif task == "classification"
    Mdl = fitckernel(X_train,y_train);
end

y_hat_train = predict(Mdl,X_train);
y_hat_test = predict(Mdl,X_test);

if task == "regression"
    KSVM_train_loss = mean((y_hat_train-y_train).^2);
    KSVM_test_loss = mean((y_hat_test-y_test).^2);
elseif task == "classification"
    KSVM_train_loss = mean(sign(y_hat_train)~=y_train);
    KSVM_test_loss = mean(sign(y_hat_test)~=y_test);
end

%% KNN
Mdl = fitcknn(X_train,y_train);

y_hat_train = predict(Mdl,X_train);
y_hat_test = predict(Mdl,X_test);

if task == "regression"
    KNN_train_loss = mean((y_hat_train-y_train).^2);
    KNN_test_loss = mean((y_hat_test-y_test).^2);
elseif task == "classification"
    KNN_train_loss = mean(sign(y_hat_train)~=y_train);
    KNN_test_loss = mean(sign(y_hat_test)~=y_test);
end

%% MLP
if task == "regression"
    net = fitnet(10);
    net = train(net, X_train', y_train');
    
    y_hat_train = net(X_train')';
    y_hat_test =  net(X_test')';
    
    MLP_train_loss = mean((y_hat_train-y_train).^2);
    MLP_test_loss = mean((y_hat_test-y_test).^2);
    
elseif task == "classification"
    net = patternnet(10);
    t = double([y_train<0,y_train>0]);
    net = train(net, X_train', t');
    
    y_hat_train = 2*vec2ind(net(X_train'))'-3;
    y_hat_test =  2*vec2ind(net(X_test'))'-3;
    
    MLP_train_loss = mean(sign(y_hat_train)~=y_train);
    MLP_test_loss = mean(sign(y_hat_test)~=y_test);
end

%% Linear regression
coeff = lsqminnorm(X_train, y_train);

y_hat_train = X_train*coeff;
y_hat_test = X_test*coeff;

LR_train_loss = mean((y_hat_train-y_train).^2);
LR_test_loss = mean((y_hat_test-y_test).^2);

%% GP
Mdl = fitrgp(X_train,y_train);

y_hat_train = predict(Mdl,X_train);
y_hat_test = predict(Mdl,X_test);

GP_train_loss = mean((y_hat_train-y_train).^2);
GP_test_loss = mean((y_hat_test-y_test).^2);

%% Plotting
if task == "regression"
    c = categorical({'DC','SVM','KernelSVM','KNN','MLP','GP','Linear Regression'});
    losses = [DC_train_loss, DC_test_loss; SVM_train_loss, SVM_test_loss;...
        KSVM_train_loss, KSVM_test_loss; KNN_train_loss, KNN_test_loss;...
        MLP_train_loss, MLP_test_loss ; GP_train_loss, GP_test_loss;...
        LR_train_loss, LR_test_loss];
    bar(c,losses)
    ylabel('L2 error', "FontSize", 15)
    legend("train","test", "FontSize", 15)
elseif task == "classification"
    c = categorical({'DC','SVM','KernelSVM','KNN','MLP'});
    losses = [DC_train_loss, DC_test_loss; SVM_train_loss, SVM_test_loss;...
        KSVM_train_loss, KSVM_test_loss; KNN_train_loss, KNN_test_loss ;...
        MLP_train_loss, MLP_test_loss];
    bar(c,losses)
    ylabel('0-1 loss', "FontSize", 15)
    legend("train","test", "FontSize", 15)
end
regression_plot_dc(X_train,y_train, @(X) regfunction(X,0), f_hat);
