function loss = cross_validate(y, X, tCL, n_folds, task)


[n, ~] = size(X);
if (n ~= length(y))
    disp('ERROR: num rows of X must equal length of y');
    return;
end

% Permute the rows of X and y
rp = randperm(n);
y = y(rp);
X = X(rp, :);

% Initializing different measure
loss = zeros(1,n_folds);

for i=1:n_folds
    
    %% splitting the data to test and train
    test_start = ceil(n/n_folds * (i-1)) + 1;
    test_end = ceil(n/n_folds * i);
    
    y_train = [];
    X_train = [];
    if i > 1
        y_train = y(1:test_start-1);
        X_train = X(1:test_start-1,:);
    end
    if i < n_folds
        y_train = [y_train; y(test_end+1:length(y))];
        X_train = [X_train; X(test_end+1:length(y), :)];
    end
    
    X_test = X(test_start:test_end, :);
    y_test = y(test_start:test_end);
    
    %% learning with the x_train and predicting with it
    f_hat = feval(tCL, y_train, X_train);
    
    n_test = size(y_test, 1);
    if task == "classification"
        scores = zeros(n_test, max(y_train));
        for k=1:max(y_train)
            scores(:,k) = f_hat{k}(X_test);
        end
    
        [~, y_hat_test] = max(scores, [], 2);
        
        loss(i) = mean(y_hat_test ~= y_test);
    elseif task == "regression"
        y_hat_test = f_hat(X_test);
        loss(i) = mean(abs(y_hat_test-y_test));
    end
    
end
loss = mean(loss);