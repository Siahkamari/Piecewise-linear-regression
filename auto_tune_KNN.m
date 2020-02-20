function Mdl = auto_tune_KNN(y, X, task)
n_folds = 5;

hidd_array = 1:10;
n_hyper_choices = length(hidd_array);
cvErr = zeros(n_hyper_choices,1);

for hyper_choice = 1:n_hyper_choices
    CVO = cvpartition(length(y),'k',n_folds);
    errcv = zeros(CVO.NumTestSets,1);
    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        
        Mdl = fitcknn(X(trIdx,:),y(trIdx),'NumNeighbors',hidd_array(i));
        y_hat_test = predict(Mdl,X(teIdx,:));
        
        if task == "regression"
            errcv(i)  = mean((y_hat_test-y(teIdx)).^2);
        elseif task == "classification"
            errcv(i)  = mean(y_hat_test~=y(teIdx));
        end
    end
    cvErr(hyper_choice) = mean(errcv);
end

[~,i] = min(cvErr);
K = hidd_array(i);

Mdl = fitcknn(X,y,'NumNeighbors',K);
