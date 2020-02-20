function net = auto_tune_pattern_net(y, X)
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
        
        
        net = patternnet(hidd_array(hyper_choice));
        net.trainParam.showWindow = false;
        
        t = ind2vec(y(trIdx)');
        

        net = train(net, X(trIdx,:)', t);

        
        y_hat_test =  vec2ind(net(X(teIdx,:)'))';
        errcv(i) = mean(y_hat_test~=y(teIdx));
    end
    cvErr(hyper_choice) = mean(errcv);
end

[~,i] = min(cvErr);
hiddenLayerSize = hidd_array(i);


net = patternnet(hiddenLayerSize);
net.trainParam.showWindow = false;

t = ind2vec(y');


net = train(net, X', t);
