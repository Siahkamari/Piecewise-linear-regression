function net = auto_tune_mlp(y, X)
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
        
        net = fitnet(hidd_array(hyper_choice));
        net.divideParam.trainRatio = 80/100;
        net.divideParam.valRatio = 20/100;
        net.trainParam.showWindow = false;
        

        [~ , tr] = train(net,X(trIdx,:)',y(trIdx)');
        errcv(i) = tr.best_vperf;
    end
    cvErr(hyper_choice) = mean(errcv);
end

[~,i] = min(cvErr);
hiddenLayerSize = hidd_array(i);

net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;

net.trainParam.showWindow = false;

net = train(net,X',y');
