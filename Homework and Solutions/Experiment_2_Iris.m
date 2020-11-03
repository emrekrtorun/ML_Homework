clear all
close all 

load fisheriris

X = meas();
Y = species;
y = numel(unique(Y));

[m,n] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);


model = fitcecoc(X_train,Y_train);
Y_predict = predict(model, X_test);

model_t = fitctree(X_train,Y_train);
Y_treepredict = predict(model_t,X_test);

model_knn = fitcknn(X_train,Y_train,'NumNeighbors',3);
Y_knnpredict = predict(model_knn,X_test);

figure()
confusionchart(Y_test,Y_predict);
title('Confussion Matrix for SVM')
[CM,~] = confusionmat(Y_test,Y_predict);
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for SVM : ')
disp(Metric_Table)
figure()
PlotBar(Metric_Table,y)
title('Metrics for SVM')


figure()
confusionchart(Y_test,Y_treepredict);
title('Confussion Matrix for Decision Tree')
[CM,~] = confusionmat(Y_test,Y_treepredict);
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for Decision Tree : ')
disp(Metric_Table)
figure()
PlotBar(Metric_Table,y)
title('Metrics for Decision Tree')


figure()
confusionchart(Y_test,Y_knnpredict);
title('Confussion Matrix for KNN')
[CM,~] = confusionmat(Y_test,Y_knnpredict);
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for KNN : ')
disp(Metric_Table)
figure()
PlotBar(Metric_Table,y)
title('Metrics for KNN')









