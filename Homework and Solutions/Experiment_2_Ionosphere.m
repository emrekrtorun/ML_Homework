%% 
clear all
close all

load ionosphere

[m,~] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

svm = fitcsvm(X_train,Y_train,'KernelFunction','Linear');
Y_predict = predict(svm,X_test);
y = numel(svm.ClassNames);
[CM,~] =confusionmat(Y_test,Y_predict);
figure()
confusionchart(Y_test,Y_predict)
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for SVM : ')
disp(Metric_Table)
PlotBar(Metric_Table,y)
title('Metrics for SVM')



%% 

clear all
close all

load ionosphere

[m,~] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

tree = fitctree(X_train,Y_train);
Y_treepredict = predict(tree,X_test);
y = numel(tree.ClassNames);
[CM,~] =confusionmat(Y_test,Y_treepredict);
figure(1)
confusionchart(Y_test,Y_treepredict)
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for Decision Tree : ')
disp(Metric_Table)
PlotBar(Metric_Table,y)
title('Metrics for Decision Tree')

%% 
clear all
close all

load ionosphere

[m,~] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

knn = fitcknn(X_train,Y_train);
Y_knnpredict = predict(knn,X_test);
y = numel(knn.ClassNames);
[CM,~] =confusionmat(Y_test,Y_knnpredict);
figure()
confusionchart(Y_test,Y_knnpredict)
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for KNN : ')
disp(Metric_Table)
PlotBar(Metric_Table,y)
title('Metrics for KNN')
