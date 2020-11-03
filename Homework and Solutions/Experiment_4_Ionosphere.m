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
[Y_predict,scores] = predict(svm,X_test);
y = numel(svm.ClassNames);
[CM,~] =confusionmat(Y_test,Y_predict);

figure()
for i=1:2
   [x,t] = perfcurve(Y_test,scores(:,i),svm.ClassNames{i});  
    plot(x,t) 
    hold on 
end
  grid minor  
  xlabel('False positive rate') 
  ylabel('True positive rate')
  title('ROC Curve for SVM')
  legend('b','g')
title('ROC Curve for SVM')


figure()
for i=1:2
   [x,t] = perfcurve(Y_test,scores(:,i),svm.ClassNames{i},'XCrit','tpr','YCrit','ppv');
    plot(x,t)
    hold on
end
  grid minor  
  xlabel('Recall') 
  ylabel('Precision')
  title('ROC Curve for SVM')
  legend('b','g')
title('ROC Curve for SVM')






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
[Y_treepredict,scores] = predict(tree,X_test);
y = numel(tree.ClassNames);
[CM,~] =confusionmat(Y_test,Y_treepredict);

figure()
for i=1:2
   [x,t] = perfcurve(Y_test,scores(:,i),tree.ClassNames{i});
    plot(x,t)
    hold on
    xlabel('False positive rate') 
    ylabel('True positive rate')
    legend('b','g')
end
grid minor
title('ROC Curve for Decision Tree')

figure()
for i=1:2
   [x,t] = perfcurve(Y_test,scores(:,i),tree.ClassNames{i},'XCrit','tpr','YCrit','ppv');
    plot(x,t)
    hold on
    xlabel('Recall') 
    ylabel('Precision')
    legend('b','g')
end
grid minor
title('ROC Curve for Decision Tree')

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
[Y_knnpredict,scores] = predict(knn,X_test);
y = numel(knn.ClassNames);
[CM,~] =confusionmat(Y_test,Y_knnpredict);

figure()
for i=1:2
   [x,t] = perfcurve(Y_test,scores(:,i),knn.ClassNames{i});
    plot(x,t)
    hold on
    xlabel('False positive rate') 
    ylabel('True positive rate')
    legend('b','g')
end
grid minor
title('ROC Curve for KNN')


figure()
for i=1:2
   [x,t] = perfcurve(Y_test,scores(:,i),knn.ClassNames{i},'XCrit','tpr','YCrit','ppv');
    plot(x,t)
    hold on
    xlabel('Recall') 
    ylabel('Precision')
    legend('b','g')
end
grid minor
title('ROC Curve for KNN')