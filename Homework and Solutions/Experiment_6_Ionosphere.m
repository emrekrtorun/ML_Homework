   
clear all;close all;

load ionosphere

[m,~] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

y = numel(unique(Y));


tic;
svm = fitcsvm(X_train,Y_train,'KernelFunction','Linear');
t = toc;
disp(['Training Time for SVM: ',num2str(t),' sec']);
tic;
[Y_predict,scores] = predict(svm,X_test);
t1 = toc;
disp(['Prediction Time for SVM: ',num2str(t1),' sec']);
[CM,~] =confusionmat(Y_test,Y_predict);
[Metric_Table] = CalculateMetric(CM,y);
F1_Score=Metric_Table{{'Average'},'F1'};
disp(['F1 Score : ',num2str(F1_Score)]);
plot(t1,F1_Score,'*','MarkerSize',9)
hold on
fprintf('\n');

tic;
tree = fitctree(X_train,Y_train);
t2=toc;
disp(['Training Time for Decision Tree: ',num2str(t2),' sec']);
tic;
[Y_treepredict,scores] = predict(tree,X_test);
t3=toc;
disp(['Prediction Time for Decision Tree: ',num2str(t3),' sec']);
[CM,~] =confusionmat(Y_test,Y_treepredict);
[Metric_Table] = CalculateMetric(CM,y);
F1_Score=Metric_Table{{'Average'},'F1'};
disp(['F1 Score : ',num2str(F1_Score)]);
plot(t3,F1_Score,'*','MarkerSize',9)
hold on
fprintf('\n');


tic;
knn = fitcknn(X_train,Y_train);
t4=toc;
disp(['Training Time for KNN: ',num2str(t4),' sec']);
tic;
[Y_knnpredict,scores] = predict(knn,X_test);
t5=toc;
disp(['Prediction Time for KNN: ',num2str(t5),' sec']);

[CM,~] =confusionmat(Y_test,Y_knnpredict);
[Metric_Table] = CalculateMetric(CM,y);
F1_Score=Metric_Table{{'Average'},'F1'};
disp(['F1 Score : ',num2str(F1_Score)]);
plot(t5,F1_Score,'*','MarkerSize',9)
hold on
grid minor
ylabel('F1 Score')
xlabel('Prediction Time [s]')
ylim([0 100])
legend('SVM','Decision Tree','KNN')




