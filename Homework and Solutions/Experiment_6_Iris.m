
clear all;
close all;

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

tic;
model = fitcecoc(X_train,Y_train);
t = toc;
disp(['Training Time for SVM: ',num2str(t),' sec']);
tic;
Y_predict = predict(model, X_test);
t1 = toc;
disp(['Prediction Time for SVM: ',num2str(t1),' sec']);

%y = numel(model.ClassNames);
[CM,~] =confusionmat(Y_test,Y_predict);
[Mecric_Table] = CalculateMetric(CM,y);
F1_Score=Mecric_Table{{'Average'},'F1'};
disp(['F1 Score : ',num2str(F1_Score)]);

plot(t1,F1_Score,'*','MarkerSize',9)
hold on
fprintf('\n');

tic;
model_t = fitctree(X_train,Y_train);
t2 = toc;
disp(['Training Time for Decision Tree: ',num2str(t2),' sec']);
tic;
Y_treepredict = predict(model_t,X_test);
t3 = toc;
disp(['Prediction Time for Decision Tree: ',num2str(t3),' sec']);


[CM,~] =confusionmat(Y_test,Y_treepredict);
[Mecric_Table] = CalculateMetric(CM,y);
F1_Score=Mecric_Table{{'Average'},'F1'};
disp(['F1 Score : ',num2str(F1_Score)]);
plot(t3,F1_Score,'*','MarkerSize',9)
hold on
grid minor
fprintf('\n');

tic;
model_knn = fitcknn(X_train,Y_train,'NumNeighbors',5);
t4 = toc;
disp(['Training Time for KNN: ',num2str(t4),' sec']);
tic;
Y_knnpredict = predict(model_knn,X_test);
t5 = toc;
disp(['Prediction Time for KNN: ',num2str(t5),' sec']);

[CM,~] =confusionmat(Y_test,Y_knnpredict);
[Mecric_Table] = CalculateMetric(CM,y);
F1_Score=Mecric_Table{{'Average'},'F1'};
disp(['F1 Score : ',num2str(F1_Score)]);
plot(t5,F1_Score,'*','MarkerSize',9)
hold off
ylabel('F1 Score')
xlabel('Prediction Time [s]')
ylim([0 100])
legend('SVM','Decision Tree','KNN')









