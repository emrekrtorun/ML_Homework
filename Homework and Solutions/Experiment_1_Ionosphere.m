%% -- Visualize the data
clear all;
close all;

load ionosphere

inds = ~strcmp(Y,'b');
C = categorical(inds,[0 1],{'bad','good'});
h=histogram(C,'BarWidth',0.5);
ylabel('Count')
xlabel('Classes')
title('Histogram of Class(Good/Bad)')

figure()
gscatter(X(:,3),X(:,4),Y)
ylabel('4th feature of X')
xlabel('3rd feature of X')
grid minor


%% -- Task 2
clear all;
close all;

load ionosphere

X(:,2)=[];
[m,~] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

svm_model = fitcsvm(X_train,Y_train,'KernelFunction','Linear');
Y_predict = predict(svm_model,X_test)

[CM,~] =confusionmat(Y_test,Y_predict)
y = numel(svm_model.ClassNames);
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for SVM : ')
disp(Metric_Table)
PlotBar(Metric_Table,y)
title('Metrics for SVM')

%% -- Task 3
clear all;
close all;

load ionosphere

M = mean(X);
T = std(X);

X_yeni = zeros(size(X));
for i=1:4
    X_yeni(:,i) = X(:,i)-M(i);
end

for i =1:4
    X_yeni(:,i) = X_yeni(:,i)/T(i);
end

X = X_yeni;
X(:,2)=[];

[m,~] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

svm_model = fitcsvm(X_train,Y_train,'KernelFunction','Linear');
Y_predict = predict(svm_model,X_test)

[CM,~] =confusionmat(Y_test,Y_predict)
y = numel(svm_model.ClassNames);
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for SVM : ')
disp(Metric_Table)
PlotBar(Metric_Table,y)
title('Metrics for SVM')



