clear all
close all

load fisheriris

X = meas(: ,[1 2]);
Y = species;

[m,n] = size(X);
P = 0.8;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);


t = templateSVM('KernelFunction','Linear');
model = fitcecoc(X_train,Y_train,'Learners',t);
[Y_predict,score] = predict(model, X_test);


figure(1)
for i=1:3
    [x,y] = perfcurve(Y_test,score(:,i),model.ClassNames{i});
    plot(x,y)
    hold on
    grid minor
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC Curve for SVM')

end
legend('Setosa','Versicolor','Virginica')

clear x y 

figure(2)
for i=1:3
    
    [x,y] = perfcurve(Y_test,score(:,i),model.ClassNames{i},'XCrit','tpr','YCrit','ppv');
    plot(x,y)
    hold on
    grid minor
    xlabel('Recall') 
    ylabel('Precision')
    title('ROC Curve for SVM')

end
legend('Setosa','Versicolor','Virginica')

%% 

clear all
close all

load fisheriris

X = meas(:,[1 2]);
Y = species;

[m,n] = size(X);
P = 0.80;
idx = randperm(m);
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

model_t = fitctree(X_train,Y_train);
[Y_tpredict,score] = predict(model_t,X_test);

diffscore1 = score(:,1) - max(score(:,2),score(:,3));
diffscore2 = score(:,2) - max(score(:,1),score(:,3));
diffscore3 = score(:,3) - max(score(:,1),score(:,2));

diffscore = [diffscore1,diffscore2,diffscore3];

figure(1)
for i=1:3
    [x,y] = perfcurve(Y_test,diffscore(:,i),model_t.ClassNames{i});
    plot(x,y)
    hold on
    grid minor
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC Curve for Decision Tree')

end
legend('Setosa','Versicolor','Virginica')

clear x y 

figure(2)
for i=1:3
    
    [x,y] = perfcurve(Y_test,score(:,i),model_t.ClassNames{i},'XCrit','tpr','YCrit','ppv');
    plot(x,y)
    hold on
    grid minor
    xlabel('Recall') 
    ylabel('Precision')
    title('ROC Curve for Decision Tree')

end
legend('Setosa','Versicolor','Virginica')

%% 
clear all
close all

load fisheriris

X = meas(:,[1 2]);
Y = species;

[m,n] = size(X);
P = 0.80;
idx = randperm(m);
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

model_knn = fitcknn(X_train,Y_train,'NumNeighbors',5);
[Y_kpredict,score] = predict(model_knn,X_test)


diffscore1 = score(:,1) - max(score(:,2),score(:,3));
diffscore2 = score(:,2) - max(score(:,1),score(:,3));
diffscore3 = score(:,3) - max(score(:,1),score(:,2));

diffscore = [diffscore1,diffscore2,diffscore3];

figure(1)
for i=1:3
    [x,y] = perfcurve(Y_test,diffscore(:,i),model_knn.ClassNames{i});
    plot(x,y)
    hold on
    grid minor
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC Curve for KNN')

end
legend('Setosa','Versicolor','Virginica')

clear x y 

figure(2)
for i=1:3
    
    [x,y] = perfcurve(Y_test,score(:,i),model_knn.ClassNames{i},'XCrit','tpr','YCrit','ppv');
    plot(x,y)
    hold on
    grid minor
    xlabel('Recall') 
    ylabel('Precision')
    title('ROC Curve for KNN')

end
legend('Setosa','Versicolor','Virginica')

