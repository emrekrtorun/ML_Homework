%% ---Visualize the data
clear all
close all

load fisheriris
X = meas(:,[1 3]);
Y = species;
classes = unique(Y);
y=numel(unique(Y));

figure()
t1 = sum((strcmp(Y,'setosa')));
t2 = sum((strcmp(Y,'versicolor')));
t3 = sum((strcmp(Y,'virginica')));
c = categorical({'Setosa','Versicolor','Virginica'});
bar(c,[t1 t2 t3])
set(gca,'ylim',([0 60]))
ylabel('Number of Elements')
xlabel('Classes')
ylim([0 60])
grid minor

figure()
gscatter(X(:,1),X(:,2),Y);
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
grid minor

step_size = 0.02; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):step_size:max(X(:,1)),min(X(:,2)):step_size:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

for j=1:3
 indx = strcmp(Y,classes(j));
 SVMModels{j} = fitcsvm(X,indx,'KernelFunction','Linear');
 [~,scores] = predict(SVMModels{j},xGrid);
figure(3)
gscatter(X(:,1),X(:,2),Y);
hold on
grid minor

contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');  % Decision boundary
hold on
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend({'setosa','versicolor','virginica','Decision Boundary'});

end
%% --- Task 2
clear all
close all

load fisheriris
X = meas(:,[1 3]);
Y = species;
classes = unique(Y);
y=numel(unique(Y));


[m,n] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);

model = fitcecoc(X_train,Y_train);
Y_predict = predict(model, X_test);

[CM,~] = confusionmat(Y_test,Y_predict);
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for SVM : ')
disp(Metric_Table)
PlotBar(Metric_Table,y)
title('Metrics for SVM')

%% --- Task 3 

clear all
close all

load fisheriris
X = meas(:,[1 3]);
Y = species;
classes=unique(Y);
y=numel(classes);

mean_ = mean(meas);
T = std(meas);

meas_yeni = zeros(size(meas));
for i=1:4
    meas_yeni(:,i) = meas(:,i)-mean_(:,i);
end

for i =1:4
    meas_yeni(:,i) = meas_yeni(:,i)/T(:,i);
end

X = meas_yeni(:,[1 3]);
Y = species;

[m,n] = size(X);
P = 0.80;
idx = transpose(randperm(m));
X_train = X(idx(1:round(P*m)),:);
Y_train = Y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:);
Y_test = Y(idx(round(P*m)+1:end),:);


SVMmodel = fitcecoc(X_train,Y_train);

Y_predict = predict(SVMmodel, X_test);

[CM,~] = confusionmat(Y_test,Y_predict)
[Metric_Table] = CalculateMetric(CM,y);
disp('Metrics for SVM : ')
disp(Metric_Table)
PlotBar(Metric_Table,y)
title('Metrics for SVM')