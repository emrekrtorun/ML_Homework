clear all
close all

load fisheriris 

X = meas();
Y = species;
y=numel(unique(Y));

Avg_Precision = zeros(1,6);
Avg_Recall = zeros(1,6);

svm = fitcecoc(X,Y);

k =5;
while k<11
cval_svm = crossval(svm,'KFold',k);
Y_predict = kfoldPredict(cval_svm);
[CM,~]=confusionmat(Y,Y_predict);
[Metric_Table] = CalculateMetric(CM,y);
Avg_Precision(k-4)=Metric_Table{{'Average'},'Precision'};
Avg_Recall(k-4)= Metric_Table{{'Average'},'Recall'};
k = k+1;
end


subplot(1,2,1)
plot([5:10],Avg_Precision,'r-o')
xlabel('K Values');ylabel('Precision')
grid minor
subplot(1,2,2)
plot([5:10],Avg_Recall,'m-o')
xlabel('K Values');ylabel('Recall')
grid minor
sgtitle('Metrics for SVM')

figure()
Graph = [Avg_Precision;Avg_Recall];
bar(Graph');
set(gca,'YLim',[80 100]);
set(gca,'XTickLabel',[1:6]+4);
legend('Precision','Recall');
xlabel('K Values')
grid minor;



%% 

clear all
close all

load fisheriris

X = meas();
Y = species;
y=numel(unique(Y));

Avg_Precision = zeros(1,6);
Avg_Recall = zeros(1,6);

tree = fitctree(X,Y);
k = 5;
while k<11
cval_tree = crossval(tree,'KFold',k);
Y_predict = kfoldPredict(cval_tree);
[CM,~]=confusionmat(Y,Y_predict);
[Metric_Table] = CalculateMetric(CM,y);
Avg_Precision(k-4)=Metric_Table{{'Average'},'Precision'};
Avg_Recall(k-4)= Metric_Table{{'Average'},'Recall'};
k=k+1;
end

subplot(1,2,1)
plot([5:10],Avg_Precision,'r-o')
xlabel('K Values');ylabel('Precision')
grid minor
subplot(1,2,2)
plot([5:10],Avg_Recall,'m-o')
xlabel('K Values');ylabel('Recall')
grid minor
sgtitle('Metrics for Decision Tree')


figure();
Graph = [Avg_Precision;Avg_Recall];
bar(Graph');
set(gca,'YLim',[80 100]);
set(gca,'XTickLabel',[1:6]+4);
legend('Precision','Recall');
xlabel('K Values')
grid minor;

%% 

clear all
close all

load fisheriris 

X = meas() ;
Y = species;
y=numel(unique(Y));

Avg_Precision = zeros(1,6);
Avg_Recall = zeros(1,6);

knn = fitcknn(X,Y);
k = 5;

while k<11
cval_knn = crossval(knn,'KFold',k);
Y_predict = kfoldPredict(cval_knn);
[CM,~]=confusionmat(Y,Y_predict);
[Metric_Table] = CalculateMetric(CM,y);
Avg_Precision(k-4)=Metric_Table{{'Average'},'Precision'};
Avg_Recall(k-4)= Metric_Table{{'Average'},'Recall'};
k=k+1;
end

subplot(1,2,1)
plot([5:10],Avg_Precision,'r-o')
xlabel('K Values');ylabel('Precision')
grid minor
subplot(1,2,2)
plot([5:10],Avg_Recall,'m-o')
xlabel('K Values');ylabel('Recall')
grid minor
sgtitle('Metrics for KNN'),

figure();
Graph = [Avg_Precision;Avg_Recall];
bar(Graph');
set(gca,'YLim',[80 100]);
set(gca,'XTickLabel',[1:6]+4);
legend('Precision','Recall');
xlabel('K Values')
grid minor;

