function [Metric_Table] = CalculateMetric(CM,y)


for i=1:y
   TP(i) = CM(i,i);
   FP(i) = sum(CM(:,i))-CM(i,i);
   FN(i) = sum(CM(i,:))-CM(i,i);
   TN(i) = sum(sum(CM))-TP(i)-FP(i)-FN(i);       
end

 for i=1:y
 Accuracy(i) = ((TP(i)+TN(i))/(TP(i)+FP(i)+FN(i)+TN(i)))*100;
 Recall(i)=(CM(i,i)/sum(CM(i,:)))*100;
 Precision(i) = (CM(i,i)/sum(CM(:,i)))*100;
 end
 
 F1 = (2*Precision(:).*Recall(:)./(Precision(:)+Recall(:)))';

for i = 1:y
   TPR(i) = (TP(i)/(TP(i)+FN(i)))*100;
   FPR(i) = (FP(i)/(FP(i)+TN(i)))*100;
end


Avg_Accuracy =mean(Accuracy);
Avg_Precision = mean(Precision);
Avg_Recall = mean(Recall);
Avg_F1 = mean(F1);
Avg_TPR = mean(TPR);
Avg_FPR = mean(FPR);

if y==2
T =table(Accuracy',Precision',Recall',F1',TPR',FPR','VariableNames',{'Accuracy','Precision','Recall','F1','TPR','FPR'},'RowNames',{'bad';'good'});
T1 =table(Avg_Accuracy,Avg_Precision,Avg_Recall,Avg_F1,Avg_TPR,Avg_FPR,'VariableNames',{'Accuracy','Precision','Recall','F1','TPR','FPR'},'RowNames',{'Average'});
Metric_Table = [T;T1];

elseif y == 3
T =table(Accuracy',Precision',Recall',F1',TPR',FPR','VariableNames',{'Accuracy','Precision','Recall','F1','TPR','FPR'},'RowNames',{'setosa';'versicolor';'virginica'});
T1 =table(Avg_Accuracy,Avg_Precision,Avg_Recall,Avg_F1,Avg_TPR,Avg_FPR,'VariableNames',{'Accuracy','Precision','Recall','F1','TPR','FPR'},'RowNames',{'Average'});
Metric_Table = [T;T1];

end


end

