function PlotBar(Metric_Table,y)
z=table2array(Metric_Table);

if y==2
    figure()
    h = bar(z');
    set(gca,'XtickLabel',{'Accuracy','Precision','Recall','F1','TPR','FPR'})
    legend('B','G','Average')
    grid minor
        
elseif y == 3
    figure()
    h = bar(z');
    set(gca,'XtickLabel',{'Accuracy','Precision','Recall','F1','TPR','FPR'})
    legend('Setosa','Versicolor','Virginica','Average')
    grid minor
end
end

