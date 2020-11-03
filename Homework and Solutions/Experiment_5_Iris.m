
clear all
close all

load fisheriris 


X = meas;
Y = species ;

[idx,C] = kmeans(X,3)

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
hold on
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
hold on


plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
grid minor
legend('Cluster 1','Cluster 2','Cluster 3','Cluster Center',...
       'Location','NW')
title ('Cluster Centers');




