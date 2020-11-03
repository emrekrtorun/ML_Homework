
clear all
close all

load ionosphere

X = [X(:,3),X(:,4)];
[idx,C] = kmeans(X,2)

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
legend('Cluster 1','Cluster 2','Cluster Centers',...
       'Location','NW')
title ('Cluster Centers');










