close all
clear all
clc
y = [3.9    4.9     8.1     4.0     %iris
     6.6     8.3     11.9    14.2   %bal

         12.5    18.6    19.3    14.1    %ecol
         
    21.3    4.8     5.2     31.3    %wine
    46.5    42.5    42.1    52.9    %heart

    11.1    14.3    13.4    14];    %ion

std_dev = [0.7  1.2     1.5     0.8   %iris
            0.7  1.3     0.9     1.6    %bal
            
            1.4  1.0     1.6     1.8     %ecol
           2.5  1.0     1.0     1.9     %wine
           
           1.4  1.5     1.7     2.4     %heart
           
           1.3  1.8     0.8     1.3];    %ion

num = 6; %number of different subcategories
c = 1:num;
col = get(gca,'colororder');

%% dists
dis = 0.03;
wid = 0.15;

%%Figure
figH = figure;
axes1 = axes;
% title('Title');
% xlabel('data-set');
ylabel('Mis-classification %');
hold on
%%Bar(s)
%You can not color differently the same bar.
for i = 1:num
    for j = 1:4
        bar(c(i) +(j-3.5)*(dis+wid) ,y(i,j),wid, 'FaceColor',col(j,:));
    end
end
%%Errorbar
errH = cell(4,1);
for j=1:4
    errH{j} = errorbar(c+(j-3.5)*(dis+wid),y(:,j),std_dev(:,j),'.','Color',0.8*col(j,:));
    errH{j}.LineWidth = 1.5;
end

%%Set x-ticks
set(axes1,'Xlim',[0.3 6.3]);
set(axes1,'XTick',[1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7],'XTickLabel',...
    {'Iris',' ','Balance Scale',' ','Ecoli',' ','Wine',' ','Heart Disease',' ','Ionosphere',' '});

legend({'DC';'MLP';"SVM";"KNN"},'Location','northeast')
% ylim([0,300])