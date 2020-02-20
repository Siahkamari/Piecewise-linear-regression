close all
clear all
clc
y = [4,    90.6   115.1  115     
    15.9,  60,      93    50   
    33,    95.8,    114   197     
    34.1,    29.1,    30     52.7   
    89.5,   33,       185  124.6    
    42.4 ,     26    16.8   93   ];

std_dev = [ 2.2/sqrt(2)      58      12.7       57/sqrt(2)    
            6.6/sqrt(2)      17.7       14.3     16/sqrt(2)       
            21.4/sqrt(2)      36.8       16.2   59/sqrt(2)      
            5.2/sqrt(2)         5.7   3.9         8.2/sqrt(2)     
            21.9/sqrt(2)       20    94.7      25/sqrt(2)     
            3.7                2.8    2         12        ];

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
ylabel('Normalized MSE');
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
    {'Acetylene',' ','Moore',' ','Reaction',' ','Car Small',' ','Cereal',' ','Boston Housing',' '});

legend({'DC';'MLP';"MARS";"KNN"},'Location','northeast')
ylim([0,300])