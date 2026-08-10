% This program plots the projection of the misclosure space for IDS, ordinary DIA or DS_DIA
% on the plane spanned by the fault lines of P1 and P2. 

pathname = 'H:\My Documents\PhD\Python\Case Study\IDS\Sim Data\cross-section P1P2\';
partitions = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P21" "P23" "P24" "P31" "P32" "P34" "P41" "P42" "P43"];

partitions_legend = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P23" "P24" "P34"];

colors_partitions = ["black" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FFAA00" "magenta" "#FF9F65" "green" "magenta" "#65F1FF" "#D0B623" "#FF9F65" "#65F1FF"];

colors_partitions_alldiff = ["black" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FACE76" "magenta" "#FF9F65" "#6BFF64" "#FFAAF6" "#65F1FF" "#D0B623" "#C45715" "#018A98"];

dict_parts = dictionary(partitions, colors_partitions);
dict_parts_alldiff = dictionary(partitions, colors_partitions_alldiff);

%% Partitions to consider:
partitions = ["P0" "P1" "P2" "P12" "P21"];

%% Full grid 
pathname_grid = 'H:\My Documents\PhD\Python\Case Study\IDS\Sim Data\';
x = importdata(append(pathname_grid, 'grid_x.txt'));
y = importdata(append(pathname_grid, 'grid_y.txt'));
z = importdata(append(pathname_grid, 'grid_z.txt'));

%% Plotting the IDS partitioning regions projected on the unit sphere, 
% imported from python generated .txt files
figure
surf(x,y,z,'EdgeColor','none','FaceColor','white', FaceAlpha=0.001);
hold on
% for i = 1 : k
i = 1;
for part = partitions
    xx = importdata(append(pathname, 'Partitioned_grid\', part, '_xx.txt'));
    yy = importdata(append(pathname, 'Partitioned_grid\', part, '_yy.txt'));
    zz = importdata(append(pathname, 'Partitioned_grid\', part, '_zz.txt'));



    h(i)=surf(xx,yy,zz,'EdgeColor','none','FaceColor', dict_parts_alldiff(part));
    hypt = str2num(replace(part, "P", ""));
    p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
    i=i+1;
    
end

title('IDS partitioning')
legend(h,p_name,'Interpreter','latex','box','off','FontSize',14)
axis equal
% axis off