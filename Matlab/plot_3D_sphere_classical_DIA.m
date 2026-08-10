clear; clc;
fault_vectors = importdata('C:\Users\bgvannoort\Documents\IDS\Sim Data\fault_vectors.txt');
pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';

pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));



partitions = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P21" "P23" "P24" "P31" "P32" "P34" "P41" "P42" "P43"];

partitions_legend = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P23" "P24" "P34"];

colors_partitions = ["black" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FFAA00" "magenta" "#FF9F65" "green" "magenta" "#65F1FF" "#D0B623" "#FF9F65" "#65F1FF"];

colors_partitions_alldiff = ["black" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FACE76" "magenta" "#FF9F65" "#6BFF64" "#FFAAF6" "#65F1FF" "#C19929" "#C45715" "#018A98"];

dict_parts = dictionary(partitions, colors_partitions);
dict_parts_alldiff = dictionary(partitions, colors_partitions_alldiff);

% u = identifications;
%% Plotting the classical DIA partitioning regions projected on the unit sphere, 
% imported from python generated .txt files

% loop over factors
% for factor=[2.8  3  3.2 3.4 4 5 6 7 8 9 10 15 21]
for factor = [6]
    figure('Position', [100, 100, 1000, 1000])
    ax=gca;
    % ax.Position = ax.Position + [-0.1 -0.1 0.2 0.2];
    % t = tiledlayout(1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    % ax = nexttile;
    surf(x,y,z,'EdgeColor','none','FaceColor','white');
    hold on
    % for i = 1 : k
    i = 1;
    for part = partitions
        dir_path_string = append(pathname, 'ordinary_DIA\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\');
        datafile = append(dir_path_string, part, '_xx.txt');
        if exist(datafile, 'file')==2
            xx = importdata(append(pathname, 'ordinary_DIA\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt'));
            yy = importdata(append(pathname, 'ordinary_DIA\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\', part, '_yy.txt'));
            zz = importdata(append(pathname, 'ordinary_DIA\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\', part, '_zz.txt'));
        
            fprintf(part)
            fprintf('\n')
            fprintf(dict_parts(part))
            fprintf('\n')
            if any(partitions_legend == part)
                h(i)=surf(xx,yy,zz,'EdgeColor','none','FaceColor', dict_parts(part));
                hypt = str2num(replace(part, "P", ""));
                p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
                i=i+1;
            else
                % these are the identifications P21, P31, P41, P42, P43 and P32. 
                % ## all partitions are the same color
                surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts(part)) %
                
                h(i) = surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts_alldiff(part));
                hypt = str2num(replace(part, "P", ""));
                p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
                i=i+1;
            end
                % p_name{i} = ['$\mathcal{P}_{' num2str(i) '}$'];
        end
    end
    % Plotting the ending points of the normalized cti vectors
    for i = 1:4
        cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
    end   
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')

    title(append('Classical DIA partitioning R=', num2str(factor)))
    
    legend(h,p_name,'Interpreter','latex','box','on','FontSize',14, 'Location', 'best')
    axis equal
    axis 'off'
    ax.Position = [0 0 0.8750 0.9150];
    % ax = gca;
    % outerPos = ax.OuterPosition;
    % tightInset = ax.TightInset;
    % ax.Position = [outerPos(1) + tightInset(1), ...
    %                outerPos(2) + tightInset(2), ...
    %                outerPos(3) - tightInset(1) - tightInset(3), ...
    %                outerPos(4) - tightInset(2) - tightInset(4)];
    view(45, 30);
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    
    topDir = 'C:\Users\bgvannoort\Documents\IDS\';

    % Save the first figure as an image-based PDF with specified resolution
    exportgraphics(gcf, append(topDir, 'Figures_report\classical_DIA\Partitioning_classical_DIA_noaxis_v1_factor_', num2str(factor), '.pdf'), ...
        'ContentType', 'image', 'Resolution', 500);
    
    view(135, 30);
    
    % Turn off the legend for this one
    legend('off');
    
    % Save the second figure as an image-based PDF with specified resolution
    exportgraphics(gcf, append(topDir, 'Figures_report\classical_DIA\Partitioning_classical_DIA_noaxis_v2_factor_', num2str(factor), '.pdf'), ...
        'ContentType', 'image', 'Resolution', 500);
end

%
