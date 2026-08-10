
fault_vectors = importdata('C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\fault_vectors.txt');

pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\';

pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));

type_of_DS = 'A';


path_to_color_dict = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\colors_partitions_dict.json';
jsontxt = fileread(path_to_color_dict);
colorDictStruct = jsondecode(jsontxt);

partitions = transpose(string(fieldnames(colorDictStruct))); % Cell array of keys

% the 'old' colors


colors_partitions = transpose(string(struct2cell(colorDictStruct))); % Cell array of values


partitions_legend = transpose(partitions);


dict_parts = dictionary(partitions, colors_partitions);


% u = identifications;
%% Plotting the classical DIA partitioning regions projected on the unit sphere, 
% imported from python generated .txt files

% loop over factors
for factor=[6]

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
        dir_path_string = append(pathname, 'DS_DIA\', type_of_DS, '\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\');
        datafile = append(dir_path_string, part, '_xx.txt');
        if exist(datafile, 'file')==2
            xx = importdata(append(pathname, 'DS_DIA\', type_of_DS, '\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt'));
            yy = importdata(append(pathname, 'DS_DIA\', type_of_DS, '\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\', part, '_yy.txt'));
            zz = importdata(append(pathname, 'DS_DIA\', type_of_DS, '\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\', part, '_zz.txt'));
        
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
                if hypt == '99'
                    p_name{i} = ['$\mathcal{P}_{\Omega}$'];
                else
                    p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
                end
                i=i+1;
            end
                % p_name{i} = ['$\mathcal{P}_{' num2str(i) '}$'];
        end
    end
    % Plotting the ending points of the normalized cti vectors
    for i = 1:7
        cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
    end   
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')

    title(append('Data snooping type ', type_of_DS, ' partitioning R=', num2str(factor)))
    
    legend(h,p_name,'Interpreter','latex','box','on','FontSize',16, 'Location', 'best')
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
    
    topDir  = 'C:\Users\bgvannoort\Documents\IDS\';
    outDir  = append(topDir, 'Figures_report\SPP_GNSS\DS\', type_of_DS, '\');
    if ~exist(outDir, 'dir'); mkdir(outDir); end   % creates all missing parent folders too
    
    print(gcf, append(outDir, 'Partitioning_DS_', type_of_DS, '_noaxis_v1_factor_', num2str(factor)), '-dpdf', '-bestfit'); % For saving as a PDF
    
    view(135, 30);
    % turn of legend for this one
    legend('off')
    
    % Print the figure using the '-bestfit' option
    print(gcf, append(topDir, 'Figures_report\SPP_GNSS\DS\', type_of_DS, '\Partitioning_DS_', type_of_DS, '_noaxis_v2_factor_', num2str(factor)), '-dpdf', '-bestfit'); % For saving as a PDF
end

%
