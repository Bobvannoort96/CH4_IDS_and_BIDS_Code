%% Uncomment all!! Has been used to check the example from the SS report!

% fault_vectors = importdata('C:\Users\bgvannoort\Documents\PhD\Python\Case Study\6sat example\Partitions_for_matlab\Blanch\fault_vectors.txt');
% type_of_example = 'ARAIM_UNDEC_GNSS';
type_of_example = 'SPP_GNSS';
fault_vectors = importdata(fullfile('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\fault_vectors.txt'));
fault_vectors_MHSS = importdata(fullfile('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\fault_vectors_MHSS.txt'));

A_mat = importdata(fullfile('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\A_mat.txt'));
Bt = importdata(fullfile('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\B_transpose_matrix.txt'));

pathname = fullfile('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example);
% pathname = 'H:\My Documents\PhD\Python\Case Study\6sat_example\Data\Partitions_for_matlab\';

pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
% pathname_full_grid = 'H:\My Documents\PhD\Python\Case Study\6sat_example\Data\Partitions_for_matlab\';
x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));




path_to_color_dict = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\colors_partitions_dict.json';
jsontxt = fileread(path_to_color_dict);
colorDictStruct = jsondecode(jsontxt);

partitions = transpose(string(fieldnames(colorDictStruct))); % Cell array of keys

% the 'old' colors


colors_partitions = transpose(string(struct2cell(colorDictStruct))); % Cell array of values


partitions_legend = transpose(partitions);


dict_parts = dictionary(partitions, colors_partitions);

% % For example SS report
% partitions = ["P0" "P1" "P2" "P3" "P4" "P5" "P6" "P99"];
% 
% colors_partitions = ["black" "#1F77B4" "#FF7F0E" "#2CA02C" "#D62728" "#9467BD" "#8C564B" "#808080"];
%
% partitions_legend = transpose(partitions);
% 
% dict_parts = dictionary(partitions, colors_partitions);
% END for exmaple SS report

% u = identifications;
%% Plotting the classical DIA partitioning regions projected on the unit sphere, 
% imported from python generated .txt files
ARAIM_partition_type="Zhai";
inPlane=false;
sigma_used = 10.0;
% loop over factors
for factor=[4]

    % figure('Position', [100, 100, 800, 800])
    figure('Position', [100, 100, 950, 950])
    ax=gca;
    % ax.Position = ax.Position + [-0.1 -0.1 0.2 0.2];
    % t = tiledlayout(1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    % ax = nexttile;
    surf(x,y,z,'EdgeColor','none','FaceColor','white');
    hold on
    i = 1;
    for part = partitions
        % dir_path_string = append(pathname, 'DS_DIA\', type_of_DS, '\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\');
        if ARAIM_partition_type == "Blanch"
            % dir_path_string = 'C:\Users\bgvannoort\Documents\PhD\Python\Case Study\6sat example\Partitions_for_matlab\Blanch\';
            dir_path_string = fullfile('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\ARAIM\Blanch\separate_partitionings\alpha_type_Kok_IDS\Partitioned_grid\');
            title_string =['Blanch et al. V-ARAIM, R=' num2str(factor)];
        else 
            % dir_path_string = 'C:\Users\bgvannoort\Documents\PhD\Python\Case Study\6sat example\Partitions_for_matlab\Zhai\';
            dir_path_string = fullfile('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\ARAIM\Zhai\separate_partitionings\alpha_type_Kok_IDS\Partitioned_grid\');
            title_string = ['Zhai et al. V-ARAIM, R=' num2str(factor)];
            if sigma_used == 10.0
                dir_path_string = fullfile('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\ARAIM\Zhai\separate_partitionings\alpha_type_Kok_IDS\sigma=10.0\Partitioned_grid\');
            end
        end
        datafile = append(dir_path_string, 'factor_', num2str(factor), '\', part, '_xx.txt');
         
        if exist(datafile, 'file')==2
            xx = importdata(append(dir_path_string, 'factor_', num2str(factor), '\', part, '_xx.txt'));
            yy = importdata(append(dir_path_string, 'factor_', num2str(factor), '\', part, '_yy.txt'));
            zz = importdata(append(dir_path_string, 'factor_', num2str(factor), '\', part, '_zz.txt'));
        
            fprintf(part)
            fprintf('\n')
            fprintf(dict_parts(part))
            fprintf('\n')
            if any(partitions_legend == part)
                h(i)=surf(xx,yy,zz,'EdgeColor','none','FaceColor', dict_parts(part));
                hypt = str2num(replace(part, "P", ""));
                digits_str = extractAfter(part, 1);               % '123'
                digit_chars = char(digits_str);                   % ['1','2','3']
                digit_cells = cellstr(digit_chars.');             % {'1';'2';'3'}
                hypt0 = strjoin(digit_cells, ',') ;               % '1,2,3' as char vector
                if part == 'P99'
                    p_name{i} = ['$\mathcal{P}_{\Omega}$'];
                else 
                    p_name{i} = ['$\mathcal{P}_{' hypt0 '}$'];
                % else 
                
                %     p_name{i} = ['$\mathcal{P}_{' int2str(alt) '}$'];
                end 
                i=i+1;
              
            else
                % these are the identifications P21, P31, P41, P42, P43 and P32. 
                % ## all partitions are the same color
                surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts(part)) %
                
                h(i) = surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts_alldiff(part));
                hypt = str2num(replace(part, "P", ""));
                if hypt==99
                    p_name{i} = ['$\mathcal{P}_{\Omega}$'];
                else
                    p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
                end 
                i=i+1;
            end
        end
        
        
        
    end
      
    
    % % Plot fault vectors for the DIA method, i.e. simply cti
    % for i = 1:7
    %     if inPlane 
    %         cti = 11*fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
    %     else 
    %         cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
    %     end
    %     plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',4)
    %     plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',4)
    %     % plot3(cti(1),cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
    %     % plot3(-cti(1),-cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
    %     hold on
    % 
    % end  
    % plot the MHSS fault vectors Aplus @ ci @ ctiplus 
    i=1;
    for part =partitions
        if (part == 'P0' || part == 'P99')
            continue 
        end
        cti_MHSS = fault_vectors_MHSS(:,i)*1.0;
        plot3(cti_MHSS(1),cti_MHSS(2),cti_MHSS(3),'Color',dict_parts(part),'Marker','+','LineWidth',3,'MarkerSize',4, 'MarkerEdgeColor','k')
        plot3(-cti_MHSS(1),-cti_MHSS(2),-cti_MHSS(3),'Color',dict_parts(part),'Marker','+','LineWidth',3,'MarkerSize',4, 'MarkerEdgeColor','k')
        i=i+1;
    end
    xlabel('t_1')
    ylabel('t_2')
    zlabel('t_3')

    % title(append('Data snooping type ', type_of_DS, 'partitioning R=', num2str(factor)))
    title(title_string)
    legend(h,p_name,'Interpreter','latex','FontSize',14, 'Location', 'east')
    axis equal
    axis 'off'
    % ax.Position = [0 0 0.8750 0.9150];
    view(0,90)
    % ax = gca;
    % outerPos = ax.OuterPosition;
    % tightInset = ax.TightInset;
    % ax.Position = [outerPos(1) + tightInset(1), ...
    %                outerPos(2) + tightInset(2), ...
    %                outerPos(3) - tightInset(1) - tightInset(3), ...
    %                outerPos(4) - tightInset(2) - tightInset(4)];
    % view(45, 30);
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    topDir = 'C:\Users\bgvannoort\Documents\IDS\';
    subdirs = fullfile(topDir, 'Figures_report', type_of_example, 'ARAIM', ARAIM_partition_type);
    mkdir(subdirs)

    % print(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v1_factor_', num2str(factor), 'not_separate'), '-dpdf', '-bestfit'); % For saving as a PDF
    % exportgraphics(gcf, append(subdirs, '\Partitioning_ARAIM_', ARAIM_partition_type,'_noaxis_v1_factor_', num2str(factor), '.pdf'), ...
    % 'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');

    view(135, 30);
    % turn of legend for this one
    % legend('off')
    % Print the figure using the '-bestfit' option
    % print(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v2_factor_', num2str(factor), 'not_separate'), '-dpdf', '-bestfit'); % For saving as a PDF
    % exportgraphics(gcf, append(subdirs, '\Partitioning_ARAIM_',ARAIM_partition_type,'_noaxis_v2_factor_', num2str(factor), 'not_separate.pdf'), ...
    % 'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');
end

%
