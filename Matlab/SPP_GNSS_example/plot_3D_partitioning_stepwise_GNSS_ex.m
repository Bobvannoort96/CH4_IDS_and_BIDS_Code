clear all;
clc;

%% Make it possible to also plot for arbitrary A matrix, using type_of_example
type_of_example = 'SPP_GNSS';
% type_of_example = 'ARAIM_UNDEC_GNSS';
fault_vectors = importdata(append('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\fault_vectors.txt'));

pathname = append('C:\Users\bgvannoort\Documents\IDS\Sim Data\', type_of_example, '\');
% pathname = 'H:\My Documents\PhD\Python\Case Study\6sat_example\Data\Partitions_for_matlab\';

pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
% pathname_full_grid = 'H:\My Documents\PhD\Python\Case Study\6sat_example\Data\Partitions_for_matlab\';
x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));

qmax=1;



path_to_color_dict = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\colors_partitions_dict.json';
jsontxt = fileread(path_to_color_dict);
colorDictStruct = jsondecode(jsontxt);

partitions = transpose(string(fieldnames(colorDictStruct))); % Cell array of keys

% the 'old' colors


colors_partitions = transpose(string(struct2cell(colorDictStruct))); % Cell array of values


partitions_legend = transpose(partitions);


dict_parts = dictionary(partitions, colors_partitions);

  
%% Plotting the IDS partitioning regions projected on the unit sphere, 
% imported from python generated .txt files

type_of_alpha='Kok_IDS';
separate_partitions = false; % manually change the directory for loading xx, yy, and zz!!
last_OMT = false; %% this parameter is really redundant. We cannot have Reverse IDS without 'last OMT'
inPlane = false;
alpha_0='default';
% loop over factors
% for factor=[3 4 5 6 11] %% change this line back after inPlane = False!
for factor=[4 7 10] %% change this line back after inPlane = False!
    clear h p_name
    figure('Position', [100, 100, 950, 950])
    ax=gca;
     
    
    surf(x,y,z,'EdgeColor','none','FaceColor','white')
      
    i=0;
    hold on
    % for i = 1 : k
    i = i+1;
    for part = partitions
        if inPlane
            datafile = append(pathname, 'R_IDS\', type_of_DS, '\separate_partitionings\lastOMT\alpha_type_', ...
                type_of_alpha, '\inPlane\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt');
        else
            if alpha_0 == "default"
                datafile = append(pathname, 'stepwise\no_separate_partitionings\lastOMT\alpha_type_', type_of_alpha, ...
                    '\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt');
                
            else
                datafile = append(pathname, 'R_IDS\', type_of_DS, '\no_separate_partitionings\lastOMT\alpha_type_', type_of_alpha, ...
                    '\alpha_0=', num2str(alpha_0),'\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt');
            end
        end 
        if exist(datafile, 'file')==2
            xx = importdata(datafile);
            yy = importdata(strrep(datafile, '_xx.txt', '_yy.txt'));
            zz = importdata(strrep(datafile, '_xx.txt', '_zz.txt'));

            fprintf(part)
            fprintf('\n')
            fprintf(dict_parts(part))
            fprintf('\n')
            % Start commenting because of inPlane plot
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
            elseif separate_partitions
                % these are the identifications P21, P31, P41, P42, P43 and P32. 
                % ## all partitions are the same color
                surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts(part)) %

                h(i) = surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts_alldiff(part));
                hypt = str2num(replace(part, "P", ""));
                p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
                i=i+1;
            else
                surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts(part)) %
            end
            % End commenting because of inPlane plot
                % p_name{i} = ['$\mathcal{P}_{' num2str(i) '}$'];
            
        end
        
    end
    % Plotting the ending points of the normalized cti vectors
    for i = 1:7
        if inPlane 
            cti = 11*fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        else 
            cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        end
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)

        hold on

    end   

    % plot_intersectionplane_offset(a1vec, a2vec, zeros(3,1), 0)

    % Set the limits for the x, y, and z axes
    xlim([-2 2]);    % Set x-axis limit from -2 to 2
    ylim([-2 2]);    % Set y-axis limit from -3 to 3
    zlim([-2 2]);  % Set z-axis limit from -10 to 10
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')
    

    title(append('Partitioning for stepwise type C at R=', num2str(factor)), 'FontSize', 18)
    legend(h,p_name,'Interpreter','latex', 'FontSize',18)
    
    legend boxon
    axis equal
    axis 'off'
    % ax.Position = [0 0 0.8750 0.9150];
    view(45, 30);
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    
    topDir = 'C:\Users\bgvannoort\Documents\IDS\';
    % Print the figure using the '-bestfit' option

    if alpha_0=="default"
        subdirs = fullfile(topDir, 'Figures_report', type_of_example, 'stepwise_C');
    else
        subdirs = fullfile(topDir, 'Figures_report', type_of_example, 'stepwise_C', type_of_DS, type_of_alpha, append('alpha_0=', num2str(alpha_0) ) );
    end    

    if inPlane
        subdirs = fullfile(subdirs,'inPlaneProjections');
    end

    if ~isfolder(subdirs)
        mkdir(subdirs)
    end

    if separate_partitions
        % Uncomment this again if inPlane = False
        if inPlane
            exportgraphics(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v1_rotation_', num2str(factor), 'separate', '_plane_intersect.pdf'), ...
        'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');
            view(135, 30);
            % turn of legend for this one
            legend('off')
            % Print the figure using the '-bestfit' option
            exportgraphics(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v2_rotation_', num2str(factor), 'separate', '_plane_intersect.pdf'), ...
        'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');
        else
            print(gcf, append(subdirs, '\Partitioning_stepwise_noaxis_v1_factor_', num2str(factor), 'separate'), '-dpdf', '-bestfit'); % For saving as a PDF

            view(135, 30);
            % turn of legend for this one
            legend('off')
            % Print the figure using the '-bestfit' option

            print(gcf, append(subdirs, '\Partitioning_stepwise_noaxis_v2_factor_', num2str(factor), 'separate'), '-dpdf', '-bestfit'); % For saving as a PDF
        end
    else
        % print(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v1_factor_', num2str(factor), 'not_separate'), '-dpdf', '-bestfit'); % For saving as a PDF
        exportgraphics(gcf, append(subdirs, '\Partitioning_stepwise_factor_', num2str(factor), 'not_separate.pdf'), ...
        'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');

        view(135, 30);
        % turn of legend for this one
        legend('off')
        % Print the figure using the '-bestfit' option
        % print(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v2_factor_', num2str(factor), 'not_separate'), '-dpdf', '-bestfit'); % For saving as a PDF
        exportgraphics(gcf, append(subdirs, '\Partitioning_stepwise_factor_', num2str(factor), 'not_separate.pdf'), ...
        'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');
    end
end


