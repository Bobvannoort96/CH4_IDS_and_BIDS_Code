clear; clc;
pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';

fault_vectors = importdata('C:\Users\bgvannoort\Documents\IDS\Sim Data\fault_vectors.txt');
pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
Qtt = eye(3);

x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));


partitions = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P21" "P23" "P24" "P31" "P32" "P34" "P41" "P42" "P43" "P99"];

partitions_legend = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P23" "P24" "P34" "P99"];

colors_partitions = ["#F5F5F5" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FFAA00" "magenta" "#FF9F65" "green" "magenta" "#65F1FF" "#D0B623" "#FF9F65" "#65F1FF" "black"];

colors_partitions_alldiff = ["#F5F5F5" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FACE76" "magenta" "#FF9F65" "#6BFF64" "#FFAAF6" "#65F1FF" "#C19929" "#C45715" "#018A98" "black"];

dict_parts = dictionary(partitions, colors_partitions);
dict_parts_alldiff = dictionary(partitions, colors_partitions_alldiff);

% u = identifications;
%% Plotting the IDS partitioning regions projected on the unit sphere, 
% imported from python generated .txt files
type_of_DS = 'B';
type_of_alpha='Kok_IDS';
% type_of_alpha = 'manual';
separate_partitions = false; % manually change the directory for loading xx, yy, and zz!!
last_OMT = false;
inPlane = false;
alpha_0='default';
% loop over factors
plot_proj_fault_lines=false;
for factor=[2.89 2.9] %% change this line back after inPlane = False!
    clear h p_name
    figure('Position', [100, 100, 900, 900])
    ax=gca;
    surf(x,y,z,'EdgeColor','none','FaceColor','white');

    hold on
    i = 1;
    
    if last_OMT
        if separate_partitions
            firstpartDir = append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\lastOMT\');
        else
            firstpartDir = append(pathname, 'IDS\', type_of_DS, '\no_separate_partitionings\lastOMT\');
        end
    else
        if separate_partitions
            firstpartDir=append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\');
        else
            firstpartDir=append(pathname, 'IDS\', type_of_DS, '\no_separate_partitionings\');
        end
    end

    if type_of_alpha == "manual"
        datadir = append(firstpartDir, 'alpha_type_', type_of_alpha,'\alpha_0=', num2str(alpha_0), '\Partitioned_grid\factor_', num2str(factor));
    else
        datadir = append(firstpartDir, 'alpha_type_', type_of_alpha, '\Partitioned_grid\factor_', num2str(factor));
    end

    for part = partitions 
        datafile = append(datadir, '\', part, '_xx.txt');
        % datafile = append('C:\Users\bgvannoort\Documents\IDS\Sim Data\R_IDS\A\separate_partitionings\alpha_type_Kok_IDS\Partitioned_grid\factor_3\', part, '_xx.txt');
        if exist(datafile, 'file')==2
            xx = importdata(append(datadir, '\', part, '_xx.txt'));
            yy = importdata(append(datadir, '\', part, '_yy.txt'));
            zz = importdata(append(datadir, '\', part, '_zz.txt'));
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
                if hypt == 99
                    p_name{i} = ['$\mathcal{P}_{\Omega}$'];
                elseif length(hypt0) == 1
                    p_name{i} = ['$\mathcal{P}_{' hypt0 '}$'];
                else 
                    p_name{i} = ['$\mathcal{P}_{' hypt0 '}$'];
                end 
                i=i+1;
            elseif separate_partitions
                % these are the identifications P21, P31, P41, P42, P43 and P32. 
                % ## all partitions are the same color
                
                
                h(i) = surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts_alldiff(part));
                hypt = str2num(replace(part, "P", ""));
                digits_str = extractAfter(part, 1);               % '123'
                digit_chars = char(digits_str);                   % ['1','2','3']
                digit_cells = cellstr(digit_chars.');             % {'1';'2';'3'}
                hypt0 = strjoin(digit_cells, ',') ;               % '1,2,3' as char vector
                if hypt == 99
                    p_name{i} = ['$\mathcal{P}_{\Omega}$'];
                else 
                    p_name{i} = ['$\mathcal{P}_{\{' hypt0 '\}}$'];
                end 

                i=i+1;
            else
                surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts(part)) %
            end
                % p_name{i} = ['$\mathcal{P}_{' num2str(i) '}$'];
            
        end
        
    end
    % Plotting the ending points of the normalized cti vectors
    for i = 1:4
        cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',5)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',5)

        cti = cti*1.05; % Start of plotting the projected fault lines
        if (i == 1) && plot_proj_fault_lines
            pcti = cti * inv(cti' * inv(Qtt) * cti) * cti' * inv(Qtt);
            pcti_perp = eye(3) - pcti;
            for zeta = 1:4
                if zeta ~= i
                    ctjbar = pcti_perp * fault_vectors(:,zeta);
                    ctjbar = 0.4*ctjbar / vecnorm(fault_vectors(:,zeta));
                    prtsion = append('P', num2str(zeta));
                    clr = dict_parts(prtsion);
                    % print it on the 'minus' location of the cti vectors
                    % instead, so it aligns with 
                    quiver3(-cti(1), -cti(2), -cti(3), ctjbar(1), ctjbar(2), ctjbar(3), 0, "Color", clr, 'Linewidth', 2)
                    quiver3(-cti(1), -cti(2), -cti(3), -ctjbar(1), -ctjbar(2), -ctjbar(3), 0, "Color", clr, 'Linewidth', 2)
                end
            end
        end
    end   

   if inPlane % check here if the vectors a1_vec.txt and a2_vec.txt exist
        % if so, plot a plane. 

        dir_vectors = append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Rotate_Along_c1c2_vectors\Partitioned_grid\', num2str(rotation), '\');
        if exist(append(dir_vectors, 'a1_vec.txt'), 'file')
            hold on
            a1vec = load(append(dir_vectors, 'a1_vec.txt'));
            a2vec = load(append(dir_vectors, 'a2_vec.txt'));
            plot_intersectionplane(a1vec, a2vec, zeros(3,1))
        end

    end
    % Set the limits for the x, y, and z axes
    xlim([-2 2]);    % Set x-axis limit from -2 to 2
    ylim([-2 2]);    % Set y-axis limit from -3 to 3
    zlim([-2 2]);  % Set z-axis limit from -10 to 10
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')
    if inPlane
        title(append('Partitioning for type ', type_of_DS, ' IDS', ', R = ', num2str(factor), ' Rot =', num2str(rotation)))
    else
        title(append('Partitioning for type ', type_of_DS, ' IDS', ', R = ', num2str(factor)))
    end
    legend(h,p_name,'Interpreter','latex','box','on','FontSize',14, 'Location', 'northeast')
    axis equal
    axis 'off'

    view(45, 30);
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    ax.Position = [0 0 0.8750 0.9150];
    
    topDir = 'C:\Users\bgvannoort\Documents\IDS\';
    % Print the figure using the '-bestfit' option
    if type_of_alpha == "manual"
        subdirs = fullfile(topDir, 'Figures_report', 'IDS', type_of_DS, type_of_alpha, append('alpha_0=', num2str(alpha_0)));
    else
        subdirs = fullfile(topDir, 'Figures_report', 'IDS', type_of_DS, type_of_alpha);
    end

    if ~exist(subdirs, 'dir')
        mkdir(subdirs) 
    end
    if separate_partitions
        % Uncomment this again if inPlane = False
        if inPlane
            subdirs = 'C:\Users\bgvannoort\Documents\IDS\Figures_report\IDS\B\Kok_IDS\inPlaneProjections';
            print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v1_rotation_', num2str(rotation), 'separate', '_lastOMT=', mat2str(last_OMT), '_plane_intersect'), '-dpdf', '-bestfit'); % For saving as a PDF
            
            view(135, 30);
            % turn of legend for this one
            % legend('off')
            % Print the figure using the '-bestfit' option
            print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v2_rotation_', num2str(rotation), 'separate', '_lastOMT=', mat2str(last_OMT), '_plane_intersect'), '-dpdf', '-bestfit'); % For saving as a PDF
        elseif plot_proj_fault_lines
 
            exportgraphics(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v1_factor_', num2str(factor), 'separate_plot_fault_lines_true_lastOMT=', mat2str(last_OMT), '.pdf'), ...
            'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');
            
            viewdirection_ct1 = fault_vectors(:,1)/vecnorm(fault_vectors(:,1));
            

            campos(-viewdirection_ct1*5)
            % turn of legend for this one
            legend('off')
             
            exportgraphics(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v2_factor_', num2str(factor), 'separate_plot_fault_lines_true_lastOMT=', mat2str(last_OMT), '.pdf'), ...
            'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');

        
        else
            % print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v1_factor_', num2str(factor), 'separate', '_lastOMT=', mat2str(last_OMT)), '-dpdf', '-bestfit'); % For saving as a PDF
            exportgraphics(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v1_factor_', num2str(factor), 'separate', '_lastOMT=', mat2str(last_OMT), '.pdf'), ...
            'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');
            view(135, 30);
            % turn of legend for this one
            % legend('off')
            % Print the figure using the '-bestfit' option
                  
            % print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v2_factor_', num2str(factor), 'separate', '_lastOMT=', mat2str(last_OMT)), '-dpdf', '-bestfit'); % For saving as a PDF
            exportgraphics(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v2_factor_', num2str(factor), 'separate', '_lastOMT=', mat2str(last_OMT), '.pdf'), ...
            'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');
        end
    else
        print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v1_factor_', num2str(factor), 'not_separate', '_lastOMT=', mat2str(last_OMT)), '-dpdf', '-bestfit'); % For saving as a PDF

        view(135, 30);
        % turn of legend for this one
        legend('off')
        % Print the figure using the '-bestfit' option
        print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v2_factor_', num2str(factor), 'not_separate', '_lastOMT=', mat2str(last_OMT)), '-dpdf', '-bestfit'); % For saving as a PDF
    end
end



function plot_intersectionplane(v1, v2, P0)
    % This function plots a plane in 3D defined by two vectors and a point.
    % Inputs:
    %   v1 - First 3D vector
    %   v2 - Second 3D vector
    %   P0 - A point on the plane (usually the same as v1 or v2)
    
    normal = cross(v1, v2);
    d = - dot(P0, normal);
    % Create a parameter grid for the linear combinations of v1 and v2
    [X, Y] = meshgrid(-1:0.05:1, -1:0.05:1);  % Adjust grid size and range as needed

    % 
    Z = (-normal(1) * X - normal(2) * Y - d) * 1. /normal(3);
    
    % where is Z > 1.5 or smaller than -1.5
    bool_array1 = Z > 1.5;
    bool_array2 = Z < -1.5;


    Z(bool_array1) = NaN;
    Z(bool_array2) = NaN;

    X(bool_array1) = NaN;
    X(bool_array2) = NaN;
    
    Y(bool_array1) = NaN;
    Y(bool_array2) = NaN;

    % % Ensure that X, Y, Z are within the range [-1.5, 1.5]
    % X = max(min(X, 1.5), -1.5);
    % Y = max(min(Y, 1.5), -1.5);
    % Z = max(min(Z, 1.5), -1.5);

    % Find and display the min and max of X, Y, Z
    min_X = min(X(:));
    max_X = max(X(:));
    
    min_Y = min(Y(:));
    max_Y = max(Y(:));
    
    min_Z = min(Z(:));
    max_Z = max(Z(:));
    
    % Display the results
    disp(['Min of X: ', num2str(min_X)]);
    disp(['Max of X: ', num2str(max_X)]);
    
    disp(['Min of Y: ', num2str(min_Y)]);
    disp(['Max of Y: ', num2str(max_Y)]);
    
    disp(['Min of Z: ', num2str(min_Z)]);
    disp(['Max of Z: ', num2str(max_Z)]);
    


    % Plot the surface/plane using surf
    surf(X, Y, Z, 'FaceColor', '#808080', 'FaceAlpha', 0.3);
    end

