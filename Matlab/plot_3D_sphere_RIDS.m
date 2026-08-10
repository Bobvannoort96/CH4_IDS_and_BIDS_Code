clear all;
clc;
pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';


fault_vectors = importdata('C:\Users\bgvannoort\Documents\IDS\Sim Data\fault_vectors.txt');
pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';

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
separate_partitions = false; % manually change the directory for loading xx, yy, and zz!!
last_OMT = true; %% this parameter is really redundant. We cannot have Reverse IDS without 'last OMT'
inPlane = false;
alpha_0='default';
% loop over factors
for factor=[2.89 2.9] %% change this line back after inPlane = False!
    clear h p_name
    figure('Position', [100, 100, 950, 950])
    ax=gca;
    
    if inPlane
        % surf(11*x,11*y,11*z,'EdgeColor','none','FaceColor','white')
    else
        surf(x,y,z,'EdgeColor','none','FaceColor','white')
    end 
    i=0;
    hold on
    % for i = 1 : k
    i = i+1;
    for part = partitions
        if inPlane
            datafile = append(pathname, 'R_IDS\', type_of_DS, '\separate_partitionings\alpha_type_', type_of_alpha,'\inPlane\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt');
        else
            if alpha_0 == "default"
                datafile = append(pathname, 'R_IDS\', type_of_DS, '\separate_partitionings\lastOMT\alpha_type_', type_of_alpha, ...
                    '\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt');
            else
                datafile = append(pathname, 'R_IDS\', type_of_DS, '\separate_partitionings\lastOMT\alpha_type_', type_of_alpha, ...
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
                if part == 'P99'
                    p_name{i} = ['$\mathcal{P}_{\Omega}$'];
                else 
                    p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
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
    for i = 1:4
        if inPlane 
            cti = 11*fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        else 
            cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        end
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        hold on

    end   

    if inPlane % check here if the vectors a1_vec.txt and a2_vec.txt exist
        % if so, plot a plane. 
        dir_vectors = append(pathname, 'R_IDS\', type_of_DS, '\separate_partitionings\alpha_type_', type_of_alpha,'\inPlane\Partitioned_grid\factor_', num2str(factor), '\');
        if exist(append(dir_vectors, 'a1_vec.txt'), 'file')
            hold on
            a1vec = load(append(dir_vectors, 'a1_vec.txt'));
            a2vec = load(append(dir_vectors, 'a2_vec.txt'));
            a1vec = a1vec / norm(a1vec) ; 
            a2vec = a2vec / norm(a2vec);
            % plot_intersectionplane_offset(a1vec, a2vec, zeros(3,1), factor)
        end

        % also plot the default partition, i.e. for R=11 
        % for part = partitions
        %     datafile_inPlane = append(pathname, 'R_IDS\', type_of_DS, '\separate_partitionings\alpha_type_', type_of_alpha,'\Partitioned_grid\factor_11\', part, '_xx.txt');
        % 
        % 
        %     if exist(datafile_inPlane, 'file')==2
        %         xx = importdata(datafile_inPlane);
        %         yy = importdata(strrep(datafile_inPlane, '_xx.txt', '_yy.txt'));
        %         zz = importdata(strrep(datafile_inPlane, '_xx.txt', '_zz.txt'));
        % 
        %         surf(xx*11,yy*11,zz*11,'EdgeColor','none','FaceColor', dict_parts(part));
        %     end 
        % end

    end
    % Set the limits for the x, y, and z axes
    % xlim([-2 2]);    % Set x-axis limit from -2 to 2
    % ylim([-2 2]);    % Set y-axis limit from -3 to 3
    % zlim([-2 2]);  % Set z-axis limit from -10 to 10
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')
    
    if inPlane
        title(append('Partitioning for type ', type_of_DS, ' Reverse IDS on plane', 'alpha_0=', alpha_0), 'FontSize', 18)
    else 
        title(append('Partitioning for type ', type_of_DS, ' Reverse IDS', ', R = ', num2str(factor), ', \alpha_0=', num2str(alpha_0)), 'FontSize', 18)
        legend(h,p_name,'Interpreter','latex', 'box', 'on', 'FontSize',18)
    end
    
    axis equal
    axis 'off'
    % ax.Position = [0 0 0.8750 0.9150];
    view(45, 30);
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    
    topDir = 'C:\Users\bgvannoort\Documents\IDS\';
    % Print the figure using the '-bestfit' option

    if alpha_0=="default"
        subdirs = fullfile(topDir, 'Figures_report', 'R_IDS', type_of_DS, type_of_alpha);
    else
        subdirs = fullfile(topDir, 'Figures_report', 'R_IDS', type_of_DS, type_of_alpha, append('alpha_0=', num2str(alpha_0) ) );
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
            print(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v1_factor_', num2str(factor), 'separate'), '-dpdf', '-bestfit'); % For saving as a PDF

            view(135, 30);
            % turn of legend for this one
            legend('off')
            % Print the figure using the '-bestfit' option

            print(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v2_factor_', num2str(factor), 'separate'), '-dpdf', '-bestfit'); % For saving as a PDF
        end
    else
        % print(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v1_factor_', num2str(factor), 'not_separate'), '-dpdf', '-bestfit'); % For saving as a PDF
        exportgraphics(gcf, append(subdirs, '\Partitioning_R_IDS_', type_of_DS,'_noaxis_v1_factor_', num2str(factor), 'not_separate.pdf'), ...
        'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');

        view(135, 30);
        % turn of legend for this one
        legend('off')
        % Print the figure using the '-bestfit' option
        % print(gcf, append(subdirs, '\Partitioning_R_IDS_noaxis_v2_factor_', num2str(factor), 'not_separate'), '-dpdf', '-bestfit'); % For saving as a PDF
        exportgraphics(gcf, append(subdirs, '\Partitioning_R_IDS_', type_of_DS,'_noaxis_v2_factor_', num2str(factor), 'not_separate.pdf'), ...
        'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'image');
    end
end



function plot_intersectionplane_offset(v1, v2, P0, factor)
    % This function plots a plane in 3D defined by two vectors and a point.
    % Inputs:
    %   v1 - First 3D vector
    %   v2 - Second 3D vector
    %   P0 - A point on the plane (usually the same as v1 or v2)
    
    normal = cross(v1, v2);
    normal = normal /norm(normal);
    d = - dot(v1 + factor*normal, normal);
    % Create a parameter grid for the linear combinations of v1 and v2
    [X, Y] = meshgrid(-11:0.05:11, -11:0.05:11);  % Adjust grid size and range as needed

    % 
    Z = (-normal(1) * X - normal(2) * Y - d) * 1. /normal(3);
    
    % % where is Z > 1.5 or smaller than -1.5
    % bool_array1 = Z > 1.5;
    % bool_array2 = Z < -1.5;


    % Z(bool_array1) = NaN;
    % Z(bool_array2) = NaN;
    % 
    % X(bool_array1) = NaN;
    % X(bool_array2) = NaN;
    % 
    % Y(bool_array1) = NaN;
    % Y(bool_array2) = NaN;

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

