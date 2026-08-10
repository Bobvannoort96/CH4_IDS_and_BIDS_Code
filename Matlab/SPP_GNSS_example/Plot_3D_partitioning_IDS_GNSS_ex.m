clc;
clear;
type_of_example = 'SPP_GNSS';
% type_of_example = 'ARAIM_UNDEC_GNSS';

pathname_full_grid= 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
pathname = fullfile(pathname_full_grid, type_of_example, '\');

fault_vectors = importdata(fullfile(pathname, 'fault_vectors.txt'));
Bt = importdata(fullfile(pathname, 'B_transpose_matrix.txt'));

separate_partitions = false;
if separate_partitions
    path_to_color_dict = fullfile(pathname, 'separate_colors_partitions_dict.json');
else
    path_to_color_dict = fullfile(pathname, 'colors_partitions_dict.json');
end

Qyy_diag = importdata(fullfile(pathname, 'Qyy_diag.txt'));
Qyy = diag(Qyy_diag);
Qtt = Bt * Qyy * transpose(Bt);
Qtt = eye(3);

x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));


jsontxt = fileread(path_to_color_dict);
colorDictStruct = jsondecode(jsontxt);

partitions = transpose(string(fieldnames(colorDictStruct))); % Cell array of keys

% the 'old' colors


colors_partitions = transpose(string(struct2cell(colorDictStruct))); % Cell array of values


partitions_legend = transpose(partitions);


dict_parts = dictionary(partitions, colors_partitions);
% dict_parts_alldiff = dictionary(partitions, colors_partitions_alldiff);

% u = identifications;
%% Plotting the IDS partitioning regions projected on the unit sphere, 
% imported from python generated .txt files
type_of_DS = 'C';
type_of_alpha='Kok_IDS';
type_of_testing = 'IDS';

plot_proj_fault_lines=false;
last_OMT = false;
inPlane = false;


if inPlane
    newGrid = fullfile(pathname, type_of_testing, type_of_DS, 'separate_partitionings', 'inPlane', ...
                   ['alpha_type_' type_of_alpha], 'Rotate_Along_c1c2_vectors', '\');
    x = importdata(append(newGrid, 'grid_x.txt'));
    y = importdata(append(newGrid, 'grid_y.txt'));
    z = importdata(append(newGrid, 'grid_z.txt'));
end

% loop over factors
% for factor=[3 4 5 6 7 8 9 10] %% change this line back after inPlane = False!
for factor=[4 7 11 ]
    clear h p_name
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
        if strlength(part) == 3
            if ~separate_partitions && str2num(extractBetween(part, 2,2)) > str2num(extractBetween(part, 3,3))
                continue 
            end
        else
            ...
        end
        if inPlane
            startDir = append(pathname, type_of_testing, '\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Rotate_Along_c1c2_vectors\Partitioned_grid\', num2str(factor));
        elseif last_OMT
            startDir = append(pathname,  type_of_testing, '\', type_of_DS, '\no_separate_partitionings\lastOMT\alpha_type_', type_of_alpha,'\Partitioned_grid\factor_', num2str(factor));
        else
            startDir = append(pathname,  type_of_testing, '\', type_of_DS, '\no_separate_partitionings\alpha_type_', type_of_alpha,'\Partitioned_grid\factor_', num2str(factor));
        end
        datafile = append(startDir, '\', part, '_xx.txt');
        if exist(datafile, 'file')==2
            xx = importdata(append(startDir, '\', part, '_xx.txt'));
            yy = importdata(append(startDir, '\', part, '_yy.txt'));
            zz = importdata(append(startDir, '\', part, '_zz.txt'));
        
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
                end
                i=i+1;
            elseif separate_partitions
                % these are the identifications P21, P31, P41, P42, P43 and P32. 
                % ## all partitions are the same color
                surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts(part)) %
                
                h(i) = surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts_alldiff(part));
                hypt = str2num(replace(part, "P", ""));
                p_name{i} = ['$\mathcal{P}_{' hypt0 '}$'];
                i=i+1;
            else
                surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts(part)) %
            end
                % p_name{i} = ['$\mathcal{P}_{' num2str(i) '}$'];
        end
        
    end
    % Plotting the ending points of the normalized cti vectors
    for i = 1:7
        if (i == 1 || i == 3) && inPlane
            % Plot a line segment from -cti to +cti
            cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
            % plot3(10*[-cti(1), cti(1)], 10*[-cti(2), cti(2)], 10*[-cti(3), cti(3)], ...
            %       'Color', 'black', 'LineWidth', 1, 'LineStyle', '--');
        elseif ~inPlane
            cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
            plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
            plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
            cti = cti*1.05;

        end
        % continue
        if (i == 1) && plot_proj_fault_lines
            pcti = cti * inv(cti' * inv(Qtt) * cti) * cti' * inv(Qtt);
            pcti_perp = eye(3) - pcti;
            for zeta = 1:7
                if zeta ~= i
                    ctjbar = pcti_perp * fault_vectors(:,zeta);
                    % ctjbar = ctjbar / vecnorm(fault_vectors(:,zeta));
                    % ctjbar = ctjbar / vecnorm(ctjbar)*0.5;
                    prtsion = append('P', num2str(zeta));
                    clr = dict_parts(prtsion);
                    quiver3(cti(1), cti(2), cti(3), ctjbar(1), ctjbar(2), ctjbar(3), 0, "Color", clr, 'Linewidth', 2)
                    quiver3(cti(1), cti(2), cti(3), -ctjbar(1), -ctjbar(2), -ctjbar(3), 0, "Color", clr, 'Linewidth', 2)
                end
            end
        end
    
    end   

    if inPlane % check here if the vectors a1_vec.txt and a2_vec.txt exist
        % if so, plot a plane. 

        % dir_vectors = append(pathname, 'IDS\', type_of_DS, '\no_separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Rotate_Along_c1c2_vectors\Partitioned_grid\', num2str(rotation), '\');
        % if exist(append(dir_vectors, 'a1_vec.txt'), 'file')
        %     hold on
        %     a1vec = load(append(dir_vectors, 'a1_vec.txt'));
        %     a2vec = load(append(dir_vectors, 'a2_vec.txt'));
        %     plot_intersectionplane(a1vec, a2vec, zeros(3,1))
        % end

    end
    % Set the limits for the x, y, and z axes
    xlim([-2 2]);    % Set x-axis limit from -2 to 2
    ylim([-2 2]);    % Set y-axis limit from -3 to 3
    zlim([-2 2]);  % Set z-axis limit from -10 to 10
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')
    if inPlane
        % title(append('Partitioning for type ', type_of_DS, ' IDS', ', R = ', num2str(factor), ' Rot =', num2str(rotation)))
        title(['Partitioning for type ', type_of_DS, ' ', type_of_testing, ' R = ', num2str(factor), ...
       ' in $\mathcal{R}([c_{t_1}, c_{t_3}])$'], 'Interpreter', 'latex')
    else
        title(append('Partitioning for type ', type_of_testing, ' ', type_of_DS, ', R = ', num2str(factor),' ', type_of_example))
    end

    legend(h,p_name,'box','on','FontSize',14, 'Interpreter','latex')
    axis equal
    axis 'off'
    if plot_proj_fault_lines
        ctvec = fault_vectors(:,1) / vecnorm(fault_vectors(:,1));
        campos(3*ctvec)
    elseif inPlane
        ct1 = fault_vectors(:, 1) / vecnorm(fault_vectors(:,1));
        ct3 = fault_vectors(:,3) / vecnorm(fault_vectors(:,3));
        normVec = cross(ct1, ct3);
        campos(3*normVec)
    else
        view(45,30);
    end
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    
    topDir = 'C:\Users\bgvannoort\Documents\IDS\';
    % Print the figure using the '-bestfit' option
    subdirs = fullfile(topDir, 'Figures_report', type_of_example,type_of_testing, type_of_DS, type_of_alpha);
    
    if ~isfolder(subdirs)
        mkdir(subdirs)
    end
    
    if separate_partitions
        % Uncomment this again if inPlane = False
        if inPlane
            subdirs = ['C:\Users\bgvannoort\Documents\IDS\Figures_report\', type_of_example, '\', type_of_testing, type_of_DS, '\inPlaneProjections\'];
            print(gcf, append(subdirs, '\Partitioning_', type_of_testing, '_noaxis_v1_rotation_', num2str(rotation), 'separate', '_lastOMT=', mat2str(last_OMT), '_plane_intersect.pdf'), '-dpdf', '-bestfit'); % For saving as a PDF
            
            view(135, 30);
            % turn of legend for this one
            legend('off')
            % Print the figure using the '-bestfit' option
            print(gcf, append(subdirs, '\Partitioning_', type_of_testing, '_noaxis_v2_rotation_', num2str(rotation), 'no_separate', '_lastOMT=', mat2str(last_OMT), '_plane_intersect.pdf'), '-dpdf', '-bestfit'); % For saving as a PDF
        else
            print(gcf, append(subdirs, '\Partitioning_', type_of_testing, '_noaxis_v1_factor_', num2str(factor), 'no_separate', '_lastOMT=', mat2str(last_OMT), '.pdf'), '-dpdf', '-bestfit'); % For saving as a PDF
            
            view(135, 30);
            % turn of legend for this one
            legend('off')
            % Print the figure using the '-bestfit' option
                  
            print(gcf, append(subdirs, '\Partitioning_', type_of_testing, '_noaxis_v2_factor_', num2str(factor), 'no_separate', '_lastOMT=', mat2str(last_OMT), '.pdf'), '-dpdf', '-bestfit'); % For saving as a PDF
        end
    else
        if ~plot_proj_fault_lines

            print(gcf, append(subdirs, '\Partitioning_', type_of_testing, '_noaxis_v1_factor_', num2str(factor), 'not_separate', '_lastOMT=', mat2str(last_OMT), '.pdf'), '-dpdf', '-bestfit'); % For saving as a PDF
    
            view(135, 30);
            % turn of legend for this one
            legend('off')
            % Print the figure using the '-bestfit' option
            print(gcf, append(subdirs, '\Partitioning_', type_of_testing, '_noaxis_v2_factor_', num2str(factor), 'not_separate', '_lastOMT=', mat2str(last_OMT), '.pdf'), '-dpdf', '-bestfit'); % For saving as a PDF
        else
            print(gcf, append(subdirs, '\Partitioning_', type_of_testing, '_noaxis_v1_factor_', num2str(factor), 'not_separate_with_proj_faultlines', '_lastOMT=', mat2str(last_OMT), '.pdf'), '-dpdf', '-bestfit'); % For saving as a PDF
    
            view(135, 30);
            % turn of legend for this one
            legend('off')
            % Print the figure using the '-bestfit' option
            print(gcf, append(subdirs, '\Partitioning_', type_of_testing, '_noaxis_v2_factor_', num2str(factor), 'not_separate_with_proj_faultlines', '_lastOMT=', mat2str(last_OMT), '.pdf'), '-dpdf', '-bestfit'); % For saving as a PDF
        end
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

