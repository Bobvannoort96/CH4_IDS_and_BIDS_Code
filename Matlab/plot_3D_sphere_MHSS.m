
pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';

%% Building a grid of points on the surface of a zero-centred unit sphere (3D)
t1 = 0:0.01:2*pi;
t2 = -pi/2:0.01:pi/2;
[tet,si] = meshgrid(t1,t2);
method = 'Blanch';
% pathname = append('C:\Users\bgvannoort\Documents\PhD\Python\Case Study\6sat example\Partitions_for_matlab', '\', method);
pathname = append('C:\Users\bgvannoort\Documents\IDS\Sim Data\MHSS\', method, '\');
fault_vectors = importdata(append('C:\Users\bgvannoort\Documents\IDS\Sim Data\', 'fault_vectors.txt'));

pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));


partitions = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P21" "P23" "P24" "P31" "P32" "P34" "P41" "P42" "P43" "P99"];

partitions_legend = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P23" "P24" "P34" "P99"];

colors_partitions = ["#F5F5F5" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FFAA00" "magenta" "#FF9F65" "green" "magenta" "#65F1FF" "#D0B623" "#FF9F65" "#65F1FF" "black"];

colors_partitions_alldiff = ["#F5F5F5" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FACE76" "magenta" "#FF9F65" "#6BFF64" "#FFAAF6" "#65F1FF" "#C19929" "#C45715" "#018A98" "black"];

%SPP example
% partitions = ["P0" "P1" "P2" "P3" "P4" "P5" "P6" "P99"];
% 
% colors_partitions = ["black" "#1F77B4" "#FF7F0E" "#2CA02C" "#D62728" "#9467BD" "#8C564B" "#808080"];
dict_parts = dictionary(partitions, colors_partitions);
% dict_parts_alldiff = dictionary(partitions, colors_partitions_alldiff);

% u = identifications;
%% Plotting the IDS partitioning regions projected on the unit sphere, 
% imported from python generated .txt files

type_of_alpha='iteration';
separate_partitions = false; % manually change the directory for loading xx, yy, and zz!!
last_OMT = false;
inPlane = false;
alpha_prime = 0.038;
alpha_per_hypt = 0.004;
% loop over factors
for factor=[3 4 ] %% change this line back after inPlane = False!
    % clear h p_name
    figure('Position', [100, 100, 1000, 1000])
    ax=gca;
    ax.FontSize=24;
    % h(i) = surf(x,y,z,'EdgeColor','none','FaceColor','black');
    % p_name{i} = '$\mathcal{P}_{\Omega}$';
    i=1;
    surf(x,y,z,'EdgeColor','none','FaceColor','white');
    % h(i) = surf(x,y,z,'EdgeColor','none','FaceColor','#F5F5F5');
    % p_name{i} = '$\mathcal{P}_0$';
    % i=i+1;
    hold on
    % for i = 1 : k
    
    dir_path_string = append(pathname, '\no_separate_partitionings_alpha_prime_0.038_alpha_per_hypt_0.004\Partitioned_grid\factor_', num2str(factor), '\');
    
    % Initialize handles and legend names
    h = []; % Initialize empty array for surface handles
    p_name = {}; % Initialize empty cell array for legend names
    
    % Loop over partitions to plot surfaces
    for part = partitions
        datafile = append(dir_path_string, part, '_xx.txt');
        if exist(datafile, 'file') == 2
            xx = importdata(append(dir_path_string, part, '_xx.txt'));
            yy = importdata(append(dir_path_string, part, '_yy.txt'));
            zz = importdata(append(dir_path_string, part, '_zz.txt'));
    
            fprintf(part);
            fprintf('\n');
            fprintf(dict_parts(part));
            fprintf('\n');
    
            
            h(end+1) = surf(xx, yy, zz, 'EdgeColor', 'none', 'FaceColor', dict_parts(part)); % Append handle
            hypt = str2num(replace(part, "P", ""));
            if part == 'P99'
                p_name{end+1} = ['$\mathcal{P}_{\Omega}$'];
            else
                p_name{end+1} = ['$\mathcal{P}_{' int2str(hypt) '}$']; % Append legend entry
            end
            
            
        end
    end

    % Validate legend handles and names
    if ~isempty(h) && ~isempty(p_name)
        legend(h, p_name, 'Interpreter', 'latex', 'box', 'off', 'FontSize', 14);
    else
        warning('No valid surfaces found for legend.');
    end
    % Plotting the ending points of the normalized cti vectors
    for i = 1:4
        cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)

    end  

    % Set the limits for the x, y, and z axes
    xlim([-2 2]);    % Set x-axis limit from -2 to 2
    ylim([-2 2]);    % Set y-axis limit from -3 to 3
    zlim([-2 2]);  % Set z-axis limit from -10 to 10
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')
    legend(h,p_name,'Interpreter','latex', 'box', 'on', 'FontSize',18)
    title(append("Partitioning for ", method, " MHSS, R=", num2str(factor) ), 'FontSize', 18)
    axis equal
    axis off
    ax.Position = [0 0 0.8750 0.9150];

    view(45, 30);
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    subdirs = append('C:\Users\bgvannoort\Documents\IDS\Figures_report\MHSS\', method, '\alpha_prime_', num2str(alpha_prime),  'alpha_per_hypt_', num2str(alpha_per_hypt));
    
    % Check if the directory exists
    if ~isfolder(subdirs)
        % Create the directory if it doesn't exist
        mkdir(subdirs);
    end

    % print(gcf, append(subdirs, '\Partitioning_', method, '_MHSS_noaxis_v1_factor_', num2str(factor), 'not_separate'), '-dpdf', '-bestfit'); % For saving as a PDF
    exportgraphics(gcf, fullfile(subdirs, ['Partitioning_', method, '_MHSS_noaxis_v1_factor_', num2str(factor), '_not_separate.pdf']), 'BackgroundColor', 'white');

    view(135, 30);
    % turn of legend for this one
    legend('off')
    % Print the figure using the '-bestfit' option
    % print(gcf, append(subdirs, '\Partitioning_', method, '_MHSS_noaxis_v2_factor_', num2str(factor), 'not_separate'), '-dpdf', '-bestfit'); % For saving as a PDF
    exportgraphics(gcf, fullfile(subdirs, ['Partitioning_', method, '_MHSS_noaxis_v2_factor_', num2str(factor), '_not_separate.pdf']), 'BackgroundColor', 'white');



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

