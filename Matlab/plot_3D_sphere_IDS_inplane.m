% plot the 3D misclosure space partitioning but then for an intersection
% i.e. in the plane of intersection with two fault lines for example.

clear all;
clc;
m=4;
sigma = 1.0;
A = ones(4,1);
Qy= sigma^2 * eye(m);
bool_low_res = false;

% SphereVisualisationIDS(A, Qyy)

%% Description
% This function visualizes the misclosure space partitioning, for datasnooping when r=3, in
% 3D space. The partitioning regions are projected on the surface of a unit
% sphere. 

%Author: Safoora Zaminpardaz

%% Inputs
% A   : m x n design matrix
% Qy : m x m observation VCV matrix 


%% Computing B-matrix and misclosure VCV matrix
m = size(A,1);
k=10; % nr of hypotheses in IDS = 2
B = null(A');
Qt = B'*Qy*B;
%Transforming t to a misclsure with a VCV matrix equal to the identity matrix
[V,D] = eig(Qt); 
B = B*V*(diag(diag(D^-0.5)))*V';

pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';

x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));
%% Building a grid of points on the surface of a zero-centred unit sphere (3D)
t1 = 0:0.01:2*pi;
t2 = -pi/2:0.01:pi/2;
[tet,si] = meshgrid(t1,t2);
fault_vectors = importdata('C:\Users\bgvannoort\Documents\IDS\Sim Data\fault_vectors.txt');
if ~bool_low_res
    pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
else
    pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\lower_res\';
end


partitions = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P21" "P23" "P24" "P31" "P32" "P34" "P41" "P42" "P43"];

partitions_legend = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P23" "P24" "P34"];

colors_partitions = ["black" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FFAA00" "magenta" "#FF9F65" "green" "magenta" "#65F1FF" "#D0B623" "#FF9F65" "#65F1FF"];

colors_partitions_alldiff = ["black" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FACE76" "magenta" "#FF9F65" "#6BFF64" "#FFAAF6" "#65F1FF" "#C19929" "#C45715" "#018A98"];

dict_parts = dictionary(partitions, colors_partitions);
dict_parts_alldiff = dictionary(partitions, colors_partitions_alldiff);

% u = identifications;
%% Plotting the IDS partitioning regions projected on the unit sphere, 
% imported from python generated .txt files
type_of_DS = 'B';
type_of_alpha='Kok_IDS';
separate_partitions = true; % manually change the directory for loading xx, yy, and zz!!
last_OMT = false;
inPlane = true;

% the vectors of the plane
dir_vectors = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\IDS\B\separate_partitionings\inPlane\alpha_type_Kok_IDS\Partitioned_grid';
a1vec = load(append(dir_vectors, '\a1_vec.txt') );
a2vec = load(append(dir_vectors, '\a2_vec.txt'));

topDir = 'C:\Users\bgvannoort\Documents\IDS\';
% Print the figure using the '-bestfit' option
subdirs = fullfile(topDir, 'Figures_report', 'IDS', type_of_DS, type_of_alpha, 'inPlaneProjections');
% Define the base directory name
baseDir = 'run';

% Initialize the counter
counter = 1;

% Create the first directory name
dirName = [subdirs, '\', baseDir, ' ', num2str(counter)];

% Check if the directory already exists and increment until a new name is found
while exist(dirName, 'dir')  % 'dir' checks for the existence of directories
    counter = counter + 1;
    dirName = [subdirs, '\', baseDir, ' ', num2str(counter)];
end
subdirs = dirName;
mkdir(subdirs)



% loop over factors
for factor=[0 20 40 60 90]
    clear h p_name
    figure
    i=1;
    % if factor < 4 && type_of_DS ~= 'C'
    %     h(i) = surf(x,y,z,'EdgeColor','none','FaceColor','#F5F5F5');
    %     p_name{i} = '$\mathcal{P}_{0}$';
    % elseif factor >= 4 && type_of_DS ~= 'C'
    %     if type_of_DS == 'B' && ~last_OMT % there is no undecided region anyway
    %         surf(x,y,z,'EdgeColor','none','FaceColor','white');
    %         i=0;
    %     elseif last_OMT
    %         h(i) = surf(x,y,z,'EdgeColor','none','FaceColor','black');
    %         p_name{i} = '$\mathcal{P}_{\Omega}$';
    %     else
    %         h(i) = surf(x,y,z,'EdgeColor','none','FaceColor','black');
    %         p_name{i} = '$\mathcal{P}_{\Omega}$';
    %     end
    % elseif factor < 6 && type_of_DS == 'C' % means we have type_of_DS='C'
    %     h(i) = surf(x,y,z,'EdgeColor','none','FaceColor','#F5F5F5');
    %     p_name{i} = '$\mathcal{P}_{0}$';
    % else % merely plot the sphere, as we have type C DS but no undecided region. 
    %     surf(x,y,z,'EdgeColor','none','FaceColor','black')
    %     i=0;
    % 
    % end
    % hold on
    % for i = 1 : k
    for part = partitions
        datafile = append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Rotate_Along_c1c2_vectors\Partitioned_grid\', num2str(factor), '\', part, '_xx.txt');
        if exist(datafile, 'file')==2
            % xx = importdata(append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt'));
            % yy = importdata(append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Partitioned_grid\factor_', num2str(factor), '\', part, '_yy.txt'));
            % zz = importdata(append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Partitioned_grid\factor_', num2str(factor), '\', part, '_zz.txt'));
            % 
            xx = importdata(append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Rotate_Along_c1c2_vectors\Partitioned_grid\', num2str(factor), '\', part, '_xx.txt'));
            yy = importdata(append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Rotate_Along_c1c2_vectors\Partitioned_grid\', num2str(factor), '\', part, '_yy.txt'));
            zz = importdata(append(pathname, 'IDS\', type_of_DS, '\separate_partitionings\inPlane\alpha_type_', type_of_alpha,'\Rotate_Along_c1c2_vectors\Partitioned_grid\', num2str(factor), '\', part, '_zz.txt'));
        
            fprintf(part)
            fprintf('\n')
            fprintf(dict_parts(part))
            fprintf('\n')
            if any(partitions_legend == part) 
                h(i)=surf(xx,yy,zz,'EdgeColor','none','FaceColor', dict_parts(part));
                hypt = str2num(replace(part, "P", ""));
                p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
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
                % p_name{i} = ['$\mathcal{P}_{' num2str(i) '}$'];
            hold on
        end
        
    end
    % Plotting the ending points of the normalized cti vectors
    for i = 1:4
        cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        % plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        % plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        % plot3(cti(1)*20,cti(2)*20,cti(3)*20,'Color','black')
        % hold on
        % plot3(-cti(1)*20,-cti(2)*20,-cti(3)*20,'Color','black')
        % hold on
    end   
    xlim([-10 10]);    % Set x-axis limit from -2 to 2
    ylim([-10 10]);    % Set y-axis limit from -3 to 3
    zlim([-10 10]);  % Set z-axis limit from -10 to 10
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')
    title(append('Partitioning for type ', type_of_DS, ' IDS', ' Rot = ', num2str(factor)))
    legend(h,p_name,'Interpreter','latex','box','off','FontSize',14)
    axis equal
    axis 'off'
    
    % the vectors of the plane
    dir_vectors = append('C:\Users\bgvannoort\Documents\IDS\Sim Data\IDS\B\separate_partitionings\inPlane\alpha_type_Kok_IDS\Rotate_Along_c1c2_vectors\Partitioned_grid\', num2str(factor), '\');
    a1vec = load(append(dir_vectors, 'a1_vec.txt') );
    a2vec = load(append(dir_vectors, 'a2_vec.txt'));


    
    % Set the view in MATLAB using the calculated azimuth and elevation
    view(45, 30);   

    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    
    if separate_partitions
        print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v1_rotation_', num2str(factor), 'separate_inplane', '_lastOMT=', mat2str(last_OMT), '_rotation_c1c2vectors'), '-dpdf', '-bestfit'); % For saving as a PDF
    
        view(135,30)
        print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v2_rotation_', num2str(factor), 'separate_inplane', '_lastOMT=', mat2str(last_OMT), '_rotation_c1c2vectors'), '-dpdf', '-bestfit'); % For saving as a PDF
        
        % Define the 3D viewing direction vector
        viewdir = cross(a1vec,a2vec);

        % Normalize the vector to ensure it's a unit vector
        viewdir = viewdir / norm(viewdir);

        % Calculate azimuth (in degrees) using atan2 to account for quadrants
        if abs(viewdir(1)) < 0.01
            azimuth=0.0;
        else
            azimuth = atan2d(viewdir(2), viewdir(1)); % azimuth = atan2(y, x)
        end
        % Calculate elevation (in degrees) using arcsin for the vertical component
        elevation = asind(viewdir(3)); % elevation = arcsin(z)
        view(azimuth, elevation)
        view(2)
        print(gcf, append(subdirs, '\Partitioning_IDS_v3_rotation_', num2str(factor), 'separate_inplane_view_onto_plane', '_lastOMT=', mat2str(last_OMT), '_rotation_c1c2vectors'), '-dpdf', '-bestfit'); % For saving as a PDF
        
    else
        print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v1_rotation_', num2str(factor), 'not_separate_inplane', '_lastOMT=', mat2str(last_OMT),  '_rotation_c1c2vectors'), '-dpdf', '-bestfit'); % For saving as a PDF
        
        view(135,30)
        print(gcf, append(subdirs, '\Partitioning_IDS_noaxis_v2_rotation_', num2str(factor), 'separate_inplane', '_lastOMT=', mat2str(last_OMT), '_rotation_c1c2vectors'), '-dpdf', '-bestfit'); % For saving as a PDF
    
    end

    % write again the vectors a1 and a2 to the directory
    writematrix(a1vec,append(subdirs,'\a1vec.txt'),'Delimiter',',')  
    writematrix(a2vec,append(subdirs,'\a2vec.txt'),'Delimiter',',')  
    
end

%

% axis off




