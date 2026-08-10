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

%% Building a grid of points on the surface of a zero-centred unit sphere (3D)
t1 = 0:0.01:2*pi;
t2 = -pi/2:0.01:pi/2;
[tet,si] = meshgrid(t1,t2);
fault_vectors = importdata('C:\Users\bgvannoort\Documents\IDS\Sim Data\fault_vectors.txt');
if ~bool_low_res
    pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
    pathname = 'C:\Users\bgvannoort\Documents\PhD\Python\IDS_IDS\Sim Data\';
else
    pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\lower_res\';
end
pathname_full_grid = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\';
x = importdata(append(pathname_full_grid, 'grid_x.txt'));
y = importdata(append(pathname_full_grid, 'grid_y.txt'));
z = importdata(append(pathname_full_grid, 'grid_z.txt'));
% factor = 6;
% 
% filename = append(pathname, 't_3D_data_factor_', int2str(factor), '.txt');
% fprintf(filename)
% t_3D = importdata( filename);
% 
% identifications = importdata(append(pathname, 'corresponding_identifications_factor_', int2str(factor) ));
% 
% v = [x(:)';y(:)';z(:)']; %collection of unit vectors in 3D
% v = t_3D;
% mv = size(v,2);

%% Computing w-tests for all the points in the generated grid

% ct = B';
% w = zeros(m,mv);
% for i = 1 : m
%     cti = ct(:,i);
%     cinorm = vecnorm(cti);
%     w(i,:) = abs((cti'*v)/cinorm);
% end
% [~,u] = max(w,[],1);%returning the index of the selected hypotheses based upon samples of t


partitions = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P21" "P23" "P24" "P31" "P32" "P34" "P41" "P42" "P43"];

partitions_legend = ["P0" "P1" "P2" "P3" "P4" "P12" "P13" "P14" "P23" "P24" "P34"];

colors_partitions = ["black" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FFAA00" "magenta" "#FF9F65" "green" "magenta" "#65F1FF" "#D0B623" "#FF9F65" "#65F1FF"];

colors_partitions_alldiff = ["black" "yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FACE76" "magenta" "#FF9F65" "#6BFF64" "#FFAAF6" "#65F1FF" "#C19929" "#C45715" "#018A98"];

dict_parts = dictionary(partitions, colors_partitions);
dict_parts_alldiff = dictionary(partitions, colors_partitions_alldiff);

% u = identifications;
%% Plotting the IDS partitioning regions projected on the unit sphere, 
% imported from python generated .txt files

% loop over factors
for factor=[6  11 21]

    figure
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
    title('IDS partitioning')
    legend(h,p_name,'Interpreter','latex','box','off','FontSize',14)
    axis equal
    axis 'off'

    view(45, 30);
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    
    % topDir = 'C:\Users\bgvannoort\Documents\IDS\';
    % % Print the figure using the '-bestfit' option
    % print(gcf, append(topDir, 'Figures_report\Partitioning_IDS_noaxis_v1_factor_', num2str(factor), 'separate'), '-dpdf', '-bestfit'); % For saving as a PDF
    % 
    % view(135, 30);
    % % turn of legend for this one
    % legend('off')
    % % Print the figure using the '-bestfit' option
    % print(gcf, append(topDir, 'Figures_report\Partitioning_IDS_noaxis_v2_factor_', num2str(factor), 'separate'), '-dpdf', '-bestfit'); % For saving as a PDF
end

%

% axis off

%% Plot the normal DIA method (k=10)

factors = 2.0:0.2:10.0;
factors = horzcat(factors, 11:1:36);
contents = dir(append(pathname, 'ordinary_DIA\Partitioned_grid'));
contents = contents(~ismember({contents.name}, {'.', '..'}));
folders = contents.name;

% for entry = factors
for el = 1:length(contents)
    folder_name = contents(el).name;
    figure
    set(gcf, 'Position', [100, 100, 1000, 800]); % [left, bottom, width, height]
    surf(x,y,z,'EdgeColor','none','FaceColor','white');
    hold on
    % for i = 1 : k
    i = 1;
    fprintf("Ordinary DIA")
    % factor = entry;%round(entry,2);
    factor = str2num(replace(folder_name, 'factor_', ''));
    % factor = 200;
    % folder_name = append('factor_', num2str(factor));
    for part = partitions_legend
        % xx = x';
        % xx = importdata(append(pathname, 'ordinary_DIA\Partitioned_grid\factor_', num2str(factor),'\', part, '_xx.txt'));
        % yy = importdata(append(pathname, 'ordinary_DIA\Partitioned_grid\factor_', num2str(factor),'\', part, '_yy.txt'));
        % zz = importdata(append(pathname, 'ordinary_DIA\Partitioned_grid\factor_', num2str(factor),'\', part, '_zz.txt'));
        xx = importdata(append(pathname, 'ordinary_DIA\Partitioned_grid\', folder_name, '\', part, '_xx.txt'));
        yy = importdata(append(pathname, 'ordinary_DIA\Partitioned_grid\', folder_name,'\', part, '_yy.txt'));
        zz = importdata(append(pathname, 'ordinary_DIA\Partitioned_grid\', folder_name,'\', part, '_zz.txt'));
        fprintf(part)
        fprintf('\n')
        fprintf(dict_parts(part))
        fprintf('\n')
        if any(partitions_legend == part)
            tt(i)=surf(xx,yy,zz,'EdgeColor','none','FaceColor', dict_parts(part));
            hypt = str2num(replace(part, "P", ""));
            p_name_ordDIA{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
            i=i+1;
        else
            % these are the identifications P21, P31, P41, P42, P43 and P32. 
            surf(xx,yy,zz,'EdgeColor','none','FaceColor',dict_parts(part));
        end
            % p_name{i} = ['$\mathcal{P}_{' num2str(i) '}$'];
        
    end
    for i = 1:4
        cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
    end   
    title(append('Classical DIA R=', num2str(factor)))
    legend(tt,p_name_ordDIA,'Interpreter','latex','box','off','FontSize',14)
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')
    view(3)
    axis equal
    % saveas(gcf, append(pathname, 'ordinary_DIA\Partition_factor_', num2str(factor), '.png'))

    axis off
    % view(45, 30);
    % % Set the PaperPositionMode to 'auto'
    % set(gcf, 'PaperPositionMode', 'auto');
    % 
    % topDir = 'C:\Users\bgvannoort\Documents\IDS\';
    % % Print the figure using the '-bestfit' option
    % print(gcf, append(topDir, 'Figures_report\classical_DIA\Partitioning_class_DIA_noaxis_v1_factor_', num2str(factor)), '-dpdf', '-bestfit'); % For saving as a PDF
    % 
    % view(135, 30);
    % % turn of legend for this one
    % legend('off')
    % % Print the figure using the '-bestfit' option
    % print(gcf, append(topDir, 'Figures_report\classical_DIA\Partitioning_class_DIA_noaxis_v2_factor_', num2str(factor)), '-dpdf', '-bestfit'); % For saving as a PDF

    % for i = 1 : m
    %     h(i)= plot3(v(1,u==i),v(2,u==i),v(3,u==i),'Color',[1 1 1]*i/7,'Marker','.','LineStyle','none');
    % end
    % legend(h,p_name,'Interpreter','latex','box','off','FontSize',14)
    % axis equal
    % axis off
end
%% For ordinary DS (k=4)
partitions_DS = ["P1" "P2" "P3" "P4"];
figure
surf(x,y,z,'EdgeColor','none','FaceColor','white');
hold on
% for i = 1 : k
i = 1;
fprintf("DIA Data snooping")
factor=11;
for part = partitions_DS
    % xx = x';
    xx = importdata(append(pathname, 'DS_DIA\Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt'));
    yy = importdata(append(pathname, 'DS_DIA\Partitioned_grid\factor_', num2str(factor), '\', part, '_yy.txt'));
    zz = importdata(append(pathname, 'DS_DIA\Partitioned_grid\factor_', num2str(factor), '\', part, '_zz.txt'));
    % fprintf(part)
    % fprintf('\n')
    % fprintf(dict_parts(part))
    % fprintf('\n')
    if any(partitions_legend == part) % for IDS relevant if e.g. P43 is considered.
        ll(i)=surf(xx,yy,zz,'EdgeColor','none','FaceColor', dict_parts(part));
        hypt = str2num(replace(part, "P", ""));
        p_name_DSDIA{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
        i=i+1;
    else
        surf(xx,yy,zz,'EdgeColor','none','FaceColor',dict_parts(part));
    end
    for l_i = 1:4
        cti = fault_vectors(:,l_i)/vecnorm(fault_vectors(:,l_i));
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',2)
    end   
end

title('Datasnooping DIA')
legend(ll,p_name_DSDIA,'Interpreter','latex','box','off','FontSize',14)
axis equal
axis off

view(45, 30);
% Set the PaperPositionMode to 'auto'
set(gcf, 'PaperPositionMode', 'auto');

topDir = 'C:\Users\bgvannoort\Documents\IDS\';
% Print the figure using the '-bestfit' option
print(gcf, append(topDir, 'Figures_report\Partitioning_DS_DIA_noaxis_v1'), '-dpdf', '-bestfit'); % For saving as a PDF

view(135, 30);
% turn of legend for this one
legend('off')
% Print the figure using the '-bestfit' option
print(gcf, append(topDir, 'Figures_report\Partitioning_DS_DIA_noaxis_v2'), '-dpdf', '-bestfit'); % For saving as a PDF

