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

fault_vectors = importdata('C:\Users\bgvannoort\Documents\IDS\Sim Data\fault_vectors.txt');

pathname = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\IDS\separate_partitionings\';
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
partitions = ["P1" "P2" "P3" "P4" "P12" "P13" "P14" "P21" "P23" "P24" "P31" "P32" "P34" "P41" "P42" "P43", "P99"];

partitions_legend = ["P1" "P2" "P3" "P4" "P12" "P13" "P14" "P23" "P24" "P34", "P99"];

colors_partitions = ["yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FFAA00" "magenta" "#FF9F65" "green" "magenta" "#65F1FF" "#D0B623" "#FF9F65" "#65F1FF", "black"];

colors_partitions_alldiff = ["yellow" "red" "blue" "#A99BA7" "#FFAA00" "green" "#D0B623" "#FACE76" "magenta" "#FF9F65" "#6BFF64" "#FFAAF6" "#65F1FF" "#D0B623" "#C45715" "#018A98", "black"];

dict_parts = dictionary(partitions, colors_partitions);
dict_parts_alldiff = dictionary(partitions, colors_partitions_alldiff);

% u = identifications;
%% Plotting the IDS partitioning regions projected on the unit sphere, 
% imported from python generated .txt files
fileExists = true;
% loop over factors
for factor=[6 11 21]

    figure
    surf(x,y,z,'EdgeColor','none','FaceColor','white');
    hold on
    % for i = 1 : k
    i = 1;
    for part = partitions
        if part == "P99" % it is the undecided region. Sometimes, may not be present in directory, so check for that. 
            if exist(append(pathname, 'Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt'), "file")
                fileExists = true;
            else
                fileExists=false;
            end
        else
            fileExists=true;
        end
        if fileExists
            xx = importdata(append(pathname, 'Partitioned_grid\factor_', num2str(factor), '\', part, '_xx.txt'));
            yy = importdata(append(pathname, 'Partitioned_grid\factor_', num2str(factor), '\', part, '_yy.txt'));
            zz = importdata(append(pathname, 'Partitioned_grid\factor_', num2str(factor), '\', part, '_zz.txt'));
    
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
                % surf(xx,yy,zz, 'EdgeColor','none','FaceColor',dict_parts(part)) %
                
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
    axis off
    
    % view(45, 30);
    % % Set the PaperPositionMode to 'auto'
    % set(gcf, 'PaperPositionMode', 'auto');
    % 
    % topDir = 'C:\Users\bgvannoort\Documents\IDS\';
    % % Print the figure using the '-bestfit' option
    % print(gcf, append(topDir, 'Figures_report\IDS last OMT\separate_partitionings\Partitioning_IDS_DIA_noaxis_factor_', num2str(factor), '_v1'), '-dpdf', '-bestfit'); % For saving as a PDF
    % 
    % view(135, 30);
    % % turn of legend for this one
    % legend('off')
    % % Print the figure using the '-bestfit' option
    % print(gcf, append(topDir, 'Figures_report\IDS last OMT\separate_partitionings\Partitioning_IDS_DIA_noaxis_factor_', num2str(factor), '_v2'), '-dpdf', '-bestfit'); % For saving as a PDF

end
