bool_low_res = false;

% SphereVisualisationIDS(A, Qyy)

%% Description


%% Inputs
% A   : m x n design matrix
% Qy : m x m observation VCV matrix 


%% Computing B-matrix and misclosure VCV matrix


%% Building a grid of points on the surface of a zero-centred unit sphere (3D)
t1 = 0:0.01:2*pi;
t2 = -pi/2:0.01:pi/2;
[tet,si] = meshgrid(t1,t2);
type_of_example = 'SPP_GNSS';
fault_vectors = importdata(['C:\Users\bgvannoort\Documents\IDS\Sim Data\' type_of_example '\fault_vectors.txt']);
pathname = ['C:\Users\bgvannoort\Documents\IDS\Sim Data\' type_of_example '\'];
 
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

path_to_color_dict = ['C:\Users\bgvannoort\Documents\IDS\Sim Data\' type_of_example '\colors_partitions_dict.json'];
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
        dir_path_string = append(pathname, 'ordinary_DIA\no_separate_partitionings\Partitioned_grid\factor_', num2str(factor), '\');
        % dir_path_string = append(pathname, 'ordinary_DIA\penalized_setup1\factor_', num2str(factor), '\', 'Partitioned_grid\'); %uncomment later
        datafile = append(dir_path_string, part, '_xx.txt');
        if exist(datafile, 'file')==2
            xx = importdata(append(dir_path_string, part, '_xx.txt'));
            yy = importdata(append(dir_path_string, part, '_yy.txt'));
            zz = importdata(append(dir_path_string, part, '_zz.txt'));
        
            fprintf(part)
            fprintf('\n')
            fprintf(dict_parts(part))
            fprintf('\n')

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
            
        end
    end
    % Plotting the ending points of the normalized cti vectors
    for i = 1:7
        cti = fault_vectors(:,i)/vecnorm(fault_vectors(:,i));
        plot3(cti(1),cti(2),cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',4)
        plot3(-cti(1),-cti(2),-cti(3),'Color','black','Marker','+','LineWidth',3,'MarkerSize',4)
    end   
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')

    title(append('Classical DIA partitioning R=', num2str(factor)))
    
    legend(h,p_name,'Interpreter','latex','box','on','FontSize',14, 'Location', 'best')
    axis equal
    axis 'off'
    ax.Position = [0 0 0.8750 0.9150];
    view(45,30);
    % ax = gca;
    % outerPos = ax.OuterPosition;
    % tightInset = ax.TightInset;
    % ax.Position = [outerPos(1) + tightInset(1), ...
    %                outerPos(2) + tightInset(2), ...
    %                outerPos(3) - tightInset(1) - tightInset(3), ...
    %                outerPos(4) - tightInset(2) - tightInset(4)];
    % view(45, 30);
    % viewdirection_ct1 = fault_vectors(:,3)/vecnorm(fault_vectors(:,3));
    % 
    % 
    % campos(-viewdirection_ct1*2)
    % Set the PaperPositionMode to 'auto'
    set(gcf, 'PaperPositionMode', 'auto');
    
    topDir = 'C:\Users\bgvannoort\Documents\IDS\'; 
    outDir  = append(topDir, 'Figures_report\SPP_GNSS\classical_DIA\');
    if ~exist(outDir, 'dir'); mkdir(outDir); end   % creates all missing parent folders too
    print(gcf, append(topDir, ['Figures_report\' type_of_example '\classical_DIA\Partitioning_classical_DIA_noaxis_v1_factor_', num2str(factor)]), '-dpdf', '-bestfit'); % For saving as a PDF
    exportgraphics(gcf, append(topDir, ['Figures_report\' type_of_example '\classical_DIA\Partitioning_classical_DIA_noaxis_v1_factor_', num2str(factor) ], '.png'), 'Resolution', 300)
    view(135, 30);
    % turn of legend for this one
    legend('off')

    % Print the figure using the '-bestfit' option
    print(gcf, append(topDir, ['Figures_report\' type_of_example '\classical_DIA\Partitioning_classical_DIA_noaxis_v2_factor_', num2str(factor)]) , '-dpdf', '-bestfit'); % For saving as a PDF
    exportgraphics(gcf, append(topDir, ['Figures_report\' type_of_example '\classical_DIA\Partitioning_classical_DIA_noaxis_v2_factor_', num2str(factor)], '.png'), 'Resolution', 300)
end

%
