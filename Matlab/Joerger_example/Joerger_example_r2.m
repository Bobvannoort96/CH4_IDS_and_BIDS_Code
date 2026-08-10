% Import fault vectors
fault_vectors = importdata('C:\Users\bgvannoort\Documents\PhD\Python\Case Study\Simple_examples\Example_Joerger_et_al_three_measurements\Partitions\fault_vectors.txt');

% Define path variables
pathname = 'C:\Users\bgvannoort\Documents\PhD\Python\Case Study\Simple_examples\Example_Joerger_et_al_three_measurements\Partitions\';
pathname_full_grid = pathname;

% Import grid data
x = importdata(fullfile(pathname_full_grid, 'grid_x.txt'));
y = importdata(fullfile(pathname_full_grid, 'grid_y.txt'));
z = importdata(fullfile(pathname_full_grid, 'grid_z.txt'));

% Define partitions and corresponding colors
partitions = ["P0", "P1", "P2", "P3", "P99"];

% Convert hex color codes to RGB triplets
colors_partitions_rgb = {[0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]};
dict_parts = containers.Map(partitions, colors_partitions_rgb);

% Initialize figure
figure();
hold on;
ax = gca;

% Initialize legend variables
h = [];  % Array to store handles for legend
p_name = {};  % Cell array to store legend labels
i = 1;

% Loop through partitions and plot data
for part = partitions
    datafile = fullfile(pathname, part + "_xx.txt");
    
    if exist(datafile, 'file') == 2
        % Import partition grid data
        xx = importdata(fullfile(pathname, part + "_xx.txt"));
        yy = importdata(fullfile(pathname, part + "_yy.txt"));
        zz = importdata(fullfile(pathname, part + "_zz.txt"));

        

        % Generate legend label
        hypt = str2double(replace(part, "P", ""));
        if hypt == 99
            p_name{i} = ['$\mathcal{P}_{\Omega}$'];
            % p_name{i} = ['E$_{\Omega}$'];
            % Plot partition as a surface
            h(i) = surf(xx, yy, zz, 'EdgeColor', 'none', 'FaceColor', dict_parts(part));
        elseif hypt == 0
            % Plot partition as a surface
            h(i) = surf(xx, yy, zz, 'EdgeColor', 'none', 'FaceColor', dict_parts(part));
            p_name{i} = ['$\mathcal{P}_{0}$'];
        else
            p_name{i} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
            % p_name{i} = ['E$_{' int2str(hypt) '}$'];
            h(i) = surf(xx, yy, zz, 'EdgeColor', 'none', 'FaceColor', dict_parts(part));
        end
        
        i = i + 1;
    end
end

% Plot the fault vectors
for i = 1:3
    if norm(fault_vectors(:, i)) ~= 0
        cti = fault_vectors(:, i) / norm(fault_vectors(:, i)); % Normalize vector
        
        plot3([cti(1), -cti(1)] * 20, [cti(2), -cti(2)] * 20, [0, 0], ...
              'Color', 'black', 'Marker', '+', 'LineWidth', 3, 'MarkerSize', 2, 'LineStyle','--');
    end
end   

% Set axis labels and limits
xlabel('t_1');
ylabel('t_2');
zlabel('t3');
xlim([-20, 20]);
ylim([-20, 20]);
zlim([-20,20]);
axis equal;

% Add legend only if there are valid handles
if ~isempty(h)
    legend(h, p_name, 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'best');
end

% View settings
view(0, 90);
set(gcf, 'PaperPositionMode', 'auto');



directory_ = 'U:\IGNSS\BNoort\PhD\Python\Case Study\Simple_examples\Example_Joerger_et_al_three_measurements';
exportgraphics(gcf, fullfile(directory_, 'figure.pdf'), 'ContentType', 'vector', ...
    'BackgroundColor', 'none', 'Resolution', 300);
% savefig(fullfile(directory_, 'figure1.')) 

hold off;
