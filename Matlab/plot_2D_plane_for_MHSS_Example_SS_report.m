%% Setup: method, paths, and partition definitions
method = 'Zhai';  % set to 'Zhai' to skip the PFA_Chi2 loop (not defined for Zhai)

zoomedOut= false;

if zoomedOut
    pathname = append('C:\Users\bgvannoort\Documents\Dissertation\Chapter 4 MHSS DIA\Data\Example_6_sats\Partitions_for_matlab\zoomedout\', method);
else
    pathname = append('C:\Users\bgvannoort\Documents\Dissertation\Chapter 4 MHSS DIA\Data\Example_6_sats\Partitions_for_matlab\', method);
end

% Fault (characteristic) vectors used to draw the fault directions
fault_vectors = importdata(append(pathname, '\fault_vectors.txt'));

% Partition labels and their plotting colors (SPP example)
partitions = ["P0" "P1" "P2" "P3" "P4" "P5" "P6" "P99"];
colors_partitions = ["#F5F5F5" "#1F77B4" "#FF7F0E" "#2CA02C" "#D62728" "#9467BD" "#8C564B" "black"];

% Map each partition label to its color
dict_parts = dictionary(partitions, colors_partitions);

%% Plotting the MHSS ARAIM partitioning regions projected on the unit sphere
% Surfaces are imported from the python-generated .txt files (one set of
% xx/yy/zz files per partition).

% Zhai has no PFA_Chi2 value defined, so it is plotted only once. For the
% other methods we loop over the PFA_Chi2 settings.
if strcmp(method, 'Zhai')
    PFA_Chi2_values = NaN;  % placeholder: not used in the load path for Zhai
else
    PFA_Chi2_values = [1e-4 1e-5 1e-6];
end

for PFA_Chi2 = PFA_Chi2_values
    figure
    set(gcf, 'Position', [100, 100, 900, 900]);  % 900x900 pixels
    ax = gca;
    hold on

    % Directory holding the partition surface files. Zhai has no PFA_Chi2
    % subfolder, the other methods do.
    if strcmp(method, 'Zhai')
        dir_path_string = append(pathname, '\');
    else
        dir_path_string = append(pathname, '\PFA_Chi2=', num2str(PFA_Chi2), '\');
    end

    % Handles and legend names, filled as partitions are plotted
    h = [];
    p_name = {};

    % Loop over partitions and plot each surface (if its file exists)
    for part = partitions
        datafile = append(dir_path_string, part, '_xx.txt');
        if exist(datafile, 'file') == 2
            xx = importdata(append(dir_path_string, part, '_xx.txt'));
            yy = importdata(append(dir_path_string, part, '_yy.txt'));
            zz = importdata(append(dir_path_string, part, '_zz.txt'));

            h(end+1) = surf(xx, yy, zz, 'EdgeColor', 'none', 'FaceColor', dict_parts(part));

            % Legend entry: P99 is the full hypothesis space, others are numbered
            hypt = str2num(replace(part, "P", ""));
            if part == 'P99'
                p_name{end+1} = ['$\mathcal{P}_{\Omega}$'];
            else
                p_name{end+1} = ['$\mathcal{P}_{' int2str(hypt) '}$'];
            end
        end
    end
    % Plot the normalized fault-vector directions as outlined dashed lines.
    % Each line is drawn twice: a thick black line first (the outline/border),
    % then a thinner colored line on top, since MATLAB lines have no native
    % border property. The lines are scaled well beyond the axis limits so
    % that clipping makes them span the whole figure, and lifted to z = zline
    % so they stay visible above the surfaces in the top-down view.
    factor = 10;        % large scale factor; the line is clipped to xlim/ylim
    shade = 0.65;   % <1 darkens the partition color; closer to 1 keeps it lighter

    for i = 1:6
        cti = fault_vectors(:,i) / vecnorm(fault_vectors(:,i));  % unit direction
        parts = append('P', num2str(i));

        zz1  = abs(factor / cti(1)); 
        zz2 = abs(factor / cti(2));
        L = min(zz1, zz2);
 
        base_rgb   = validatecolor(dict_parts(parts));  % partition color as RGB
        line_color = base_rgb * shade;                  % slightly darker shade

        plot3([-cti(1) cti(1)]*L, [-cti(2) cti(2)]*L, [0 0], ...
            'LineStyle', '--', 'LineWidth', 2.5, 'Color', line_color)
    end

    % Axis limits and labels
    xlim([-factor factor]);
    ylim([-factor factor]);
    zlim([-2 2]);
    xlabel('t1')
    ylabel('t2')
    zlabel('t3')

    % Legend (only if at least one surface was plotted)
    if ~isempty(h) && ~isempty(p_name)
        legend(h, p_name, 'Interpreter', 'latex', 'box', 'on', 'FontSize', 18);
    else
        warning('No valid surfaces found for legend.');
    end

    % % Title and export filename: drop PFA_Chi2 reference for Zhai
    % if strcmp(method, 'Zhai')
    %     title(append("Partitioning for ", method, " MHSS, with identity Q_{tt} matrix"), 'FontSize', 18)
    % else
    %     title(append("Partitioning for ", method, " MHSS, with identity Q_{tt} matrix, P_{FA, \chi^2} = ", num2str(PFA_Chi2)), 'FontSize', 18)
    % end

    axis equal
    axis off
    view(0, 90);

    % Export the figure as a vector PDF
    set(gcf, 'PaperPositionMode', 'auto');

    if zoomedOut
        subdirs = append('C:\Users\bgvannoort\Documents\Dissertation\Chapter 4 MHSS DIA\Figures\Example_6_sats\', method, '_partitioning_zoomed_out');
    else
        subdirs = append('C:\Users\bgvannoort\Documents\Dissertation\Chapter 4 MHSS DIA\Figures\Example_6_sats\', method, '_partitioning');
    end
    if strcmp(method, 'Zhai')
        export_name = append(subdirs, '\Partitioning_MHSS_ARAIM_', method, '.pdf');
    else
        export_name = append(subdirs, '\Partitioning_MHSS_ARAIM_', method, '_PFA_Chi2=', num2str(PFA_Chi2), '.pdf');
    end

    exportgraphics(gcf, export_name, 'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'vector');

    if PFA_Chi2 == 1e-4
        export_name = append(subdirs, '\Blanch_partitioning_with_Qtt_identity.pdf');
        title(append("Partitioning for ", method, " MHSS, with identity Q_{tt} matrix"), 'FontSize', 18)
        exportgraphics(gcf, export_name, 'Resolution', 300, 'BackgroundColor', 'none', 'ContentType', 'vector');
    end
end