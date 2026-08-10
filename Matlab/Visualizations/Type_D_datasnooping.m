% Parameters
m = 3;
n = 1;
A = ones(m, 1);
alpha = 0.01;
alpha_i = 0.001;
chi2_alpha = chi2inv(1 - alpha, m - n);
chi2_alpha_i = chi2inv(1 - alpha_i, m - n - 1);
Qyy = eye(m);

% Meshgrid
tmin = -20;
tmax = 20;
N = 1000;
[t1, t2] = meshgrid(linspace(tmin, tmax, N), linspace(tmin, tmax, N));
t = [t1(:)'; t2(:)'];

% Null space and Qtt
B = null(A');
Qtt = B' * Qyy * B;
invQtt = inv(Qtt);

% Compute OMTS values
OMTS = sum((invQtt * t) .* t, 1);

% W-tests
w_tests = zeros(m, size(t, 2));
for i = 1:m
    I = eye(m);
    ci = I(:, i);
    cti = B' * ci;
    P = cti * inv(cti' * invQtt * cti) * cti';
    Pctit = P * t;
    w_tests(i, :) = sum((invQtt * Pctit) .* Pctit, 1);
end

[max_wtests, idx_max_w] = max(w_tests, [], 1);
idx_H0_accepted = OMTS < chi2_alpha;
idx_undec = ~idx_H0_accepted & (OMTS - max_wtests > chi2_alpha_i);

% Build region map: -1 for H0 accepted, 0..m-1 for sensors, -2 for undecided
region_map = nan(size(t1));
region_map_flat = region_map(:);

region_map_flat(idx_H0_accepted) = -1;
for i = 1:m
    idx = (~idx_H0_accepted) & (idx_max_w == i);
    region_map_flat(idx) = i - 1;  % MATLAB indices start at 1
end
region_map_flat(idx_undec) = 4;
region_map = reshape(region_map_flat, size(t1));

% Define colors: first is white, then shades of grey
gray_colors = [1 1 1;              % white for P0 (H0 accepted)
               0.8 0.8 0.8;        % light grey
               0.6 0.6 0.6;
               0.4 0.4 0.4;
               0.1 0.1 0.1];       % dark grey (or extend as needed)

% Define corresponding labels
labels = {'$\mathcal{P}_0$', ...
          '$\mathcal{P}_1$', ...
          '$\mathcal{P}_2$', ...
          '$\mathcal{P}_3$', ...
          '$\mathcal{P}_\Omega$'};  % Add more if needed

% Unique region labels
unique_vals = unique(region_map(~isnan(region_map)));
num_regions = length(unique_vals);

% Start plot
figure; hold on; view(2);
Z = zeros(size(t1));  % Flat Z for 2D appearance
legend_handles = gobjects(num_regions, 1);

for i = 1:num_regions
    val = unique_vals(i)
    mask = (region_map == val);
    
    % Use NaN to mask out other areas
    Z_masked = Z;
    Z_masked(~mask) = NaN;
    
    % Plot surface manually with color
    h = surf(t1, t2, Z_masked, ...
             'EdgeColor', 'none', ...
             'FaceColor', gray_colors(i, :));
    gray_colors(i, :)
    % Add to legend
    legend_handles(i) = patch(NaN, NaN, gray_colors(i, :));
end

% --- Plot lines defined by each cti (i = 1 to m) ---
for i = 1:m
    ci = eye(m, m);  % Identity matrix
    ci = ci(:, i);   % Extract column i
    cti = B' * ci;   % Get cti vector in misclosure space

    % Define line direction and orthogonal vector
    % Line: cti' * t = 0 => line perpendicular to cti

    % To find line direction, get nullspace of cti'
    dir_vec = cti;  % 2x1 vector orthogonal to cti

    % Parametrize the line over a suitable range
    t_line = linspace(-2 * tmax, 2 * tmax, 100);
    points = dir_vec * t_line;  % 2 x 100


    zz = plot(points(1, :), points(2, :), 'k--', 'LineWidth', 1.5);
    
end

% Create dummy handle for the dashed black lines
dashed_line_handle = plot(nan, nan, 'k--', 'LineWidth', 1.5);

% Add it to the legend handles and labels
legend_handles(end+1) = dashed_line_handle;
labels{end+1} = '$c_{t_i}$';


xlabel('$t_1$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$t_2$', 'Interpreter', 'latex', 'FontSize', 15);
title({'Misclosure space partitioning', ...
       '(Type D Data Snooping)'}, ...
       'Interpreter', 'latex', 'FontSize', 15);


axis tight;
axis equal;

xlim([tmin, tmax]);
ylim([tmin, tmax]);

legend(legend_handles, labels(1:length(legend_handles)), ...
       'Interpreter', 'latex', ...
       'Location', 'northeast', ...
       'Box', 'on', ...
       'FontSize', 15);

hold off;