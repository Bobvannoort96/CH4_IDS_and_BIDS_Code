% Add necessary paths
addpath('H:\My Documents\PhD\Python\Case Study\IDS\Sim Data\');


% Load fault vectors
fault_vectors = load('fault_vectors.txt');

[m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv] = setup();

% Select i, j, k
Pcti = P_mat(fault_vectors(:, 1), Qtt);
Pcti_perp = P_perp(Pcti);

Pctj_bar = P_mat(Pcti_perp * fault_vectors(:, 2), Qtt);
Pctk_bar = P_mat(Pcti_perp * fault_vectors(:, 3), Qtt);

% Compute the quadratic form matrix
Qmat = Pctj_bar' * Qtt_inv * Pctj_bar - Pctk_bar' * Qtt_inv * Pctk_bar;

Bxx = null(Qmat);

% Generate the 3D grid
[x, y, z, t_3D] = generate_t_grid(500, 500);
factor = 10;
t_3D = t_3D * factor;
% Calculate lengths
lengths = abs(sum((t_3D' * Qmat) .* t_3D', 2));

booleans = lengths < 1e-6;

% Create the figure and plot
figure;
ax = axes('Parent', gcf, 'Projection', 'perspective');
scatter3(ax, t_3D(1, booleans), t_3D(2, booleans), t_3D(3, booleans), '.', 'DisplayName', 'plane border Hij and Hik');
hold on
% Plot fault lines
plot_faultline(ax, fault_vectors(:, 1), 'yellow', 'P1');
hold on
plot_faultline(ax, fault_vectors(:, 2), 'red', 'P2');
hold on
plot_faultline(ax, fault_vectors(:, 3), 'blue', 'P3');
hold on
plot_faultline(ax, fault_vectors(:, 4), '#808080', 'P4');
hold on
% Set axis labels and limits
xlabel(ax, 't1');
ylabel(ax, 't2');
zlabel(ax, 't3');
xlim(ax, [-1, 1]);
ylim(ax, [-1, 1]);
zlim(ax, [-1, 1]);

legend(ax, 'show');
grid on
ax.DataAspectRatio = [1 1 1];
view(ax, 3);
view(ax, [37.5, 30]);


%% Function definitions

% Function to generate the 3D grid for the misclosure space
function [xa, ya, za, t_3Da] = generate_t_grid(n_samples_x, n_samples_y)
    ua = linspace(0, 2 * pi, n_samples_x);
    va = linspace(0, pi, n_samples_y);
    
    % Express in cartesian coordinates
    [U, V] = meshgrid(ua, va);
    xa = cos(U) .* sin(V);
    ya = sin(U) .* sin(V);
    za = ones(size(U)) .* cos(V);
    t_3Da = [xa(:)'; ya(:)'; za(:)'];
end

% Function to set up the problem
function [m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv] = setup()
    rng(20);
    m = 4;
    n = 1;
    r = m - n;
    A = ones(m, 1);
    alpha = 0.1;
    sigma = 1.0;
    Qyy = eye(m) * sigma^2;
    Qyy_inv = inv(Qyy);
    
    Bt = null(A')';
    
    Qtt = Bt * Qyy * Bt';
    Qtt_inv = inv(Qtt);
end

% Function to plot fault lines
function plot_faultline(ax, cti, color, partition)
    cti_plot = [cti * 10, cti * -10];
    plot3(ax, cti_plot(1,:), cti_plot(2,:), cti_plot(3,:), 'DisplayName', ['$c_{t_i}$ for ', partition], 'Color', color);
end

% Function to calculate the plus matrix
function result = plusmat(A, Qyy)
    result = inv(A' * inv(Qyy) * A) * A' * inv(Qyy);
end

% Function to calculate the P matrix
function result = P_mat(A, Qyy)
    result = A * plusmat(A, Qyy);
end

% Function to calculate the perpendicular projection matrix
function result = P_perp(P)
    result = eye(size(P, 1)) - P;
end