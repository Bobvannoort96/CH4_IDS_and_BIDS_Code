% MATLAB version of the given Python script
% Created on Fri Jun 14 09:45:20 2024
% This script computes the border (cross section) of the partitioning in the 
% t-space of the hypotheses Hij and Hik to check if it is a plane.

function main
    % Generate t-grid in polar coordinates for the misclosure space
    [x, y, z, t_3D] = generate_t_grid(1000, 1000);
    
    % Define the problem setup
    [m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv] = setup();
    
    % Load fault vectors
    fault_vectors = load('H:\My Documents\PhD\Python\Case Study\IDS\Sim Data\fault_vectors.txt');
    
    % Calculate projection matrices
    Pcti = P_mat(fault_vectors(:, 1), Qtt);
    Pcti_perp = P_perp(Pcti);

    Pctj_bar = P_mat(Pcti_perp * fault_vectors(:, 2), Qtt);
    Pctk_bar = P_mat(Pcti_perp * fault_vectors(:, 3), Qtt);

    Qmat = Pctj_bar' * Qtt_inv * Pctj_bar - Pctk_bar' * Qtt_inv * Pctk_bar;

    Bxx = null(Qmat);

    % Plot setup
    figure;
    hold on;
    grid on;
    axis equal;
    
    % Plot fault lines and planes
    p1=plot_faultline(fault_vectors(:, 1), 'yellow', 'c_{t_1}');
    p2=plot_faultline(fault_vectors(:, 2), 'red', 'c_{t_2}');
    p3=plot_faultline(fault_vectors(:, 3), 'blue', 'c_{t_3}');
    p4=plot_faultline(fault_vectors(:, 4), '#808080', 'c_{t_4}');

    set(p1, 'HandleVisibility', 'off')
    set(p2, 'HandleVisibility', 'off')
    set(p3, 'HandleVisibility', 'off')
    set(p4, 'HandleVisibility', 'off')
    
    % Basis for the middle plane
    [n_middle, v_middle] = middle_plane_basis(fault_vectors(:, 1), fault_vectors(:, 2), fault_vectors(:, 3));

    plot_plane(fault_vectors(:, 2), fault_vectors(:, 4), 'P24', "#FF9F65", 1);
    plot_plane(fault_vectors(:, 1), fault_vectors(:, 3), 'P13', "green", 1);

    % plot_plane(fault_vectors(:, 1), fault_vectors(:, 4), 'P14', '#808080', 0.1)
    % plot_plane(fault_vectors(:, 3), fault_vectors(:, 4), 'P34', '#808080', 0.1)
    % plot_plane(fault_vectors(:, 2), fault_vectors(:, 3), 'P23', '#808080', 0.1)
    % plot_plane(fault_vectors(:, 1), fault_vectors(:, 2), 'P12', '#808080', 0.1)
        
    % Final plot properties
    xlabel('t_1');
    ylabel('t_2');
    zlabel('t_3');
    xlim([-10,10])
    ylim([-10,10])
    zlim([-10,10])
    % title('Fault planes for all q=2 hypotheses');
    % legend;
    view(3);
end

% Helper functions
function [x, y, z, t_3D] = generate_t_grid(n_samples_x, n_samples_y)
    ua = linspace(0, 2 * pi, n_samples_x);
    va = linspace(0, pi, n_samples_y);
    
    % Express in Cartesian coordinates
    [U, V] = meshgrid(ua, va);
    x = cos(U) .* sin(V);
    y = sin(U) .* sin(V);
    z = ones(size(U)) .* cos(V);
    t_3D = [x(:)'; y(:)'; z(:)'];
end

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

function P = P_mat(A, Qyy)
    P = A * (inv(A' * inv(Qyy) * A) * A' * inv(Qyy));
end

function P_perp = P_perp(P)
    P_perp = eye(size(P, 1)) - P;
end

function [n_middle, v_middle] = middle_plane_basis(a1, a2, a3)
    % Calculate normal vectors of planes
    n1 = cross(a1, a2);
    n2 = cross(a1, a3);
    
    % Normalize
    n1 = n1 / norm(n1);
    n2 = n2 / norm(n2);
    
    % Calculate the middle plane normal
    n_middle = cross(n1, n2);
    n_middle = n_middle / norm(n_middle);
    
    % Second basis vector
    v_middle = n1 + n2;
    v_middle = v_middle / norm(v_middle);
end

function plot_plane(a1, a2, partition, color, alpha)
    % Plot the fault vectors here (note that there are 2 here, so in fact a plane)
    if nargin < 6
        alpha = 0.5;
    end
    
    tmin = -10;
    tmax = 10;
    normal = cross(a1(:), a2(:));
    d_plane = -dot(a1(:), normal);
    
    if abs(normal(3)) < 1e-8  % The normal component of z is 0, plane lies in whole z-dimension
        [xx, z_val] = meshgrid(tmin:tmax, tmin:tmax);
        yy = (-normal(1) * xx - normal(3) * z_val - d_plane) * 1 / normal(2);
        
    elseif abs(normal(1)) < 1e-8
        [xx, z_val] = meshgrid(tmin:tmax, tmin:tmax);
        yy = (-normal(1) * xx - normal(3) * z_val - d_plane) * 1 / normal(2);
        
    else
        [xx, yy] = meshgrid(tmin:tmax, tmin:tmax);
        z_val = (-normal(1) * xx - normal(2) * yy - d_plane) * 1 / normal(3);
    end

    % Plot the surface on the specified axes
    surf(xx, yy, z_val, 'FaceColor', color, 'FaceAlpha', alpha, 'EdgeColor', 'none', 'DisplayName', ['fault plane for ' partition]);
end

function p = plot_faultline(fault_vector, color, partition)
    if nargin < 3
        cti_plot = [fault_vector * 10, fault_vector * -10];
        p = plot3(cti_plot(1, :), cti_plot(2, :), cti_plot(3, :), 'Color', color);
    else
        cti_plot = [fault_vector * 10, fault_vector * -10];
        p = plot3(cti_plot(1, :), cti_plot(2, :), cti_plot(3, :), 'Color', color, 'DisplayName', partition);
    end
    
end
