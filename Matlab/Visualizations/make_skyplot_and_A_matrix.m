% Example satellite az/el angles
az = [0, 45, 90, 135, 180, 225, 270, 315];
el = [30, 50, 20, 70, 40, 60, 15, 80];

% Compute design matrix and plot
A = gnss_design_matrix_skyplot(az, el);

disp('Design Matrix (ENU coordinates):');
disp(A);

skyplot(az, el)

function [A] = gnss_design_matrix_skyplot(az_deg, el_deg)
% GNSS_DESIGN_MATRIX_SKYPLOT
%   Computes the GNSS design matrix (ENU coordinates) and plots a skyplot
%   for the given azimuth and elevation angles.
%
% Inputs:
%   az_deg - vector of satellite azimuths [degrees] (0° = North, CW positive)
%   el_deg - vector of satellite elevations [degrees] (0° = horizon, 90° = zenith)
%
% Output:
%   A - design matrix [num_sats x 4] in ENU coordinates

    if numel(az_deg) ~= numel(el_deg)
        error('Azimuth and elevation vectors must have the same length.');
    end

    % Convert to radians
    az_rad = deg2rad(az_deg);
    el_rad = deg2rad(el_deg);

    % Number of satellites
    num_sats = numel(az_deg);

    % Preallocate design matrix
    A = zeros(num_sats, 4);

    % Compute ENU direction cosines
    A(:,1) = cos(el_rad) .* sin(az_rad);  % East
    A(:,2) = cos(el_rad) .* cos(az_rad);  % North
    A(:,3) = sin(el_rad);                 % Up
    A(:,4) = 1;                            % Clock bias

    % ===== Skyplot =====
    figure;
    polaraxes;
    hold on;

    % Skyplot convention: radius = 90 - elevation
    r = 90 - el_deg;
    theta = deg2rad(az_deg);

    % Plot satellites
    for i = 1:num_sats
        polarplot(theta(i), r(i), 'o', 'MarkerFaceColor','b');
        text(theta(i), r(i), sprintf(' PRN%d', i), 'FontSize', 8, 'Color', 'k');
    end

    % Adjust polar axes
    pax = gca;
    pax.ThetaZeroLocation = 'top';   % 0° = North
    pax.ThetaDir = 'clockwise';      % Azimuth clockwise
    rlim([0 90]);
    rticks(0:15:90);
    title('GNSS Skyplot');

    hold off;
end