% Skyplot from ENU design matrix A
% Each row of A: [e, n, u, 1] (unit line-of-sight vector in ENU + clock column)

nr_of_sats = 6;
if nr_of_sats ==7
    A = [ 0.0225,  0.9951, -0.0966, 1;
          0.6750, -0.6900, -0.2612, 1;
          0.0723, -0.6601, -0.7477, 1;
         -0.9398,  0.2553, -0.2269, 1;
         -0.5907, -0.7539, -0.2877, 1;
         -0.3236, -0.0354, -0.9455, 1;
        -0.6748,  0.4356, -0.5957,  1];
    
    % Extract ENU components (columns 1-3)
    e = A(:,1);
    n = A(:,2);
    u = A(:,3);
    
    % Convert ENU unit vector to azimuth and elevation
    % Azimuth: measured clockwise from North
    az = atan2d(e, n);          % [deg], result in (-180, 180]
    az(az < 0) = az(az < 0) + 360;  % wrap to [0, 360)
    
    % Elevation: angle above the horizontal plane
    el = asind(-u);              % [deg]
    
    % Unique markers for each satellite
    markers = {'o', 's', '^', 'd', 'p', 'h'};  % circle, square, triangle, diamond, pentagram, hexagram
    colors  = lines(6);                          % 6 distinct colors
    
    numSats = size(A, 1);
    
    figure;
    ax = skyplot(az, el, ["Sat 1" "Sat 2" "Sat 3" "Sat 4" "Sat 5" "Sat 6" "Sat 7"])';
    ax.LabelFontSize=12;
else
    A = [ 0.0225,  0.9951, -0.0966, 1;
      0.6750, -0.6900, -0.2612, 1;
      0.0723, -0.6601, -0.7477, 1;
     -0.9398,  0.2553, -0.2269, 1;
     -0.5907, -0.7539, -0.2877, 1;
     -0.3236, -0.0354, -0.9455, 1];

    % Extract ENU components (columns 1-3)
    e = A(:,1);
    n = A(:,2);
    u = A(:,3);
    
    % Convert ENU unit vector to azimuth and elevation
    % Azimuth: measured clockwise from North
    az = atan2d(e, n);          % [deg], result in (-180, 180]
    az(az < 0) = az(az < 0) + 360;  % wrap to [0, 360)
    
    % Elevation: angle above the horizontal plane
    el = asind(-u);              % [deg]
    
    % Unique markers for each satellite
    markers = {'o', 's', '^', 'd', 'p', 'h'};  % circle, square, triangle, diamond, pentagram, hexagram
    colors  = lines(6);                          % 6 distinct colors
    
    numSats = size(A, 1);
    
    figure;
    ax = skyplot(az, el, ["Sat 1" "Sat 2" "Sat 3" "Sat 4" "Sat 5" "Sat 6"])';
    ax.LabelFontSize=12;
end