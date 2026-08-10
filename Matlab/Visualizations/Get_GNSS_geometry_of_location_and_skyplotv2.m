% Delft coordinates (geodetic)
lat = 51.999;   % degrees
lon = 4.372;    % degrees
h   = 0;        % meters
recPos = [lat lon h];

% Observation time (UTC)
% obsTime = datetime(2025,7,1,12,0,0,'TimeZone','UTC');
obsTime = datetime(2025, 12, 28, 12, 0, 0, 'TimeZone', 'UTC');
% Elevation mask
mask = 5;   % degrees

% Full path to merged RINEX 3 broadcast nav file
% Example: if the file is in C:\Users\YourName\Data\
% rinexFile = "U:\IGNSS\BNoort\PhD\Python\Case Study\SS and Tq test power analysis\Results_of_some_examples\GNSS_example_only_VARAIM\Detection_only\Rinex\BRDC00IGS_R_20251820000_01D_MN.rnx";
rinexFile = 'C:\Users\bgvannoort\PycharmProjects\Example_ARAIM_DIA_paper\Data\BRDC00IGS_R_20253620000_01D_MN.rnx';

%% Read RINEX file
rinexData = rinexread(rinexFile); 
navDataGPS = rinexData.GPS;
navDataGAL = rinexData.Galileo;

%% Load constellations
[satPosGPS, satVelGPS, satIDGPS] = gnssconstellation(obsTime, navDataGPS);
[satPosGAL, satVelGAL, satIDGAL] = gnssconstellation(obsTime, navDataGAL);

%% Compute look angles for GPS
% Compute look angles
[azGPS_all, elGPS_all, visGPS_all] = lookangles(recPos, satPosGPS, mask);

% Only visible satellites
azGPS_vis = azGPS_all(visGPS_all);
elGPS_vis = elGPS_all(visGPS_all);
prnGPS_vis = satIDGPS(visGPS_all);  % or navDataGPS.SatelliteID

% Keep only one entry per satellite
[prnGPS_unique, ia] = unique(prnGPS_vis);  % ia = indices of first occurrence
azGPS = azGPS_vis(ia);
elGPS = elGPS_vis(ia);
sysGPS = repmat("GPS", size(prnGPS_unique));
labelsGPS = sysGPS + "-" + string(prnGPS_unique);




%% Compute look angles for Galileo
[azGAL_all, elGAL_all, visGAL_all] = lookangles(recPos, satPosGAL, mask);
azGAL_vis = azGAL_all(visGAL_all);
elGAL_vis = elGAL_all(visGAL_all);
prnGAL_vis = satIDGAL(visGAL_all);

[prnGAL_unique, ia] = unique(prnGAL_vis);
azGAL = azGAL_vis(ia);
elGAL = elGAL_vis(ia);
sysGAL = repmat("GAL", size(prnGAL_unique));
labelsGAL = sysGAL + "-" + string(prnGAL_unique);

%% Combine for plotting
az = [azGPS; azGAL];
el = [elGPS; elGAL];
labels = [labelsGPS; labelsGAL];
sys = [sysGPS; sysGAL];

%% Select only those used in the example
selected_vals = [1, 2, 4, 7, 9, 10, 11, 12, 15, 16, 17, 19];
az = az(selected_vals); 
el = el(selected_vals);
labels = labels(selected_vals);
sys = sys(selected_vals);

%% Plot skyplot
figure
hold off

% For the example in the SS vs Tq paper
% selected_array = [20 17 4 14 19 16 11 5 3 18 7];
% skyplot(az(selected_array) , el(selected_array) , labels(selected_array) , ...
%     'MaskElevation', mask)
 
 
skyplot(az , el , labels, 'MaskElevation', mask)

% title("Skyplot of GPS and Galileo over Delft – 1 July 2025")
title("Skyplot of GPS and Galileo over Delft – 28 Dec 2025")
legend("GPS","Galileo")