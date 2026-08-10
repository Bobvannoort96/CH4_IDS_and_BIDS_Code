% Delft coordinates (geodetic)
lat = 51.999;   % degrees
lon = 4.372;    % degrees
h   = 0;        % meters
recPos = [lat lon h];

% Observation time (UTC)
obsTime = datetime(2025,7,1,12,0,0,'TimeZone','UTC');

% Elevation mask
mask = 5;   % degrees

% Full path to merged RINEX 3 broadcast nav file
% Example: if the file is in C:\Users\YourName\Data\
rinexFile = "U:\IGNSS\BNoort\PhD\Python\Case Study\SS and Tq test power analysis\Results_of_some_examples\GNSS_example_only_VARAIM\Detection_only\Rinex\BRDC00IGS_R_20251820000_01D_MN.rnx";

%% Read RINEX file
rinexData = rinexread(rinexFile); 
navDataGPS = rinexData.GPS;
navDataGAL = rinexData.Galileo;

%% Load constellations
[satPosGPS, satVelGPS, satIDGPS] = gnssconstellation(obsTime, navDataGPS);
[satPosGAL, satVelGAL, satIDGAL] = gnssconstellation(obsTime, navDataGAL);

%% Compute look angles for GPS
[azGPS, elGPS, visGPS] = lookangles(recPos, satPosGPS, mask);

azGPS = azGPS(visGPS);
elGPS = elGPS(visGPS);
prnGPS = navDataGPS.SatelliteID(visGPS);
sysGPS = repmat("GPS", size(prnGPS));
labelsGPS = sysGPS + "-" + string(prnGPS);

%% Compute look angles for Galileo
[azGAL, elGAL, visGAL] = lookangles(recPos, satPosGAL, mask);

azGAL = azGAL(visGAL);
elGAL = elGAL(visGAL);
prnGAL = navDataGAL.SatelliteID(visGAL);
sysGAL = repmat("GAL", size(prnGAL));
labelsGAL = sysGAL + "-" + string(prnGAL);

%% Combine for plotting
az = [azGPS; azGAL];
el = [elGPS; elGAL];
labels = [labelsGPS; labelsGAL];
sys = [sysGPS; sysGAL];

%% Plot skyplot
figure
hold off
skyplot(az(sys=="GPS"), el(sys=="GPS"), labels(sys=="GPS"), 'MaskElevation', mask)
skyplot(az(sys=="GAL"), el(sys=="GAL"), labels(sys=="GAL"), 'MaskElevation', mask)

title("Skyplot of GPS and Galileo over Delft – 1 July 2025")
legend("GPS","Galileo")