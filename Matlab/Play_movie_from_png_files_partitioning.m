% This script plays the partitions as a video when zooming out in 3D

directory = 'C:\Users\bgvannoort\Documents\IDS\Sim Data\ordinary_DIA\';


playMovieFromPNG(directory)














%% functions
function playMovieFromPNG(pathname)
    % playMovieFromPNG - Function to play a movie from a sequence of PNG images
    % Inputs:
    %   pathname - Path to the folder containing the PNG files

    % Get a list of all PNG files in the folder
    pngFiles = dir(fullfile(pathname, 'Partition_factor_*.png'));
    
    % Check if any PNG files are found
    if isempty(pngFiles)
        error('No PNG files found in the specified folder.');
    end
    
    % Extract the numerical factors from the filenames and sort them
    factors = nan(length(pngFiles), 1);
    for k = 1:length(pngFiles)
        % Extract the factor from the filename (assuming 'name_<factor>.png' format)
        [~, filename, ~] = fileparts(pngFiles(k).name);
        factorStr = regexp(filename, 'Partition_factor_(.*)', 'tokens', 'once');
        factors(k) = str2double(factorStr{1});
    end
    
    % Sort the files based on the extracted factors
    [~, sortedIndices] = sort(factors);
    sortedFiles = pngFiles(sortedIndices);
    
    % Set up a figure for displaying the images
    figure;
    axis off; % Turn off the axis for better visualization

    % Loop through each sorted PNG file and display it
    for k = 1:length(sortedFiles)
        % Read the current PNG file
        img = imread(fullfile(pathname, sortedFiles(k).name));

        % Display the image
        imshow(img);

        % Pause for a short duration to create the animation effect
        pause(0.3); % Adjust the pause duration as needed

        % Optionally, you can add a title or other annotations
        title(sprintf('Frame %d of %d', k, length(sortedFiles)));
    end

    % Optionally, save the frames as a video file
    % Create a VideoWriter object
    video = VideoWriter(fullfile(pathname, 'Partition_movie_ordinary_DIA.avi'));
    open(video);

    % Loop through each sorted PNG file again to save the frames to the video
    for k = 1:length(sortedFiles)
        % Read the current PNG file
        img = imread(fullfile(pathname, sortedFiles(k).name));
        pause(0.3);
        % Write the frame to the video
        writeVideo(video, img);
    end

    % Close the VideoWriter object
    close(video);
end

