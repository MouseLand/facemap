# FaceMap
matlab GUI for processing face camera data from rodents
(( works for GRAYSCALE and RGB movies ))
![Alt text](/GUIscreenshot.PNG?raw=true "gui screenshot")

# supported movie files
extensions '.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf' (add more in line 60 of eyeGUI.m)

# default starting folder
**set at line 59 of eyeGUI.m (handles.filepath)**

# default folder to write binary files
**set at line 63 of eyeGUI.m (handles.binfolder)**

in GUI, click "choose folder for binary file" to change the location

This should be set to a location on your SSD for fast read/write speeds.


# folder loading structure
Choose a folder (say M012/2016-09-23) and it will assemble a list of all video files in that folder and 1 folder down (e.g. M012/2016-09-23/1/mov.mkv, M012/2016-09-23/2/mov.mkv, M012/2016-09-23/3/mov.mkv). You can choose which of these you want to process. You'll then see the ones that you chose in the drop down menu labelled by their folder names (1,2,3). You can switch between them and inspect how well the ROI works for each of the movies.

Or if M012/2016-09-23 has 3 movie files (e.g. M012/2016-09-23/mov1.mkv, M012/2016-09-23/mov2.mkv, M012/2016-09-23/mov3.mkv), then they show up in the drop down menu as mov1.mkv, mov2.mkv, mov3.mkv.

Next, you can choose among all the movies in all the folders and one folder down from the root folder that you chose.

When you choose ROIs these will be used to process ALL the movies that you see in the drop down menu when you click "Process ROIs".

# processing
you can choose which ROIs to process with the checkboxes on the right (if you've drawn the ROIs!)

# batch processing (multiple recordings, different ROI settings)

after choosing ROIs for a set of movies (seen in drop-down), click "save ROI and processing settings". load the next set of files and save settings. Then choose "Batch Process ROIs." A list of any groups of files you've saved to since opening the GUI shows up. The groups of files are labelled by the last (alphabetically) movie folder\file in the set of files.

If you want to process any movies which already have saved folders, you can use the script "BatchProcess_standalone.m". This loads the previously saved settings (using the same db structure as Suite2P) and processes the ROIs with those settings. You can change the processing settings as well. I'm showing an example in that file where I change the temporal smoothing constant for the motion ROIs.

# pupil computation

Use the saturation bar to reduce the background of the eye. The algorithm zeros out any pixels less than the saturation level. Next it finds the pixel with the largest magnitude. It draws a box around that area (1/2 the size of the ROI) and then finds the center-of-mass of that region. It then centers the box on that area. It fits a multivariate gaussian to the pixels in the box using maximum likelihood. The ellipse is then drawn at "sigma" standard deviations around the center-of-mass of the gaussian (default "sigma" = 4, but this can be changed in the GUI).

# different statistics of movement
motion: absolute value of the difference of two frames and sum over pixels greater than the threshold set by the GUI 
	
	motpix = abs(frames(:,t) - frames(:,t-1)); % this is pixels by time
	motion = sum(motpix>=saturation); % this is time by 1

motion SVD: take the svd of the motpix: 
	
	[u s v] = svd(motpix);
	proc.xx.motionMask = u;
	motionSVD = v(:,1:100);

movieSVD: take the svd of the frames:
	
	[u s v] = svd(frames);
	proc.xx.movieMask = u;
	movieSVD = v(:,1:100);
                  
# output of processing
creates a separate mat file for each video (saved in folder with video), mat file has name "videofile_proc.mat"

proc:
	
	suffix: '.mj2'     
        files: {1x3 cell} <--- all the files you processed together 
       folders: {3x1 cell}
      filename: '\\zserver.ioo.uc…' <--- filename of movie
    fitellipse: [0 1]          
          data: [1x1 struct]
      avgframe: [ypix x xpix int16]
      avgmotion: [ypix x xpix int16]

proc.data structure

for all ROIs:

	proc.data.pupil.ROI = [x y Lx Ly]
	proc.data.pupil.saturation = saturation value set by user
	proc.data.pupil.ROIX = x-1 + [1:Lx];
	proc.data.pupil.ROIY = y-1 + [1:Ly];
	proc.data.pupil.nX   = Lx;
	proc.data.pupil.nY   = Ly;

for pupil ROI:

	proc.data.pupil.area   = area of fit ellipse
	proc.data.pupil.com    = center of mass of ellipse (using pixel values)

for blink ROI:

	proc.data.blink.area   = sum of pixels greater than threshold set
	
for whisker, face, etc (see above for description of fields):

	proc.data.whisker.motion
	proc.data.whisker.motionSVD
	proc.data.whisker.movieSVD
	proc.data.whisker.motionMask
	proc.data.whisker.movieMask


