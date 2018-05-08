# FaceMap
matlab GUI for processing videos of rodents
(( works for GRAYSCALE and RGB movies ))
![Alt text](/GUIscreenshot.PNG?raw=true "gui screenshot")

# supported movie files
extensions '.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf' (add more in line 60 of MovieGUI.m)

# default starting folder
**set at line 59 of eyeGUI.m (handles.filepath)**

# folder loading structure
Choose a folder (say M012) and it will assemble a list of all video files in that folder and 1 folder down. The GUI will ask "would you like to process all movies?". If you say no, then a list of movies to choose from will appear. 

After choosing the movies, you have to tell the GUI if these are multiple videos acquired simultaneously. It will ask "are you processing multiple videos taken simultaneously?". 

Or if M012/2016-09-23 has 3 movie files (e.g. M012/2016-09-23/mov1.mkv, M012/2016-09-23/mov2.mkv, M012/2016-09-23/mov3.mkv), then they show up in the drop down menu as mov1.mkv, mov2.mkv, mov3.mkv.

Next, you can choose among all the movies in all the folders and one folder down from the root folder that you chose.

When you choose ROIs these will be used to process ALL the movies that you see in the drop down menu when you click "Process ROIs".

You'll then see the ones that you chose in the drop down menu (by filename). You can switch between them and inspect how well an ROI works for each of the movies.

# pupil computation

Use the saturation bar to reduce the background of the eye. The algorithm zeros out any pixels less than the saturation level. Next it finds the pixel with the largest magnitude. It draws a box around that area (1/2 the size of the ROI) and then finds the center-of-mass of that region. It then centers the box on that area. It fits a multivariate gaussian to the pixels in the box using maximum likelihood. The ellipse is then drawn at "sigma" standard deviations around the center-of-mass of the gaussian (default "sigma" = 4, but this can be changed in the GUI).

### small motion ROIs

The SVD of the motion is computed for each of the smaller motion ROIs. Motion is the abs(current_frame - previous_frame). The singular vectors are computed on subsets of the frames, and the top 500 components are kept.
                  
## output of processing

creates one mat file for all videos (saved in current folder), mat file has name "videofile_proc.mat"
- **nX**,**nY**: cell arrays of number of pixels in X and Y in each video taken simultaneously
- **ROI**: [# of videos x # of areas] - areas to be included for multivideo SVD
- **eROI**: [# of videos x # of areas] - areas to be excluded from multivideo SVD
- **locROI**: location of small ROIs (in order pupil1, pupil2, ROI1, ROI2, ROI3, running)
- **ROIfile**: in which movie is the small ROI
- **plotROIs**: which ROIs are being processed (these are the ones shown on the frame in the GUI)
- **files**: all the files you processed together 
- **npix**: array of number of pixels from each video used for multivideo SVD
- **tpix**: array of number of pixels in each video used for processing
- **wpix**: cell array of which pixels were used from each video for multivideo SVD 
- **avgframe**: [sum(tpix) x 1] average frame across videos computed on a subset of frames
- **avgmotion**: [sum(tpix) x 1] average frame across videos computed on a subset of frames
- **motSVD**: cell array of motion SVDs [components x time] (in order ROI1, ROI2, ROI3, multivideoSVD)
- **uMotMask**: cell array of motion masks [pixels x time]
- **running**: 2D running speed computed using phase correlation [time x 2]
- **pupil**: structure of size 2 (pupil1 and pupil2) with 3 fields: area, area_raw, and com

an ROI is [1x4]: [x0 y0 Lx Ly]
