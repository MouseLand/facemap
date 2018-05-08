# FaceMap
matlab GUI for processing videos of rodents
(( works for GRAYSCALE and RGB movies ))
![Alt text](/GUIscreenshot.PNG?raw=true "gui screenshot")

### Supported movie files
extensions '.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf' (add more in line 60 of MovieGUI.m)

### Default starting folder
set at line 59 of MovieGUI.m (handles.filepath)

## File loading structure
Choose a folder and it will assemble a list of all video files in that folder and 1 folder down. The GUI will ask *"would you like to process all movies?"*. If you say no, then a list of movies to choose from will appear. 

You will then see all the movies that you chose in the drop down menu (by filename). You can switch between them and inspect how well an ROI works for each of the movies.

### Processing movies captured simultaneously (multiple camera setups)

The GUI will ask *"are you processing multiple videos taken simultaneously?"* if across movies the **FIRST FOUR** letters of the filename vary. If the first four letters of two movies are the same, then the GUI assumed that they were acquired *sequentially* not *simultaneously*.

Example:
+ cam1_G7c1_1.avi
+ cam1_G7c1_2.avi
+ cam2_G7c1_1.avi
+ cam2_G7c1_2.avi
+ cam3_G7c1_1.avi
+ cam3_G7c1_2.avi

*"are you processing multiple videos taken simultaneously?"* ANSWER: Yes

Then the GUI assumes {cam1_G7c1_1.avi, cam2_G7c1_1.avi, cam3_G7c1_1.avi} were acquired simultaneously and {cam1_G7c1_2.avi, cam2_G7c1_2.avi, cam3_G7c1_2.avi} were acquired simultaneously. They will be processed in alphabetical order (1 before 2) and the results from the videos will be concatenated in time.

## Processing

The multivideo motion SVD and the small ROIs 1-3 are computed on the movie downsampled in space by the spatial downsampling input box in the GUI (default 4 pixels).

### Multivideo motion SVD

The union of all pixels in "areas to include" are used, excluding any pixels that intersect this union from "areas to exclude". The motion energy is computed from these pixels: abs(current_frame - previous_frame), and the average motion energy across frames is computed using a subset of frames (*avgmot*) (4000 - set at line 45 in subsampledMean.m). Then the singular vectors of the motion energy are computed on chunks of data, also from a subset of frames (50 chunks of 1000 frames each). Let *F* be the chunk of frames [pixels x time]. Then
```
uMot = [];
for j = 1:nchunks
  M = abs(diff(F,1,2));
  M = M - avgmot;
  [u,~,~] = svd(M);
  uMot = cat(2, uMot, u);
end
[uMot,~,~] = svd(uMot);
uMot = normc(uMot);
```

### Pupil computation

Use the saturation bar to reduce the background of the eye. All pixels below the saturation level in the ROI are set to the saturation level. Then template matching on the pupil area proceeds (see [getRadius.m](getRadius.m) and [getTemplates.m](getTemplates.m)). 101 different templates each with a different pupil radius are phase-correlated with the ROI in the FFT domain. The smoothness of the edges of the template are set by the pupil sigma parameter in the GUI. 2-3 is recommended for smaller pupils, and 4 for larger pupils when the animal is in darkness. 

The phase-correlation of the ROI with the 101 different templates of different radii produces 101 correlation values. This vector is upsampled 10 times using kriging interpolation with a Gaussian kernel of standard deviation of 1. The maximum of this vector is the radius of the pupil in pixels - then the area is pi* radius^2. The center-of-mass (com) of the pupil is the XY position that maximizes the phase-correlation of the best template.

This raw pupil area is post-processed (see [smoothPupil.m](smoothPupil.m))). The area is median filtered. Then all points...

### Small motion ROIs

The SVD of the motion is computed for each of the smaller motion ROIs. Motion is the abs(current_frame - previous_frame). The singular vectors are computed on subsets of the frames, and the top 500 components are kept.
                  
### Running computation


                  
## Output of processing

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
- **motSVD**: cell array of motion SVDs [components x time] (in order: multivideo, ROI1, ROI2, ROI3)
- **uMotMask**: cell array of motion masks [pixels x time]
- **running**: 2D running speed computed using phase correlation [time x 2]
- **pupil**: structure of size 2 (pupil1 and pupil2) with 3 fields: area, area_raw, and com

an ROI is [1x4]: [x0 y0 Lx Ly]
