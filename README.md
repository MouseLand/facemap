# FaceMap
matlab GUI for processing videos of rodents
(( works for GRAYSCALE and RGB movies ))
![Alt text](/GUIscreenshot.PNG?raw=true "gui screenshot")

## Supported movie files
extensions '.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf' (add more in line 60 of MovieGUI.m)

## Default starting folder
**set at line 59 of eyeGUI.m (handles.filepath)**

## File loading structure
Choose a folder (say M012) and it will assemble a list of all video files in that folder and 1 folder down. The GUI will ask "would you like to process all movies?". If you say no, then a list of movies to choose from will appear. 

After choosing the movies, you have to tell the GUI if these are multiple videos acquired simultaneously. It will ask "are you processing multiple videos taken simultaneously?". 

Or if M012/2016-09-23 has 3 movie files (e.g. M012/2016-09-23/mov1.mkv, M012/2016-09-23/mov2.mkv, M012/2016-09-23/mov3.mkv), then they show up in the drop down menu as mov1.mkv, mov2.mkv, mov3.mkv.

Next, you can choose among all the movies in all the folders and one folder down from the root folder that you chose.

When you choose ROIs these will be used to process ALL the movies that you see in the drop down menu when you click "Process ROIs".

You'll then see the ones that you chose in the drop down menu (by filename). You can switch between them and inspect how well an ROI works for each of the movies.

## Processing

The multivideo motion SVD and the small ROIs 1-3 are computed on the movie downsampled in space by the smoothing constant in the GUI (default 4 pixels).

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

Use the saturation bar to reduce the background of the eye. The algorithm zeros out any pixels less than the saturation level. Next it finds the pixel with the largest magnitude. It draws a box around that area (1/2 the size of the ROI) and then finds the center-of-mass of that region. It then centers the box on that area. It fits a multivariate gaussian to the pixels in the box using maximum likelihood. The ellipse is then drawn at "sigma" standard deviations around the center-of-mass of the gaussian (default "sigma" = 4, but this can be changed in the GUI).

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
