# FaceMap
Matlab GUI for processing videos of rodents. Works for grayscale and RGB movies. Can process multi-camera videos. Some example movies to test the GUI on are located [here](https://drive.google.com/drive/folders/1fOkIXyEsxO-lDGZLy0gCKf1d7OjnUcnQ?usp=sharing). The original FaceMap GUI which only works on single-view movies and writes a binary file for processing (only efficient if movies are compressed) is [here](https://github.com/carsen-stringer/FaceMapOriginal).

This GUI works in Matlab 2014b and above - please submit issues if it's not working. The Image Processing Toolbox is necessary to use the GUI. For GPU functionality, the Parallel Processing Toolbox is required. If you don't have the Parallel Processing Toolbox, uncheck the box next to "use GPU" in the GUI before processing.

### Data acquisition info

We use [ptgrey cameras](https://www.ptgrey.com/flea3-13-mp-mono-usb3-vision-vita-1300-camera). The software we use for simultaneous acquisition from multiple cameras is [BIAS](http://public.iorodeo.com/notes/bias/) software. 

For recording in darkness we use [IR illumination](https://www.amazon.com/Logisaf-Invisible-Infrared-Security-Cameras/dp/B01MQW8K7Z/ref=sr_1_12?s=security-surveillance&ie=UTF8&qid=1505507302&sr=1-12&keywords=ir+light)

A basic lens that works for zoomed out views [here](https://www.bhphotovideo.com/c/product/414195-REG/Tamron_12VM412ASIR_12VM412ASIR_1_2_4_12_F_1_2.html). To see the pupil well you might need a better zoom lens [10x here](https://www.edmundoptics.com/imaging-lenses/zoom-lenses/10x-13-130mm-fl-c-mount-close-focus-zoom-lens/#specs).

For 2p imaging, you'll need a tighter filter around 850nm so you don't see the laser shining through the mouse's eye/head, for example [this](https://www.thorlabs.de/thorproduct.cfm?partnumber=FB850-40). Depending on your lenses you'll need to figure out the right adapter(s) for such a filter. For our 10x lens above, you might need all of these: 
https://www.edmundoptics.com/optics/optical-filters/optical-filter-accessories/M52-to-M46-Filter-Thread-Adapter/ ;
https://www.thorlabs.de/thorproduct.cfm?partnumber=SM2A53 ;
https://www.thorlabs.de/thorproduct.cfm?partnumber=SM2A6 ;
https://www.thorlabs.de/thorproduct.cfm?partnumber=SM1L03.


### Start processing!
To start the GUI, run the command `MovieGUI` in this folder. The following window should appear. After you click an ROI button and draw an area, you have to **double-click** inside the drawn box to confirm it.

![GUI screenshot](/figs/GUIscreenshot.png?raw=true "gui screenshot")

### Supported movie files
extensions '.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf' (add more in line 60 of MovieGUI.m)

### Default starting folder
set at line 59 of MovieGUI.m (h.filepath)

## File loading structure
Choose a folder and it will assemble a list of all video files in that folder and 1 folder down. The GUI will ask *"would you like to process all movies?"*. If you say no, then a list of movies to choose from will appear. 

### Processing movies captured simultaneously (multiple camera setups)

The GUI will then ask *"are you processing multiple videos taken simultaneously?"*. If you say yes, then the script will look if across movies the **FIRST FOUR** letters of the filename vary. If the first four letters of two movies are the same, then the GUI assumed that they were acquired *sequentially* not *simultaneously*.

Example file list:
+ cam1_G7c1_1.avi
+ cam1_G7c1_2.avi
+ cam2_G7c1_1.avi
+ cam2_G7c1_2.avi
+ cam3_G7c1_1.avi
+ cam3_G7c1_2.avi

*"are you processing multiple videos taken simultaneously?"* ANSWER: Yes

Then the GUI assumes {cam1_G7c1_1.avi, cam2_G7c1_1.avi, cam3_G7c1_1.avi} were acquired simultaneously and {cam1_G7c1_2.avi, cam2_G7c1_2.avi, cam3_G7c1_2.avi} were acquired simultaneously. They will be processed in alphabetical order (1 before 2) and the results from the videos will be concatenated in time. If one of these files was missing, then the GUI will error and you will have to choose file folders again.

After the file choosing process is over, you will see all the movies in the drop down menu (by filename). You can switch between them and inspect how well an ROI works for each of the movies.

## Settings and processing

The multivideo motion SVD and the small ROIs 1-3 are computed on the movie downsampled in space by the spatial downsampling input box in the GUI (default 4 pixels).

### Multivideo motion SVD

Draw areas to be included and excluded in the multivideo SVD (or single video if you only have one view). The buttons are "area to keep" and "area to exclude" and will draw blue and red boxes respectively. The union of all pixels in "areas to include" are used, excluding any pixels that intersect this union from "areas to exclude" (you can toggle between viewing the boxes and viewing the included pixels using the "Show areas" checkbox, see example below). 

<img src="figs/incexcareas.png" width="60%" alt="example areas">

The motion energy is computed from these non-red pixels: abs(current_frame - previous_frame), and the average motion energy across frames is computed using a subset of frames (*avgmot*) (4000 - set at line 45 in [subsampledMean.m](subsampledMean.m)). Then the singular vectors of the motion energy are computed on chunks of data, also from a subset of frames (50 chunks of 1000 frames each). Let *F* be the chunk of frames [pixels x time]. Then
```
uMot = [];
for j = 1:nchunks
  M = abs(diff(F,1,2));
  M = M - avgmot;
  [u,~,~] = svd(M);
  uMot = cat(2, uMot, u);
end
[uMot,~,~] = svd(uMot);
uMotMask = normc(uMot);
```
*uMotMask* are the motion masks that are then projected onto the video at all timepoints (done in chunks of size *nt*=2000):
```
for j = 1:nchunks
  M = abs(diff(F,1,2));
  M = M - avgmot;
  motSVD0 = M' * uMotMask;
  motSVD((j-1)*nt + [1:nt],:) = motSVD0;
end
```
Example motion masks *uMotMask* and traces *motSVD*:

<img src="figs/exsvds.png" width="50%" alt="example SVDs">

We found that these extracted singular vectors explained up to half of the total explainable variance in neural activity in visual cortex and in other forebrain areas. See our [preprint](https://www.biorxiv.org/content/early/2018/04/22/306019) for more details.

### Pupil computation

The minimum pixel value is subtracted from the ROI. Use the saturation bar to reduce the background of the eye. The algorithm zeros out any pixels less than the saturation level. Next it finds the pixel with the largest magnitude. It draws a box around that area (1/2 the size of the ROI) and then finds the center-of-mass of that region. It then centers the box on that area. It fits a multivariate gaussian to the pixels in the box using maximum likelihood (see [fitMVGaus.m](utils/fitMVGaus.m)). After a Gaussian is fit, it zeros out pixels whose squared distance from the center (normalized by the standard deviation of the Gaussian fit) is greater than 2 * sigma^2 where sigma is set by the user in the GUI. It now performs the fit again with these points erased, and repeats this process 4 more times. The pupil is then defined as an ellipse sigma standard deviations away from the center-of-mass of the gaussian (default sigma = 4).

This raw pupil area trace is post-processed (see [smoothPupil.m](pupil/smoothPupil.m))). The trace is median filtered with a window of 30 timeframes. At each timepoint, the difference between the raw trace and the median filtered trace is computed. If the difference at a given point exceeds half the standard deviation of the raw trace, then the raw value is replaced by the median filtered value.

![pupil](/figs/pupilfilter.png?raw=true "pupil filtering")

### Blink computation

You may want to ignore frames in which the animal is blinking if you are looking at pupil size. The blink area is the number of pixels above the saturation level (all non-white pixels).

### Small motion ROIs

The SVD of the motion is computed for each of the smaller motion ROIs. Motion is the abs(current_frame - previous_frame). The singular vectors are computed on subsets of the frames, and the top 500 components are kept.
                  
### Running computation

The phase-correlation between consecutive frames (in running ROI) are computed in the fourier domain (see [processRunning.m](/running/processRunning.m)). The XY position of maximal correlation gives the amount of shift between the two consecutive frames. 

## Output of processing

creates one mat file for all videos (saved in current folder), mat file has name "videofile_proc.mat"
- **nX**,**nY**: cell arrays of number of pixels in X and Y in each video taken simultaneously
- **sc**: spatial downsampling constant used
- **ROI**: [# of videos x # of areas] - areas to be included for multivideo SVD (in downsampled reference)
- **eROI**: [# of videos x # of areas] - areas to be excluded from multivideo SVD (in downsampled reference)
- **locROI**: location of small ROIs (in order running, ROI1, ROI2, ROI3, pupil1, pupil2); in downsampled reference
- **ROIfile**: in which movie is the small ROI
- **plotROIs**: which ROIs are being processed (these are the ones shown on the frame in the GUI)
- **files**: all the files you processed together 
- **npix**: array of number of pixels from each video used for multivideo SVD
- **tpix**: array of number of pixels in each view that was used for SVD processing
- **wpix**: cell array of which pixels were used from each video for multivideo SVD 
- **avgframe**: [sum(tpix) x 1] average frame across videos computed on a subset of frames
- **avgmotion**: [sum(tpix) x 1] average frame across videos computed on a subset of frames
- **motSVD**: cell array of motion SVDs [components x time] (in order: multivideo, ROI1, ROI2, ROI3)
- **uMotMask**: cell array of motion masks [pixels x time]
- **runSpeed**: 2D running speed computed using phase correlation [time x 2]
- **pupil**: structure of size 2 (pupil1 and pupil2) with 3 fields: area, area_raw, and com
- **thres**: pupil sigma used
- **saturation**: saturation levels (array in order running, ROI1, ROI2, ROI3, pupil1, pupil2); only saturation levels for pupil1 and pupil2 are used in the processing, others are just for viewing ROIs

an ROI is [1x4]: [y0 x0 Ly Lx]

### Motion SVD Masks

Use the script [plotSVDmasks.m](figs/plotSVDmasks.m) to easily view motion masks from the multivideo SVD. The motion masks from the smaller ROIs have been reshaped to be [xpixels x ypixels x components].

