# eyeGUI
matlab GUI for processing eye camera data from rodents

# folder loading structure
Choose a folder (say M012/2016-09-23) and it will add all video files in that folder and 1 folder down 

(e.g. M012/2016-09-23/1/mov.mkv, M012/2016-09-23/2/mov.mkv, M012/2016-09-23/3/mov.mkv) 

You'll see them in the drop down menu labelled as 1,2,3. 

Or if M012/2016-09-23 has 3 movie files 

(e.g. M012/2016-09-23/mov1.mkv, M012/2016-09-23/mov2.mkv, M012/2016-09-23/mov3.mkv) 

then they show up in the drop down menu as mov1.mkv, mov2.mkv, mov3.mkv

When you choose ROIs these will be used to process ALL the folders that you see in the drop down menu when you click "Process ROIs".


# different statistics of movement
motion: absolute value of the difference of two frames and sum over pixels greater than the threshold set by the GUI 

motpix = abs(frame(:,t) - frame(:,t-1)); % this is pixels by time

motion = sum(motpix>=saturation); % this is time by 1

motion SVD: take the svd of the motpix: [u s v] = svd(motpix)

use u (proc.xx.motionMask) and take 100 components of it

use 100 components of v: motionSVD = v(:,1:100);

movieSVD: take the svd of the frames: [u s v] = svd(frames);

u (proc.xx.movieMask)

use 100 components of v: movieSVD = v(:,1:100);
                  
# output of GUI
creates structure where each is a different movie file (jf is the file index)
for all ROIs:

proc(jf).pupil.ROI = [x y Lx Ly]

proc(jf).pupil.saturation = saturation value set by user

proc(jf).pupil.ROIX = x-1 + [1:Lx];

proc(jf).pupil.ROIY = y-1 + [1:Ly];

proc(jf).pupil.nX   = Lx;

proc(jf).pupil.nY   = Ly;

for pupil ROI:

proc(jf).pupil.area   = area of fit ellipse

proc(jf).pupil.center = center of fit ellipse

proc(jf).pupil.com    = center of mass of ellipse (using pixel values)

for blink ROI:

proc(jf).blink.area   = sum of pixels greater than threshold set

for whisker, face, etc:

proc(jf).whisker.motion

proc(jf).whisker.motionSVD

proc(jf).whisker.movieSVD

proc(jf).whisker.motionMask

proc(jf).whisker.movieMask


   (see above for description)
          

