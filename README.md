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
                  
# output of GUI
creates a separate mat file for each video
proc:
        suffix: '.mj2'     
        files: {1x3 cell} <--- all the files you processed together 
       folders: {3x1 cell}
      filename: '\\zserver.ioo.uc…' <--- file name of movie
    fitellipse: [0 1]          
          data: [1x1 struct]
      avgframe: [480x640 uint16]

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
	proc.data.pupil.center = center of fit ellipse
	proc.data.pupil.com    = center of mass of ellipse (using pixel values)

for blink ROI:

	proc.data.blink.area   = sum of pixels greater than threshold set
	
for whisker, face, etc:

	proc.data.whisker.motion
	proc.data.whisker.motionSVD
	proc.data.whisker.movieSVD
	proc.data.whisker.motionMask
	proc.data.whisker.movieMask


   (see above for description)
          

