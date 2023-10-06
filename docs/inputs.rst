Inputs
=============================

Facemap supports grayscale and RGB movies. The software can process multi-camera videos for pose tracking and SVD analysis. 
Movie file extensions supported include:

'.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf'

Here are some `example movies <https://drive.google.com/open?id=1cRWCDl8jxWToz50dCX1Op-dHcAC-ttto>`__.

Processing multiple movies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please note:
   - simultaneous videos: user can load videos of different dimensions but must have the number of frames for each video
   - sequential videos: user can load videos with varying number of frames but the dimensions of the frames/videos must match

If you load multiple videos, the GUI will ask *"are you processing multiple videos taken simultaneously?"*. If you say yes, then the script will look if across movies the **FIRST FOUR** letters of the filename vary. If the first four letters of two movies are the same, then the GUI assumed that they were acquired *sequentially* not *simultaneously*.

Example file list:
+ cam1_G7c1_1.avi
+ cam1_G7c1_2.avi
+ cam2_G7c1_1.avi
+ cam2_G7c1_2.avi
+ cam3_G7c1_1.avi
+ cam3_G7c1_2.avi

*"are you processing multiple videos taken simultaneously?"* ANSWER: Yes

Then the GUIs assume {cam1_G7c1_1.avi, cam2_G7c1_1.avi, cam3_G7c1_1.avi} were acquired simultaneously and {cam1_G7c1_2.avi, cam2_G7c1_2.avi, cam3_G7c1_2.avi} were acquired simultaneously. They will be processed in alphabetical order (1 before 2) and the results from the videos will be concatenated in time. If one of these files was missing, then the GUI will error and you will have to choose file folders again. Also you will get errors if the files acquired at the same time aren't the same frame length (e.g. {cam1_G7c1_1.avi, cam2_G7c1_1.avi, cam3_G7c1_1.avi} should all have the same number of frames).

Note: if you have many simultaneous videos / overall pixels (e.g. 2000 x 2000) you will need around 32GB of RAM to compute the full SVD motion masks.

You will be able to see all the videos that were simultaneously collected at once. However, you can only draw ROIs that are within ONE video. Only the "multivideo SVD" is computed over all videos.

.. figure:: https://github.com/MouseLand/facemap/blob/main/figs/multivideo_fast.gif?raw=true
   :alt: example GUI with pupil, blink and motion SVD

Batch processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load a video or a set of videos and draw your ROIs and choose your processing settings. Then click "save ROIs". This will save a `\*_proc.npy` file in the output folder. Default output folder is the same folder as the video. Use file menu to change path of the output folder. The name of saved proc file will be listed below "process batch" (this button will also activate). You can then repeat this process: load the video(s), draw ROIs, choose settings, and click "save ROIs". Then to process all the listed `\*_proc.npy` files click "process batch".

Data acquisition info
~~~~~~~~~~~~~~~~~~~~~~~~~

IR illumination
---------------------

For recording in darkness we use `IR
illumination <https://www.amazon.com/Logisaf-Invisible-Infrared-Security-Cameras/dp/B01MQW8K7Z/ref=sr_1_12?s=security-surveillance&ie=UTF8&qid=1505507302&sr=1-12&keywords=ir+light>`__
at 850nm, which works well with 2p imaging at 970nm and even 920nm.
Depending on your needs, you might want to choose a different
wavelength, which changes all the filters below as well. 950nm works
just as well, and probably so does 750nm, which still outside of the
visible range for rodents.

If you want to focus the illumination on the mouse eye or face, you will
need a different, more expensive system. Here is an example, courtesy of
Michael Krumin from the Carandini lab:
`driver <https://www.thorlabs.com/thorproduct.cfm?partnumber=LEDD1B>`__,
`power
supply <https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=1710&pn=KPS101#8865>`__,
`LED <https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=2692&pn=M850L3#4426>`__,
`lens <https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=259&pn=AC254-030-B#2231>`__,
and `lens
tube <https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=4109&pn=SM1V10#3389>`__,
and another `lens
tube <https://www.thorlabs.com/thorproduct.cfm?partnumber=SM1L10>`__.

Cameras
---------------------

We use `ptgrey
cameras <https://www.ptgrey.com/flea3-13-mp-mono-usb3-vision-vita-1300-camera>`__.
The software we use for simultaneous acquisition from multiple cameras
is `BIAS <http://public.iorodeo.com/notes/bias/>`__ software. A basic
lens that works for zoomed out views
`here <https://www.bhphotovideo.com/c/product/414195-REG/Tamron_12VM412ASIR_12VM412ASIR_1_2_4_12_F_1_2.html>`__.
To see the pupil well you might need a better zoom lens `10x
here <https://www.edmundoptics.com/imaging-lenses/zoom-lenses/10x-13-130mm-fl-c-mount-close-focus-zoom-lens/#specs>`__.

For 2p imaging, you’ll need a tighter filter around 850nm so you don’t
see the laser shining through the mouse’s eye/head, for example
`this <https://www.thorlabs.de/thorproduct.cfm?partnumber=FB850-40>`__.
Depending on your lenses you’ll need to figure out the right adapter(s)
for such a filter. For our 10x lens above, you might need all of these:
`adapter1 <https://www.edmundoptics.com/optics/optical-filters/optical-filter-accessories/M52-to-M46-Filter-Thread-Adapter/>`__,
`adapter2 <https://www.thorlabs.de/thorproduct.cfm?partnumber=SM2A53>`__,
`adapter3 <https://www.thorlabs.de/thorproduct.cfm?partnumber=SM2A6>`__,
`adapter4 <https://www.thorlabs.de/thorproduct.cfm?partnumber=SM1L03>`__.

