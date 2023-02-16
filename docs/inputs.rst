Inputs
~~~~~~~~~~~~~~~~~~~

Processing multiple movies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Batch processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load a video or a set of videos and draw your ROIs and choose your processing settings. 
Then click `save ROIs`. This will save a *_proc.npy file in the output folder. 
Default output folder is the same folder as the video. Use file menu to change path of the output folder. The name of saved proc file will be listed below `process batch` (this button will also activate). You can then repeat this process: load the video(s), draw ROIs, choose settings, and click `save ROIs`. Then to process all the listed *_proc.npy files click `process batch`.

