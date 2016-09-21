function newPos = DrawROI(handles,ROI,rcol)

h = imrect(handles.axes1, ROI);
title(handles.axes1, 'Update the ROI and double-click')
newPos = wait(h);
delete(h);
