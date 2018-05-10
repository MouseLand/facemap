function newPos = drawROI(handles,ROI)

h = imrect(handles.axes1, ROI);
title(handles.axes1, 'Update the ROI and double-click')

%addNewPositionCallback(h,@(p) title(mat2str(p,3)));
%fcn = makeConstrainToRectFcn('imrect',get(gca,'XLim'),get(gca,'YLim'));
%setPositionConstraintFcn(h,fcn);

newPos = wait(h);
delete(h);
