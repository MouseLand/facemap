function handles = ResetROIs(handles)
for j = 1:6
    handles.ROI{j} = [handles.nX/4 handles.nY/4 handles.nX/4 handles.nY/4];
    if j == 6
        handles.ROI{j} = [1 1 handles.nX handles.nY];
    end
    ROI = handles.ROI{j};
    handles.rX{j}   = floor(ROI(1)-1+[1:ROI(3)]);
    handles.rY{j}   = floor(ROI(2)-1+[1:ROI(4)]);
end
handles.plotROIs  = false(6,1);
handles.lastROI   = false(6,1);
