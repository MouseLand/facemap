% constrain drawn ROI to size of figure
function roi0 = onScreenROI(ROI,nX,nY)
roi0 = ROI;
roi0(1) = min(nX,max(1,ROI(1)));
roi0(2) = min(nY,max(1,ROI(2)));
roi0(3) = min(nX-roi0(1),ROI(3));
roi0(4) = min(nY-roi0(2),ROI(4));
