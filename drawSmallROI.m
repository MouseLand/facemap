function h = drawSmallROI(h)
h.plotROIs(h.indROI) = 1;

nxS = floor(h.nX{h.whichview} / h.sc);
nyS = floor(h.nY{h.whichview} / h.sc);

h.ROIfile(h.indROI) = h.whichview;
if isempty(h.locROI{h.indROI})
    ROI0 = [nxS*.25 nyS*.25 nxS*.5 nyS*.5];
else
    ROI0 = h.locROI{h.indROI};
end
ROI = DrawROI(h,ROI0);
ROI = OnScreenROI(ROI, nxS, nyS);
h.locROI{h.indROI} = ROI;
if h.indROI > 4
    h = getTemplates(h);
end
PlotFrame(h);
axes(h.axes4);
cla;
PlotROI(h);
sats = h.saturation(h.indROI);
set(h.slider2,'Value',sats);
set(h.edit1,'String',sprintf('%1.2f',sats));
