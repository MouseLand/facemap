function h = resetROIs(h)

h.ROI{h.whichfile} = [];
h.eROI{h.whichfile} = [];

h.ROI{h.whichfile}{1} = [];
h.eROI{h.whichfile}{1} = [];
