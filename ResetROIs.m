function h = ResetROIs(h)

h.ROI{h.whichfile} = [];
h.eROI{h.whichfile} = [];

h.ROI{h.whichfile}{1} = [];
h.eROI{h.whichfile}{1} = [];
