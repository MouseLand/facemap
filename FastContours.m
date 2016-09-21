% Michael Krumin's contour strategy
function [xx,yy] = FastContours(fr,thres)

[C] = contourc(fr, [.9 .9]);
CC = getContours(C);
% picking the most central contour in the ROI, usually works better
distance = nan(length(CC), 1);
roiX = size(fr, 2)/2;
roiY = size(fr, 1)/2;
for i=1:length(CC)
    distance(i) = norm([mean(CC(i).xx)-roiX, mean(CC(i).yy)-roiY]);
end
iMin = find(distance==min(distance), 1, 'first');
if ~isempty(CC)
    xx = CC(iMin).xx;
    yy = CC(iMin).yy;
else
    xx = [];
    yy = [];
end
