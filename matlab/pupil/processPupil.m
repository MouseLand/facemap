% compute pupil area from set of frames
function [pup] = processPupil(frames,sats,thres)

sats  = min(254,max(1, sats*255));
r.fr = frames;
r.sats = sats;
r.thres = thres;
params   = findGaussianContour(r);
pup.com  = params.mu;
pup.area = params.area;
