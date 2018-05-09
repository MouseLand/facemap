% compute pupil area from set of frames
function [pup] = processPupil(h,frames,sats)

sats  = 255 - max(1, sats*255);

params   = getRadius(h, frames, sats);
pup.com  = params.mu;
pup.area = params.area;
