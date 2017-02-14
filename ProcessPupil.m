% compute pupil area from set of frames
function [pup] = ProcessPupil(handles,frames)

% take chosen area
r.fr    = frames;
r.sats  = max(1, handles.saturation(1)*255);
    %roif(j).fmap  = roif(j).fr < roif(j).sats;

nframes    = size(frames,3);
pup.com    = [];
pup.center = [];
pup.area   = [];

r.thres = handles.thres(1);
for k = 1:nframes
    isplotting = (mod(k,1000)==0);
    if isplotting
        %PlotEye(handles);
    end
    
    params   = FindGaussianContour(r, k);
    pup.com  = [pup.com; params.mu(1) params.mu(2)];
    pup.area = [pup.area; params.area]; 
end