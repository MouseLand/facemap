function [pup,blink] = ProcessPupilBlink(handles,frames,wroi,roif)

% take chosen area
for j = wroi
    roif(j).fr = frames(roif(j).rX,roif(j).rY,:);
    [nX1 nY1 nframes] = size(roif(j).fr);
    roif(j).fmap = roif(j).fr < roif(j).sats;
end

pup.com    = [];
pup.center = [];
pup.area   = [];
blink.area = [];

handles.lastROI = false(6,1);
handles.lastROI(wroi(1)) = 1;
cframe0 = handles.cframe;

thres = [0.85 .95];
for k = 1:nframes
    isplotting = (mod(k,1000)==0);
    if isplotting
        %PlotEye(handles);
    end
        
    for j = wroi
        roif(j).thres = thres(j);
        if j==1
            [params] = FindEllipseandContour(roif(j),k);
            %pup2(k).area = params0.ra^2*pi;
            pup.center = [pup.center; params.xc params.yc];
            pup.com = [pup.com; params.com];
            pup.area = [pup.area; pi*params.ra*params.rb];
        else
            blink.area = [blink.area; sum(sum(roif(j).fmap(:,:,k),1),2)];
        end
          
    end
    handles.cframe = handles.cframe+1;
end
handles.cframe = cframe0;