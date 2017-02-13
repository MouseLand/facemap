function [pup,blink] = ProcessPupilBlink(handles,frames,wroi)

% take chosen area
for j = wroi
    roif(j).fr    = frames;
    roif(j).sats  = (1-handles.saturation(j))*255;
    roif(j).fmap  = roif(j).fr < roif(j).sats;
end

nframes    = size(frames,3);
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
            %[params] = FindEllipseandContour(roif(j),k);
            %pup2(k).area = params0.ra^2*pi;
            %pup.center = [pup.center; params.xc params.yc];
            %pup.com = [pup.com; params.com];
            
            params   = FindGaussianContour(roif(j), k);
            pup.com  = [pup.com; params.mu(1) params.mu(2)];
            pup.area = [pup.area; params.area];
        else
            blink.area = [blink.area; sum(sum(roif(j).fmap(:,:,k),1),2)];
        end
          
    end
    handles.cframe = handles.cframe+1;
end
handles.cframe = cframe0;