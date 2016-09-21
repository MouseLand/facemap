function [pup,blink] = ProcessPupilBlink(handles,frames,wroi,roif)

% take chosen area
for j = wroi
    roif(j).fr = frames(roif(j).rX,roif(j).rY,:);
    [nX1 nY1 nframes] = size(roif(j).fr);
    roif(j).fmap = roif(j).fr < roif(j).sats;
end

pup(nframes) = struct();
blink(nframes) = struct();

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
            pup(k).isgood = params.isgood;
            pup(k).ra = params.ra;
            pup(k).rb = params.rb;
            pup(k).ang = params.ang;
            pup(k).center = [params.xc params.yc];
            pup(k).com = params.com;
            pup(k).area = pi*params.ra*params.rb;
            pup(k).x = params.xc;
            pup(k).y = params.yc;
        else
            %    blink2(k).area = params0.ra*params0.rb*pi;
            %             blink(k).isgood = params.isgood;
            %             blink(k).ra = params.ra;
            %             blink(k).rb = params.rb;
            %             blink(k).ang = params.ang;
            %             blink(k).center = [params.xc params.yc];
            %             blink(k).com = params.com;
            %             blink(k).area = pi*params.ra*params.rb;
            blink(k).area = sum(sum(roif(j).fmap(:,:,k),1),2);
            
        end
          
    end
    handles.cframe = handles.cframe+1;
end
handles.cframe = cframe0;