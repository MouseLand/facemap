% plot full face frame and selected ROI

function PlotEye(handles)

tstr = {'pupil','blink','whisker','groom','snout','face'};

% smoothing constants
sc = handles.sc;
tsc = handles.tsc;


axes(handles.axes1);
cla;
colormap('gray');
frame = read(handles.vr,handles.cframe);
jl = 1;
frames = zeros(handles.vr.Height, handles.vr.Width, 11, 'uint8');
for j = 1:11
    if handles.cframe+j-6 > 0 && handles.cframe+j-6<=handles.nframes
        frames(:,:,j) = read(handles.vr,handles.cframe+j-6);
    else
        frames(:,:,j) = read(handles.vr,handles.cframe);
    end
end
imagesc(frame)
title('');
axis tight;
axis off;

indROI = find(handles.plotROIs);
hold all;
if ~isempty(indROI)
    for j = indROI'
        rectangle('position',handles.ROI{j},'edgecolor',handles.colors(j,:),'linewidth',2);
    end
end

axes(handles.axesPupil)
cla;
indROI = find(handles.lastROI);
if ~isempty(indROI)
    colormap('gray')
    sat    = min(254,max(1,(handles.saturation(indROI))*255));
    fr     = single(frames(handles.rY{indROI}, handles.rX{indROI},:));
    
    % all ROIs besides pupil are down-sampled
    if indROI > 1
        [nY nX nt]  = size(fr);
        nYc = floor(nY/sc)*sc;
        nXc = floor(nX/sc)*sc;
        fr = squeeze(mean(reshape(fr(1:nYc,:,:),sc,nYc/sc,nX,nt),1));
        fr = squeeze(mean(reshape(fr(:,1:nXc,:),nYc/sc,sc,nXc/sc,nt),2));
    end
    
    if indROI == 1
        fr = my_conv2(fr, [1 1 tsc], [1 2 3]);
        fr = fr(:,:,6);
    end
    if indROI == 2
        fr = fr(:,:,6);
    end
    
    % pupil and eye area contours
    if indROI < 3
        imagesc(fr, [0 255-sat]);
        if indROI == 1
            r.fr     = fr;
            r.sats   = sat;
            r.thres  = handles.thres(indROI);
            tpt      = 1;
            params    = FindGaussianContour(r,tpt);
            
            if params.isgood
                hold all;
                plot(params.xy(:,1),params.xy(:,2),'r.')
                plot(params.mu(1), params.mu(2), 'k*');
            end
        end
    end
    
    % show difference between frames for movement areas
    if indROI > 2
        fr     = my_conv2(fr, tsc, 3);
        %keyboard;
        tdiff  = abs(fr(:,:,6)-fr(:,:,5));
        tdiff  = round(tdiff * tsc); 
        
        tdiff  = max(0, 5 - tdiff);
        sat    = min(4.99, max(0.01,(1-handles.saturation(indROI))*5));
        
        imagesc(tdiff,[0 sat]);
        %keyboard;
    end
    title(tstr{indROI},'fontsize',10);
    
    axis off;
    axis tight;
end
drawnow;
