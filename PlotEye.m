function PlotEye(handles)

tstr = {'pupil','blink','whisker','groom','snout','face'};

axes(handles.axes1);
cla;
colormap('gray');
frame = read(handles.vr,handles.cframe);
imagesc(frame)
title(sprintf('frame %d',handles.cframe),'fontsize',10);
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
    sat    = max(1,(1-handles.saturation(indROI))*255);
    fr     = my_conv2(single(frame(handles.rX{indROI},handles.rY{indROI})),[2 2],[1 2]);
    
    % pupil and eye area contours
    imagesc(fr,[0 sat]);
    if indROI < 2
        r        = handles.roif(indROI);
        r.fr     = fr;
        r.sats   = (1-handles.saturation(indROI))*255;
        tpt      = 1;
        thres = [0.85 .95];
        r.thres = thres(indROI);
        [params] = FindEllipseandContour(r,tpt);
        
        if params.isgood
            hold all;
            ellipse(params.rb,params.ra,pi-params.ang,params.yc,params.xc,[1 0 0],300,2);
        end
    end
    
    % show difference between frames for movement areas
    if indROI > 2
        frame2 = read(handles.vr,handles.cframe+1);
        tdiff = abs(single(frame2) - single(frame));
        tdiff = tdiff(handles.rX{indROI},handles.rY{indROI});
        imagesc(tdiff,[0 sat]);
    end
    title(tstr{indROI},'fontsize',10);
    
    axis off;
    axis tight;
end
drawnow;
    