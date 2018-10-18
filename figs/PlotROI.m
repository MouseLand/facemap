% plot full face frame and selected ROI
function PlotROI(h)

nROIs = numel(h.plotROIs);

kc = h.whichfile;
k = h.whichview;
tstr = {'running','ROI1','ROI2','ROI3','pupil 1','pupil 2','blink1','blink2'};

% smoothing constants
sc = h.sc;

if ~isempty(h.indROI)
    indROI = h.indROI;
    
    axes(h.axes4);
    frames = zeros(h.vr{k}.Height, h.vr{k}.Width, 3, 'uint8');
    h.cframe = double(h.cframe);
    if h.cframe-1 < 2
        h.vr{kc}.currentTime = (h.cframe) / h.vr{kc}.FrameRate;
    elseif h.cframe > h.nframes - 2
        h.vr{kc}.currentTime = (h.cframe-2) / h.vr{kc}.FrameRate;
    else
        h.vr{kc}.currentTime = (h.cframe-1) / h.vr{kc}.FrameRate;
    end
    
    
    for j = 1:3
        fr = readFrame(h.vr{kc});
        if size(fr,3) > 1
            fr = mean(single(fr),3);
        end
        frames(:,:,j) = fr;
    end
    
    sat    = min(254,max(1,(h.saturation(indROI))*255));
    
    % only small ROIs are are down-sampled
    if indROI > 1 && indROI <= nROIs-4
        [nY nX nt]  = size(frames);
        nYc = floor(nY/sc)*sc;
        nXc = floor(nX/sc)*sc;
        fr  = squeeze(mean(reshape(frames(1:nYc,:,:),sc,nYc/sc,nX,nt),1));
        fr  = squeeze(mean(reshape(fr(:,1:nXc,:),nYc/sc,sc,nXc/sc,nt),2));
	% pupil ROIs are smoothed
    elseif indROI > nROIs-4 && indROI <= nROIs-2
        fr = my_conv2(single(frames), [1 1 1], [1 2 3]);
        fr = fr(:,:,2);
	elseif indROI==1
        fr = frames;
	else
		fr = frames(:,:,2);
    end
    
    if indROI==1 || indROI>4
        iroi = max(1,floor(h.locROI{indROI} * h.sc));
    else
        iroi = h.locROI{indROI};
    end
    iroi = round(iroi);
    
    fr = fr(iroi(2)-1 + [1:iroi(4)], iroi(1)-1 + [1:iroi(3)], :);
    
	
    % pupil contours
    if indROI>nROIs-4 && indROI<=nROIs-2
        fr = fr - min(fr(:));
        imagesc(fr,[0 255-sat]);
        
        r.fr     = fr;
        r.sats   = sat;
        r.thres  = h.thres;
        params   = findGaussianContour(r);
            
        if params.isgood
            hold all;
            plot(params.xy(:,1),params.xy(:,2),'r')
            plot(params.mu(1), params.mu(2), 'k*');
        end
        colormap(h.axes4, 'gray');
    % show difference between frames for movement areas
    elseif indROI<5 
        tdiff  = abs(fr(:,:,2)-fr(:,:,1));
        
        tdiff  = min(100, max(0, tdiff));
        sat    = 100 * (255 - sat)/254;
        %sat    = min(4.99, max(0.01,(1-sat)*5));
        
        imagesc(tdiff,[0 sat]);%,[0 sat]);
        rb = redblue;
        colormap(h.axes4, rb(33:end,:));
        %keyboard;
	else
		imagesc(fr,[0 255-sat])
		colormap(h.axes4,'gray')
    end
    title(tstr{indROI},'fontsize',10);
    
    axis off;
    axis([1 size(fr,2) 1 size(fr,1)]);
    
end
