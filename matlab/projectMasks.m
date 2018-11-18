function h = projectMasks(h)
%%
ncomps = size(h.uMotMask{1},2);
%
npix = h.npix;
tpix = h.tpix;
nframes = h.nframes;
np = cumsum([0 h.npix]);
tp = cumsum([0 h.tpix]);


imendA = zeros(sum(npix),1,'single');
motSVD{1} = zeros(sum(nframes),ncomps,'single');

zf = find(h.plotROIs(2:end-4)) + 1;
for z = zf(:)'
    imend{z} = zeros(sum(h.spix{z}(:)),1,'single');
    motSVD{z} = zeros(sum(nframes),size(h.uMotMask{z},2),'single');
end
if h.plotROIs(1)
    imend{1} = zeros(sum(h.spix{1}(:)),1,'single');
end

ifr = 0;
fprintf('computing time traces\n');

[nviews,nvids] = size(h.vr);
for k = 1:nviews
    for j = 1:nvids
        h.vr{k,j}.CurrentTime = 0;
    end
end

nt0 = min(1000, min(nframes));
nsegs = ceil(sum(nframes)/nt0);
nframetimes = cumsum([0; nframes]);

wvids = [];
for k = 1:nviews
    kmotion = h.avgmotion(tp(k) + [1:tpix(k)]);
    avgmotion{k} = kmotion(h.wpix{k});
    sroi{k} = find(h.ROIfile(2:end-4)==k) + 1;
    
    if sum(h.ROIfile==k)>0 || sum(h.wpix{k}(:)) > 0
        wvids = cat(1, wvids, k);
    end
end

if h.plotROIs(1)
    h = runningFilters(h);
    h.runSpeed = zeros(sum(nframes), 2, 'single');
end
for j = numel(h.plotROIs)-3:numel(h.plotROIs)-2
    if h.plotROIs(j)
        h.pupil(j+4-numel(h.plotROIs)).area = zeros(sum(nframes),1,'single');
        h.pupil(j+4-numel(h.plotROIs)).com  = zeros(sum(nframes),2,'single');
    end
end
for j = numel(h.plotROIs)-1:numel(h.plotROIs)
    if h.plotROIs(j)
        h.blink(j+2-numel(h.plotROIs)).area = zeros(sum(nframes),1,'single');
    end
end

j=0;
ivid = 0;
while ivid <= nvids && ifr < sum(nframes)
	j=j+1;
    tc = ifr;
    % which video is tc in
    ivid = find(tc<nframetimes(2:end) & tc>=nframetimes(1:end-1));
    
    for k = wvids'
        nx = h.nX{k};
        ny = h.nY{k};
        imb = zeros(ny, nx,nt0,'uint8');
        nt = 0;
        for t = 1:nt0
            if h.vr{k,ivid}.hasFrame
                im = h.vr{k,ivid}.readFrame;
                imb(:,:,t) = im(:,:,1);
                nt = nt + 1;
            end
		end
		if ifr+nt > sum(nframes)
			nt = sum(nframes)-ifr;
		end
        if nt < nt0
            imb = imb(:,:,1:nt);
			ivid = ivid+1;
		end
        imb = reshape(single(imb), [], nt);
        
		% if pupil, blink or running ROI, take out first
		% (computed on non-downsampled frames)
		
		% running computation
        z = h.ROIfile(1) == k;
        if z
            ims = imb(h.spix{1}(:),:);
            if j == 1
                imend{1} = ims(:,1);
            end
            ims = cat(2, imend{1}, ims);
            ims = reshape(ims, h.iroi{1}(4), h.iroi{1}(3), nt+1);
            DS = processRunning(h, ims);
            h.runSpeed(ifr + [1:nt] ,:) = DS;
            imend{1} = reshape(ims(:,:,end),[],1);
		end
		
		% pupil computation
        pups = h.ROIfile(end-3:end-2) == k;
        for l = 1:2
            if pups(l)
                z = pups(l) + (numel(h.ROIfile)-4);
                ims = imb(h.spix{z}(:),:);
                ims = reshape(ims, h.iroi{z}(4), h.iroi{z}(3), nt);
                ims = my_conv2(ims, [1 1 1], [1 2 3]);
                ims = ims - min(min(ims,[],1),[],2);
                h.indROI = z;
                pupil = processPupil(ims, h.saturation(z), h.thres);
                h.pupil(l).area(ifr + [1:nt]) = pupil.area;
                h.pupil(l).com(ifr + [1:nt],:) = pupil.com;
            end
		end
		
		% blink computation (sum of pixels greater than threshold)
		blinks = h.ROIfile(end-1:end) == k;
        for l = 1:2
            if blinks(l)
                z = blinks(l) + (numel(h.ROIfile)-2);
                ims = imb(h.spix{z}(:),:);
                h.indROI = z;
				sats  = min(254,max(1, (1-h.saturation(z))*255));
                barea = sum(ims<sats);
                h.blink(l).area(ifr + [1:nt]) = barea;
            end
		end
		
        % spatial downsampling
        ns = h.sc;
        imb = reshape(imb, ny, nx, nt);
        imd = squeeze(mean(mean(reshape(imb(1:floor(ny/ns)*ns,1:floor(nx/ns)*ns,:),...
            ns, floor(ny/ns), ns, floor(nx/ns), nt), 1),3));
        
		
        % motion SVD projection for full ROIs
        imd = reshape(imd,[],nt)';
        ima = imd(:,h.wpix{k}(:))';
        imdiff = abs(diff(cat(2,imendA(np(k) + [1:npix(k)]),ima),1,2));
        if j==1
            imdiff(:,1) = imdiff(:,2);
        end
        imdiff = bsxfun(@minus, imdiff, avgmotion{k});
        motSVD{1}(ifr+[1:nt],:)  = motSVD{1}(ifr+[1:nt],:) + ...
            gather_try(imdiff' * h.uMotMask{1}(np(k) + [1:npix(k)],:));
        imendA(np(k) + [1:npix(k)]) = ima(:,end);
        
        % motion SVD projection for small ROIs
        for m = 1:numel(sroi{k})
            z = sroi{k}(m);
            ima = imd(:,h.spix{z}(:))';
            imdiff = abs(diff(cat(2,imend{z},ima),1,2));
            if j==1
                imdiff(:,1) = imdiff(:,2);
            end
            imdiff = bsxfun(@minus, imdiff, h.avgmot{z});
            motSVD{z}(ifr+[1:nt],:) = imdiff' * h.uMotMask{z};
            imend{z} = ima(:,end);
        end
    end
    
    ifr = ifr + nt;
    
    fprintf('%d / %d done in %2.2f\n',j, nsegs,toc);
    
end

h.motSVD = motSVD;

