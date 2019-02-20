% compute mean of videos on a subset of frames
% only use videos that are in multivideo SVD or ROI1-3
function h = subsampledMean(h)
nviews = numel(h.nX);
nvids  = size(h.vr,2);
nframes = zeros(nvids,1,'single');

for j = 1:nvids
    nframes(j) = h.vr{1,j}.Duration * h.vr{1,j}.FrameRate;
end

nframes = floor(nframes);

npix = [];
tpix = [];
wvids = [];
for k = 1:nviews
    for j = 1:nvids
        h.vr{k,j}.currentTime = 0;
    end
    nx = floor(h.nX{k}/h.sc);
    ny = floor(h.nY{k}/h.sc);
    h.wpix{k} = false(ny, nx);
    if ~isempty(h.ROI{k}{1})
        for j = 1:numel(h.ROI{k})
            pos = round(h.ROI{k}{j});
            h.wpix{k}(pos(2)-1 + [1:pos(4)], pos(1)-1 + [1:pos(3)]) = 1;
        end
        if ~isempty(h.eROI{k}{1})
            for j = 1:numel(h.eROI{k})
                pos = round(h.eROI{k}{j});
                h.wpix{k}(pos(2)-1 + [1:pos(4)], pos(1)-1 + [1:pos(3)]) = 0;
            end
        end
        tpix(k) = nx * ny;
    elseif sum(h.ROIfile(2:end-4)==k) > 0
        tpix(k) = nx * ny;
    else
        tpix(k) = 0;
    end
    if tpix(k) ~= 0
        wvids = cat(1,wvids,k);
    end
        
    npix(k) = sum(h.wpix{k}(:));
    
end
    
nf = min(4000,sum(nframes));
nt0 = min(100, min(nframes));
nsegs = nf/nt0;
tf = linspace(0,floor(double(sum(nframes)-(nt0))/h.vr{1}.FrameRate),nsegs);
nframetimes = double(cumsum([0; nframes])) / h.vr{1}.FrameRate;

tp = [0 tpix];
tp = cumsum(tp);

for j = 1:nsegs
    im0 = zeros(sum(tpix),nt0,'single');
    
    for k = wvids'
        tc = tf(j);
		% which video is tc in
		ivid = find(tc<nframetimes(2:end) & tc>=nframetimes(1:end-1));
		tcv = tc - nframetimes(ivid);
		if tcv > (nframes(ivid)-(nt0))/h.vr{1}.FrameRate
			ivid = ivid+1;
			tcv = 0;
			if ivid > nvids
				break;
			end
		end
        h.vr{k,ivid}.CurrentTime = tcv;
        
        for t = 1:nt0
            im = h.vr{k,ivid}.readFrame;
            [nx,ny,~] = size(im);
            ns = h.sc;
            im = squeeze(mean(mean(reshape(single(im(1:floor(nx/ns)*ns,1:floor(ny/ns)*ns,1)),...
                ns, floor(nx/ns), ns, floor(ny/ns)), 1),3));
            im0(tp(k) + [1:tpix(k)], t) = im(:);
        end
    end
    im0 = double(im0);
    
    if j==1
        avgframe = zeros(sum(tpix),1,'double');
        avgmotion = zeros(sum(tpix),1,'double');
    end
    avgframe  = avgframe + mean(im0, 2);
    avgmotion = avgmotion + mean(abs(diff(im0, 1, 2)), 2);
    
    if mod(j-1, 5) == 0
        fprintf('%d / %d done in %2.2f sec\n',j, nsegs,toc);
    end
    
end
avgframe = avgframe/double(nsegs);
avgmotion = avgmotion/double(nsegs);

h.nframes = nframes;
h.avgframe = single(avgframe);
h.avgmotion = single(avgmotion);
h.npix = npix;
h.tpix = tpix;
