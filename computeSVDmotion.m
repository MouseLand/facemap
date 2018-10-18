% load face file and compute SVD of motion and/or movie
function h = computeSVDmotion(h)
%%
np = cumsum([0 h.npix]);
tp = cumsum([0 h.tpix]);
npix = h.npix;
tpix = h.tpix;
nframes = h.nframes;
ntime = h.vr{1}.Duration;
[nviews, nvids] = size(h.vr);

wvids = [];
for k = 1:nviews
    kmotion = h.avgmotion(tp(k) + [1:tpix(k)]);
    avgmotion(np(k) + [1:npix(k)]) = kmotion(h.wpix{k});
    
    sroi{k} = find(h.ROIfile(2:end-4)==k) + 1;
    
    if ~isempty(sroi{k}) || sum(h.wpix{k}(:)) > 0 
        wvids = cat(1, wvids, k);
    end
end

nt0 = min(1000, min(nframes));
ncomps = 500;
ncompsSmall = ncomps;
fprintf('computing SVDs across all movies\n');
nsegs = min(floor(50000 / nt0), floor(double(sum(nframes))/nt0));

tf = linspace(0,floor(double(sum(nframes)-(nt0))/h.vr{1}.FrameRate),nsegs);
nframetimes = double(cumsum([0; nframes])) / h.vr{1}.FrameRate;

% first ROI is main ROI (all cameras)
ims{1} = zeros(sum(npix),ncomps,'single');
avgmot{1} = avgmotion(:);
uMot{1} = [];

% small motion SVD ROIs (excludes pupils and last ROI = running)
zf = find(h.plotROIs(2:end-4)) + 1;
for z = zf(:)'
    if h.plotROIs(z)
        k = h.ROIfile(z);
        nx = floor(h.nX{k}/h.sc);
        ny = floor(h.nY{k}/h.sc);
        h.spix{z} = false(ny, nx);
        pos = round(h.locROI{z});
        h.spix{z}(pos(2)-1 + [1:pos(4)], pos(1)-1 + [1:pos(3)]) = 1;
        
        ims{z} = zeros(sum(h.spix{z}(:)),ncomps,'single');
        
        k = h.ROIfile(z);
        kmotion = h.avgmotion(tp(k) + [1:tpix(k)]);
        avgmot{z} = kmotion(h.spix{z});
        uMot{z} = [];
    end
end
%%
for j = 1:nsegs
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
    
    for k = wvids'
        h.vr{k,ivid}.CurrentTime = tcv;
        for t = 1:nt0
            im = h.vr{k,ivid}.readFrame;
            im = im(:,:,1);
            [nx,ny] = size(im);
            ns = h.sc;
            im = squeeze(mean(mean(reshape(single(im(1:floor(nx/ns)*ns,1:floor(ny/ns)*ns)),...
                ns, floor(nx/ns), ns, floor(ny/ns)), 1),3));
            ims{1}(np(k) + [1:npix(k)], t) = im(h.wpix{k}(:));
            
            for m = 1:numel(sroi{k})
                z = sroi{k}(m);
                ims{z}(:,t) = im(h.spix{z}(:));
            end         
        end
    end
    
    for z = 1:numel(ims)
        if ~isempty(ims{z})
            imot = bsxfun(@minus, abs(diff(ims{z},1,2)), avgmot{z});
            if h.useGPU
                imot = gpuArray(imot);
            end
            [u s v] = svdecon(imot);
            u       = gather_try(u);
            uMot{z}    = cat(2, uMot{z}, u(:,1:min(200,size(u,2))));
        end
    end
    
    if mod(j-1,5) == 0
        fprintf('%d / %d done in %2.2f sec\n',j, nsegs, toc);
    end
end
clear imot;

for z = 1:numel(ims)
    if z > 1
        ncompsz = ncompsSmall;
    else
        ncompsz = ncomps;
    end
    [u s v]  = svdecon(uMot{z});
    uMotMask{z} = gather_try(u(:,1:min(ncompsz,size(u,2))));
    uMotMask{z} = normc(uMotMask{z});
end

h.avgmot = avgmot;
h.uMotMask = uMotMask;
