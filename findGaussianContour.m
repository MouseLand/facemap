% fits multivariate gaussian to COM of pupil
% to find COM, first take pixel of max darkness, zoom in there
% compute COM of zoomed in region
% recenter box on new COM
% fit gaussian
function [params] = findGaussianContour(r)

params.xy     = [];
params.area   = 0;
params.mu     = [NaN NaN];
params.isgood = 0;

NT=size(r.fr,3);
for tpt = 1:NT
    params0.area   = NaN;
    params0.mu     = [NaN NaN];
    params0.isgood = 0;
    params0.xy = [];
    
    frame = r.fr(:,:,tpt);
    r.nX  = size(frame,1);
    r.nY  = size(frame,2);
    
    %
    % zero out pixels < saturation level
    fr    = frame;
    fr    = 255-fr;
    fr    = max(0, fr-(r.sats));
    %fr(fr>40) = 255;
    %fr    = 255 - fr;
    %imagesc(fr)
    % find pixel of max brightness
    [~,ix] = max(fr(:));
    [ix,iy] = ind2sub(size(fr),ix);
    
    % find com in window of ROI size
    ixinds = ix + [-1*round(r.nX/2):round(r.nX/2)];
    ixinds(ixinds>r.nX | ixinds<1) = [];
    iyinds = iy + [-1*round(r.nY/2):round(r.nY/2)];
    iyinds(iyinds>r.nY | iyinds<1) = [];
    iyinds = repmat(iyinds(:), 1, numel(ixinds));
    ixinds = repmat(ixinds(:)', size(iyinds,1), 1);
    %
    ix     = sub2ind(size(fr), ixinds, iyinds);
    com    = [sum(ixinds(:).*fr(ix(:))) sum(iyinds(:).*fr(ix(:)))] /sum(fr(ix(:)));
    
    % recenter box on com
    if ~isnan(com(1))
        ix     = round(com(1));
        iy     = round(com(2));
        ixinds = ix + [-1*round(r.nX/2):round(r.nX/2)];
        ixinds(ixinds>r.nX | ixinds<1) = [];
        iyinds = iy + [-1*round(r.nY/2):round(r.nY/2)];
        iyinds(iyinds>r.nY | iyinds<1) = [];
        iyinds = repmat(iyinds(:), 1, numel(ixinds));
        ixinds = repmat(ixinds(:)', size(iyinds,1), 1);
        ix     = sub2ind(size(fr), ixinds, iyinds);
        
        if sum(fr(ix(:))>0) > 1
            params0 = fitMVGaus(iyinds(:), ixinds(:), fr(ix(:)), r.thres);
            params0.isgood = 1;
        end
    end
    
    params.mu(tpt,:) = params0.mu;
    params.area(tpt) = params0.area;
end
params.isgood = params0.isgood;
params.xy = params0.xy;