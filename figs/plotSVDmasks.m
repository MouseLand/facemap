nvids = numel(proc.npix);

t0 = 20000;

clf;

np = [0 proc.npix];
np = cumsum(np);
mall=[];
nc = 10;
for ic = 1:nc
    for k = 1:nvids
        i1 = proc.uMotMask{1}(np(k)+[1:proc.npix(k)], ic);
        ib = NaN*zeros(floor(proc.nY{k}/proc.sc), floor(proc.nX{k}/proc.sc));
        ib(proc.wpix{k}) = i1;
        my_subplot(nc,nvids,(ic-1)*nvids + k, [.9 .9]);
        imagesc(ib,'alphadata',~isnan(ib));
        axis image;
        axis off;
        cmap=colormap('redblue');
        
    end
end

