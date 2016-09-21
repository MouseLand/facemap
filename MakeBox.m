function [ix,iext,ixX,ixY] = MakeBox(r)

ccenter = r.ccenter;
cradius = r.cradius;
cradius = [cradius(2) cradius(1)];
nX = [r.nX r.nY];
[ix,iy] = ndgrid([1:nX(1)],[1:nX(2)]);
ix = [ix(:) iy(:)];
iext = ix;
iext0 = [];

for k = 1:2
    xmin = round(max(1,ccenter(k) - cradius(k)));
    xmax = round(min(nX(k),ccenter(k) + cradius(k)));
    ix = ix(ix(:,k)>=xmin & ix(:,k)<=xmax,:);
    boxX{k} = [xmin:xmax];
    if xmin>1
        iext0 = [iext0; ix(ix(:,k)==xmin,:)];
    end
    if xmax<nX(k)
        iext0 = [iext0; ix(ix(:,k)==xmax,:)];
    end
    if ~isempty(iext0)
        iext0 = iext0(iext0(:,k)>=xmin & iext0(:,k)<=xmax,:);
    end
end
iext = iext0;
ix = sub2ind([nX(1) nX(2)],ix(:,1),ix(:,2));
if ~isempty(iext)
    iext = sub2ind([nX(1) nX(2)],iext(:,1),iext(:,2));
end
ixX = boxX{1};
ixY = boxX{2};
