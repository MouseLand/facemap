function [carea,ctot,params,ineigh] = FindContours(fmap,frame,nX,nY,fitellipse)
 nblk = 3;
 j = 1;
 idx = round(linspace(1,nX(j),nblk+1));
 idy = round(linspace(1,nY(j),nblk+1));
 [xB,yB]=MakeQuadrants(idx,idy);
 ixy = [1:nX(j)*nY(j)]';
 ineigh = [ixy+nX(j) ixy-nX(j) ixy+1 ixy-1];
 ineigh(ineigh<1 | ineigh>nX(j)*nY(j)) = NaN;

ik=0;
carea{1}=[];
ctot=[];
fsub = fmap;

for k = 1:numel(xB)
    iadjs = [];
    [icx,icy] = find(fsub(xB{k},yB{k}),1);
    icx = icx + xB{k}(1)-1;
    icy = icy + yB{k}(1)-1;
    adjs = sub2ind([nX nY],icx,icy);
    iadjs = adjs;
    nL = 1;
    while nL<length(iadjs)+1
        ic = iadjs(nL);
        %if ic == 1614
        %    pause;
        %end
        ispos = fmap(ineigh(ic,~isnan(ineigh(ic,:))));
        adjs = ineigh(ic,find(ispos));
        adjs = adjs(~isnan(adjs));
        %if ~isempty(find(adjs==1614))
        %    pause;
        %end
        iadjs = [iadjs;adjs'];
        iadjs = unique(iadjs,'stable');
        nL = nL+1;
    end
    
    % save contour
    if ~isempty(iadjs)
        ik=ik+1;
        carea{ik} = iadjs;
        fsub(iadjs) = 0;
        ctot = [ctot;length(iadjs)];
    end
end

params.isgood = 0;
params.ra = NaN;
params.rb = NaN;
params.ang = NaN;
params.com = [NaN NaN];
params.center = [NaN NaN];
params.xc = NaN;
params.yc = NaN;
 % use largest contour & fit an ellipse
 if sum(ctot)>5
     [~,imax] = max(ctot);
     iadjs = carea{imax};
     % find outside of contour (all points with less than 4 neighbors)
     fmap = fmap(:);
     fmap = [fmap;0];
     ineigh(isnan(ineigh)) = numel(fmap);
     extpts = iadjs(sum(fmap(ineigh(iadjs,:)),2)<4);
     [ix,iy] = ind2sub([nX nY],extpts);
     
     if fitellipse
         params = FitEllipse(ix,iy);
     else
         params = FitCircle(ix,iy);
     end
     % compute center of mass of contour
     [ix,iy] = ind2sub([nX nY],iadjs);
     
     com = [sum(frame(iadjs).*ix) sum(frame(iadjs).*iy)]/sum(frame(iadjs));
     params.com = com;
     
 end

