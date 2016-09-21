%%%% uses contour found in FindContours to quickly compute pupil/blink area
function [params] = FindEllipse(r,tpt)

fmap = r.fmap(:,:,tpt);
frame = r.fr(:,:,tpt);
fmap = fmap(:);
fmap = [fmap;0];
boxinds = r.boxinds;    
boxext = r.boxext;
cradius = r.cradius;
ineigh = r.ineigh;
ineigh(isnan(ineigh)) = numel(fmap);

iextbox  = 1;
boxinds0 = boxinds(1:end-1);

while ~isempty(iextbox) && (numel(boxinds)>numel(boxinds0))
     % find outside of contour (all points with less than 4 neighbors)
     iadjs = boxinds(logical(fmap(boxinds)));
     extpts = iadjs(sum(fmap(ineigh(iadjs,:)'),1)<4);
     % does the exterior of the box intersect with the pupil at all?
     if ~isempty(boxext)
         iextbox = ismember(boxext,iadjs,'rows');
         iextbox = find(iextbox);
     else
         iextbox = [];
     end
     boxinds0 = boxinds;
     if ~isempty(iextbox)
         r.cradius = r.boxfact*r.cradius;
         [boxinds,boxext] = MakeBox(r);
     end     
end
  
if numel(iadjs)>5
   
    [ix,iy] = ind2sub([r.nX r.nY],extpts);

     if r.fitellipse
         params = FitEllipse(ix,iy);
     else
         params = FitCircle(ix,iy);
         
     end
     % compute center of mass of contour
     [ix,iy] = ind2sub([r.nX r.nY],iadjs);
     com = [sum(frame(iadjs).*ix) sum(frame(iadjs).*iy)]/sum(frame(iadjs));
     params.com = com;

else
    params.isgood = 0;
end

params.extpts = extpts;
if params.isgood==0
    params.ra = NaN;
    params.rb = NaN;
    params.ang = NaN;
    params.com = [NaN NaN];
    params.xc = NaN;
    params.yc = NaN;
end

if 0
    ixy=zeros(r.nX*r.nY,1);
    ixy(iadjs) = 1;
    imagesc(reshape(ixy,r.nX,r.nY))
end
