%%%% uses contour found in FindContours to quickly compute pupil/blink area
function [params] = FindEllipseandContour(r,tpt)

frame = r.fr(:,:,tpt);
boxX    = r.boxX;
boxY    = r.boxY;
boxinds = r.boxinds;    
boxext  = r.boxext;
cradius = r.cradius;
thres   = r.thres;

iextbox  = 1;
boxinds0 = boxinds(1:end-1);


fr = frame;
fr(fr>r.sats) = r.sats;
fr = fr/r.sats;
    %keyboard;
    
while ~isempty(iextbox) && (numel(boxinds)>numel(boxinds0))
     % find outside of contour (all points with less than 4 neighbors)
     [x,y] = FastContours(double(fr(boxX,boxY)),thres);
     x = round(x)+(boxY(1)-1);
     y = round(y)+(boxX(1)-1);
 
     extpts = sub2ind([r.nX r.nY],y,x);
     % does the exterior of the box intersect with the pupil at all?
     if ~isempty(boxext) && ~isempty(extpts)
         iextbox = ismember(boxext(:),extpts(:),'rows');
         iextbox = find(iextbox);
     else
         iextbox = [];
     end
     boxinds0 = boxinds;
     if ~isempty(iextbox)
         r.cradius = r.boxfact*r.cradius;
         [boxinds,boxext,boxX,boxY] = MakeBox(r);
     end     
end
  
if numel(extpts)>6
    [ix,iy] = ind2sub([r.nX r.nY],extpts);

     if r.fitellipse
         params = FitEllipse(ix,iy);
     else
         params = FitCircle(ix,iy);
         
     end
     % compute center of mass of contour
     fr = 1-fr;
     [ix,iy] = ind2sub([r.nX r.nY],boxinds);
     com = [sum(fr(boxinds).*ix) sum(fr(boxinds).*iy)]/sum(fr(boxinds));
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
