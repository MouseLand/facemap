function handles = InitPlot(handles,j)

r.rX       = handles.rX{j};
r.rY       = handles.rY{j};
r.nX       = numel(r.rX);
r.nY       = numel(r.rY);
r.sats     = (1-handles.saturation(j))*255;

if j==1
    r.boxfact = 1.2;
else
    r.boxfact = 1.4;
end
r.ccenter = round([r.nX/2 r.nY/2]);
if j == 1
    r.cradius = r.boxfact*[r.nX/5 r.nY/5];
else
    r.cradius = r.boxfact*[r.nX/3 r.nY/3];
end
% box around pupil
[binds,extinds,boxX,boxY] = MakeBox(r);
r.boxinds = binds;
r.boxext  = extinds;
r.boxX    = boxX;
r.boxY    = boxY;
r.fitellipse = handles.fitellipse(j);
for fn = fieldnames(r)'
    handles.roif(j).(fn{1}) = r.(fn{1});
end