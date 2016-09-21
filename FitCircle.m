function params = FitCircle(x,y)

% solves the problem
%   x^2+y^2+d*x+e*y+f=0
p=[x y ones(size(x))]\[-(x.^2+y.^2)];
% center of circle
d=p(1);
e=p(2);
f=p(3);
xc = -d/2;
yc = -e/2;
ra = sqrt((d^2+e^2)/4 - f);
rb = ra;
params.ra = ra;
params.rb = rb;
params.xc = xc;
params.yc = yc;
params.ang = 0;

params.isgood = 1;