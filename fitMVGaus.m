% fit 2D gaussian to cell with lam pixel weights
function params = fitMVGaus(iy, ix, lam0, thres)

% normalize pixel weigths
lam = lam0;

% iteratively fit the Guassian, excluding outliers
for k = 1:5
    lam     = lam / sum(lam);    
    mu      = [sum(lam.*iy) sum(lam.*ix)];    
    xydist      = bsxfun(@minus, [iy ix], mu);
    xy      = bsxfun(@times, xydist, sqrt(lam));    
    sigxy   = xy' * xy;    
    lam = lam0;
    dd = sum((xydist / sigxy) .* xydist, 2);    
    lam(dd > 2 * thres^2) = 0;
end
%%
params.mu = mu;
params.sig = sigxy;
[evec,eval]=eig(thres^2*params.sig);
eval = real(eval);

% enforce some circularity on pupil
% principal axis can only be 2x bigger than minor axis
min_eval = min(diag(eval));
eval     = min_eval * min(4, eval/min_eval);

% compute pts surrounding ellipse
n=100; % Number of points around ellipse
p=0:pi/n:2*pi; % angles around a circle
xy = [cos(p'),sin(p')] * sqrt(eval) * evec'; % Transformation
xy = bsxfun(@plus,xy,mu);

params.xy = xy;
eval=diag(eval);
params.eval = eval;
params.area = sqrt(eval(1) * eval(2)) * pi;
params.area = real(params.area);
    