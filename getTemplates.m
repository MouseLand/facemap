function h = getTemplates(h)

iroi = max(1,floor(h.locROI{h.indROI} * h.sc));

ly = iroi(4);
lx = iroi(3);
smoothSigma = h.thres;

R = linspace(1, ceil(ly/2), 101);
slope = 1/2 ;

[xs, ys] = meshgrid(1:lx, 1:ly);
% xs = repmat([1:lx], ly, 1);
% ys = repmat([1:ly]', 1, size(Jsp,2));
mx = mean(xs(:));
my = mean(ys(:));

dx = xs - mx;
dy = ys - my;

Im = zeros(size(dx,1), size(dx,2), numel(R));
for t = 1:numel(R)
    Im(:,:,t) = 1./(1+exp(-(sqrt(dx.^2 + dy.^2) - R(t)) /slope)) - 1;    
end
Im = single(gpuArray(Im));

hgx = exp(-(((0:lx-1) - fix(lx/2)).^2/(2*smoothSigma^2)));
hgy = exp(-(((0:ly-1) - fix(ly/2)).^2/(2*smoothSigma^2)));
hg = hgy'*hgx;
fhg = real(fftn(ifftshift(single(hg/sum(hg(:))))));

eps = 1e-6;
fIm = conj(fft2(Im)) ;
fIm = fIm ./ (eps + abs(fIm)) .* fhg;
fIm = single(fIm);

if h.useGPU
    fIm = gpuArray(fIm);
end
h.fIm{h.indROI} = fIm;
h.R{h.indROI} = R;