function h = runningFilters(h)
ly = h.iroi{1}(4);
lx = h.iroi{1}(3);

[ys, xs] = ndgrid(1:ly, 1:lx);
ys = abs(ys - mean(ys(:)));
xs = abs(xs - mean(xs(:)));

maskSlope = 8; % slope on taper mask preapplied to image
mY        = max(ys(:)) - maskSlope;
mX        = max(xs(:)) - maskSlope;

% SD pixels of gaussian smoothing applied to correlation map
smoothSigma = 3;
maskMul = single(1./(1 + exp((ys - mY)/maskSlope)) ./(1 + exp((xs - mX)/maskSlope)));

% Smoothing filter in frequency domain
hgx = exp(-(((0:lx-1) - fix(lx/2))/smoothSigma).^2);
hgy = exp(-(((0:ly-1) - fix(ly/2))/smoothSigma).^2);
hg = hgy'*hgx;
fhg = real(fftn(ifftshift(single(hg/sum(hg(:))))));

if h.useGPU
    maskMul = gpuArray(maskMul);
    fhg = gpuArray(fhg);
end    

h.maskMul = maskMul;
h.fsmooth = fhg;