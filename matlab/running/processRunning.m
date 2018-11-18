% outputs the dx, dy offsets between frames by registering frame N to frame
% N-1. If the movement is larger than half the frame size, outputs NaN. 
% ops.yrange, xrange are ranges to use for rectangular section of movie
function [DS,corrv] = processRunning(h, frames)

[ly, lx, nt] = size(frames);
lCorrX = ceil(lx/4);
lCorrY = ceil(ly/4);

xCorrRef = [(lx - lCorrX + 1):lx 1:(lCorrX + 1)];
yCorrRef = [(ly - lCorrY + 1):ly 1:(lCorrY + 1)];

    
J = bsxfun(@minus, frames, mean(mean(frames,1),2));
    
if h.useGPU
    J = gpuArray(J);
end
J = single(J);
    
J = bsxfun(@times, J, h.maskMul);
    
Jfft = fft2(J);

% phase correlation
eps0 = single(1e-20);
Jfft = Jfft./(eps0 + abs(Jfft));
Jfft = Jfft .* h.fsmooth;

% now register J to Jprevious
Jfft = Jfft(:,:,2:end) .* conj(Jfft(:,:,1:end-1));

Jcorr = real(ifft2(Jfft));
clear Jfft;
Jcorr = Jcorr(yCorrRef, xCorrRef, :);
Jcorr = my_conv2(Jcorr, 1, [1 2]);

[dmax, iy] = max(Jcorr, [], 1);

if h.useGPU
    iy = gather(iy);
    dmax = gather(dmax);
end

[dmax, ix] = max(dmax, [], 2);

iy = reshape(iy(sub2ind([size(iy,2) size(iy,3)], ix(:), (1:size(iy,3))')),...
    1, 1, []);

bcorr = dmax;

ix = (ix - lCorrX - 1);
iy = (iy - lCorrY - 1);

DS = [iy(:) ix(:)];
corrv = squeeze(bcorr);

