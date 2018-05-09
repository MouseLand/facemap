% function ops = getRunning(fname, ops)
% outputs the dx, dy offsets between frames by registering frame N to frame
% N-1. If the movement is larger than half the frame size, outputs NaN. 
% ops.yrange, xrange are ranges to use for rectangular section of movie


fname = 'D:\cams5\cam0_G7c1_2018_03_16_5.avi';
vr = VideoReader(fname);
Iread = readFrame(vr);

ops.yrange = 700:946;
ops.xrange = 100:700;

if size(Iread,1)<max(ops.yrange) || size(Iread,2)<max(ops.xrange)
    error('X or Y ranges larger than frame size');
end

imagesc(Iread)

vr.CurrentTime = 0;
%%
ops.useGPU = 0;

% I = randn(512, 512, 100);

ly = numel(ops.yrange);
lx = numel(ops.xrange);

[ys, xs] = ndgrid(1:ly, 1:lx);
ys = abs(ys - mean(ys(:)));
xs = abs(xs - mean(xs(:)));

maskSlope   = 8; % slope on taper mask preapplied to image. was 2, then 1.2
mY      = max(ys(:)) - maskSlope;
mX      = max(xs(:)) - maskSlope;

% SD pixels of gaussian smoothing applied to correlation map (MOM likes .6)
smoothSigma = 3;

maskMul = single(1./(1 + exp((ys - mY)/maskSlope)) ./(1 + exp((xs - mX)/maskSlope)));
eps0 = single(1e-20);

lCorrX = ceil(lx/4);
lCorrY = ceil(ly/4);

xCorrRef = [(lx - lCorrX + 1):lx 1:(lCorrX + 1)];
yCorrRef = [(ly - lCorrY + 1):ly 1:(lCorrY + 1)];

% Smoothing filter in frequency domain
hgx = exp(-(((0:lx-1) - fix(lx/2))/smoothSigma).^2);
hgy = exp(-(((0:ly-1) - fix(ly/2))/smoothSigma).^2);
hg = hgy'*hgx;
fhg = real(fftn(ifftshift(single(hg/sum(hg(:))))));


Jlast = [];

dv = [];
corrv = [];

vr = VideoReader(fname);

Nbatch = 100;
ik = 0;
while 1
    
    k = 0;
    while hasFrame(vr)
        k = k+1;
        Iread = mean(single(readFrame(vr)),3);        
        if k==1
           I = zeros(size(Iread,1), size(Iread,2), Nbatch, class(Iread)); 
        end
        
        I(:, :, k) = Iread;
        
        if k==Nbatch
           break; 
        end
    end
    if k==0
        break;
    end
    
    J = I(ops.yrange, ops.xrange, :);
    J = bsxfun(@minus, J, mean(mean(J,1),2));
    
    if ops.useGPU
       J = gpuArray(J); 
    end
    J = single(J);
    
    J = bsxfun(@times, J, maskMul);
    
    Jfft = fft2(J);
    
    Jfft = Jfft./(eps0 + abs(Jfft));
    Jfft = Jfft .* fhg;
    
    if isempty(Jlast)
        Jlast = Jfft(:,:,1);        
    end
    
    Jpre = cat(3, Jlast, Jfft(:, :, 1:end-1));    
    Jfft = Jfft .* conj(Jpre);
    
    Jcorr = real(ifft2(Jfft));
   
    % now register J to Jpre
    corrClip = Jcorr(yCorrRef, xCorrRef, :);
    corrClipSmooth = my_conv2(corrClip, 1, [1 2]);
    
    [dmax, iy] = max(corrClipSmooth, [], 1);

    if ops.useGPU
        iy = gather(iy);
        dmax = gather(dmax);
    end
    
    [dmax, ix] = max(dmax, [], 2);
    
    iy = reshape(...
        iy(sub2ind([size(iy,2) size(iy,3)], ix(:), (1:size(iy,3))')),...
        1, 1, []);
    
    bcorr = dmax;
    
    ix = (ix - lCorrX - 1);
    iy = (iy - lCorrY - 1);
    
    fi = ik + [1:size(J,3)];
    ik = ik + size(J,3);
    
    dv(fi,:) = [iy(:) ix(:)];
    corrv(fi) = squeeze(bcorr);
    
   Jlast = J(:,:,end);
      
   if k<Nbatch
       break;
   end
   
   break;

end