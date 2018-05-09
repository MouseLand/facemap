function params = getRadius(h, Jsp, mJ) 

Jsp(Jsp>mJ) = mJ;

[ly, lx, nframes] = size(Jsp);

fIm = h.fIm{h.indROI};
R   = h.R{h.indROI};

fJsp = fft2(Jsp);
fJsp = fJsp ./ (eps + abs(fJsp));
%
cc = zeros([size(fJsp,1),size(fJsp,2),size(fJsp,3) size(fIm,3)], 'single');
fJsp =gpuArray(single(fJsp));

for t = 1:size(fIm, 3)
    cc(:,:,:,t) = gather(real(ifft2(fJsp .* fIm(:,:,t))));    
end
cc = fftshift(cc, 1);
cc = fftshift(cc, 2);

cc([1:8 [end-7:end]], :, :) = 0;
cc(:, [1:8 [end-7:end]], :) = 0;

[cmax, imax] = max(cc, [], 4);

cmax = reshape(cmax, [], size(cmax,3));

[ccmax, im] = max(cmax, [], 1);

xmax = ceil(im/ly);
ymax = im - ly*(ceil(im/ly)-1);

isub = im + size(cmax,1) *([1:size(imax,3)]-1);
% imap = imax(isub);
% cmax = reshape(cmax, size(Jsp,1), size(Jsp,2), []);

cc  = reshape(cc, [], numel(R));
cpk = cc(isub, :)';

Rup = linspace(R(1), R(end), 1001);
K1 = kern(Rup, R, 1);
K2 = kern(R, R, 1) + .1 * eye(numel(R));

[~, imapup] = max(K1 * (K2\cpk), [], 1);

Rout = Rup(imapup);

params.area   = pi * Rout(:).^2;
params.mu     = [xmax(:) ymax(:)];
params.isgood = 1;
params.R = Rout(:);