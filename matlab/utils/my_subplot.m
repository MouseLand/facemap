function h = my_subplot(ny, nx, i, varargin)

dx = .7;
dy = .7;
if numel(varargin)>0
   dx = varargin{1}(1); 
   dy = varargin{1}(2); 
end

ix = rem(i-1, nx)/nx + (1-dx)/nx * 2/3;
iy = 1 - ceil(i/nx)/ny + (1-dy)/ny *2/3;

h = axes('Position', [ix iy dx/nx dy/ny]);