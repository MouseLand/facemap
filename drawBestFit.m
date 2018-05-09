function params = drawBestFit(h, r)

Jsp = r.fr;
Th = 255-r.sats;

params = getRadius(h, Jsp, Th);

th = linspace(0, 2*pi, 180)';

params.xy = [cos(th) sin(th)] * params.R + params.mu(:)';
    