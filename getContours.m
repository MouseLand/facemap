function CC = getContours(C)

CC = struct([]);
[~, N] = size(C);
iContour = 0;
startPoint = 1;
while startPoint<N
    iContour = iContour + 1;
    CC(iContour).value = C(1, startPoint);
    nPoints = C(2, startPoint);
    CC(iContour).xx = C(1, startPoint+(1:nPoints))';
    CC(iContour).yy = C(2, startPoint+(1:nPoints))';
    startPoint = startPoint + nPoints + 1;
end
