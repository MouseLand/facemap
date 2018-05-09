
function K = kern(xp, yp,len)

K = exp( -(xp(:) - yp(:)').^2 /(2*len^2));




 