function rr = smoothPupil(parea)

win = 30;

rr = parea(:);

RR = zeros(win, numel(rr));
for k =-win/2:win/2
    if k<0
        RR(k+win/2+1, 1:end+k) = rr(-k+1:end);
    else
        RR(k+win/2+1, k+1:end) = rr(1:end-k);
    end
end
mrr = nanmedian(RR,1)';

ix = find(isnan(mrr) | isnan(rr));
ix2 = find(~(isnan(mrr) | isnan(rr)));
rr(ix) = interp1(ix2, mrr(ix2), ix);
mrr(ix) = rr(ix);

adiff = abs(rr - mrr);
[~, irr] = sort(adiff, 'descend');


TH = std(rr(:))/2;

ireplace = adiff > TH;
rr(ireplace) = mrr(ireplace);