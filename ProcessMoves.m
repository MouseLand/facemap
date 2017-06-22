% processes movements in frames and projects motion and movie SVDs
% for whisker, groom, snout, face ROIs
function movs = ProcessMoves(frames,handles,j,avgframe,avgmotion)

% take chosen area
ROI = handles.ROI{j};
rX  = handles.rXc{j};
rY  = handles.rYc{j};
frames = frames(rY,rX,:);
avgf   = avgframe(rY,rX);
avgf   = avgf(:);
avgm   = avgmotion(rY,rX);
avgm   = avgm(:);
nX  = length(rX);
nY  = length(rY);
nframes = size(frames,3);

fr = [];
fr = reshape(frames,[],size(frames,3));

if handles.svdmat(j-2,1)
    %%%% compute motion
    fr = diff(fr,1,2);
    fr = abs(fr);
    
    % detect movements as change in pixels which are greater than a
    % user-specified threshold
    sats = (1-handles.saturation(j))*5;
    dfr   = squeeze(sum(fr>=sats,1));
end
motSVD = [];
if handles.svdmat(j-2,2)
    motMask = handles.motionMask{j-2};
    fr2     = bsxfun(@minus, fr, avgm);
    motSVD  = fr2'*motMask;
    motSVD  = [motSVD(1,:); motSVD];
end
movSVD = [];
if handles.svdmat(j-2,3)
    fr      = reshape(frames,[],size(frames,3));
    fr2     = bsxfun(@minus, fr, avgf);
    movMask = handles.movieMask{j-2};
    movSVD  = fr2'*movMask;
end

if ~isempty(dfr)
    movs{1} = [dfr(1);dfr(:)];
else
    movs{1} = [];
end
movs{2} = motSVD;
movs{3} = movSVD;
