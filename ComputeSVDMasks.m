% load face file and compute SVD of motion and/or movie
function handles = ComputeSVDMasks(handles)
fid = fopen(handles.facefile,'r');
fileframes = handles.fileframes;
sc         = handles.sc;
nXc        = floor(handles.nX/sc);
nYc        = floor(handles.nY/sc);
rXc        = handles.rXc;  
rYc        = handles.rYc;

clear uMov uMot;
for j=1:4
    uMov{j}=[];
    uMot{j}=[];
end
wmot = find(handles.svdmat(:,2));
wmov = find(handles.svdmat(:,3));
wmot = wmot';
wmov = wmov';
%%
nt   = 500 * sc;

k = 0;
nf = round(fileframes(end)/nt);
while 1
    fdata = fread(fid,[nXc*nYc nt]);
    if isempty(fdata)
        break;
    end
    fdata = reshape(fdata, nYc, nXc, size(fdata,2));
    if ~isempty(wmov)
    for j = wmov
        fdata0  = fdata(rYc{j+2}, rXc{j+2}, :);
        avgframe0 = handles.avgframe(rYc{j+2}, rXc{j+2});
        fdata0  = reshape(fdata0, [], size(fdata0,3));
        fdata0  = bsxfun(@minus, single(fdata0), avgframe0(:));
        [u s v] = svd(fdata0' * fdata0);
        umov0   = fdata0 * u(:,1:min(100,size(u,2)));
        uMov{j}    = cat(2, uMov{j}, umov0);
    end
    end
    if ~isempty(wmot)
    for j = wmot
        fdata0  = fdata(rYc{j+2}, rXc{j+2}, :);
        avgmotion0 = handles.avgmotion(rYc{j+2}, rXc{j+2});
        fdata0  = reshape(fdata0, [], size(fdata0,3));
        fdata0  = abs(diff(single(fdata0),1,2));
        fdata0  = bsxfun(@minus, single(fdata0), avgmotion0(:));
        [u s v] = svd(fdata0' * fdata0);
        umot0   = fdata0 * u(:,1:min(100,size(u,2)));
        uMot{j} = cat(2, uMot{j}, umot0);
    end
    end
    k = k+1;
    fprintf('frameset %d/%d  time %3.2fs\n', k, round(fileframes(end)/nt), toc);
end

% take SVD of SVD components
if ~isempty(wmov)
for j = wmov
    [u s v] = svd(uMov{j}'*uMov{j});
    uMovMask = uMov{j} * u(:,1:min(100,size(u,2)));
    handles.movieMask{j} = uMovMask;
end
end
if ~isempty(wmot)
for j = wmot
    [u s v] = svd(uMot{j}'*uMot{j});
    uMotMask = uMot{j} * u(:,1:min(100,size(u,2)));
    uMotMask = normc(uMotMask);
    handles.motionMask{j} = uMotMask;
end
end
fclose('all');
