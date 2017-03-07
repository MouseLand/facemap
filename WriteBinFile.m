% write face and pupil to binary file
function [fileframes, avgframe, avgmotion] = WriteBinFile(handles)
sc       = handles.sc;
tsc      = handles.tsc;
nX       = handles.nX;
nY       = handles.nY;
nXc   = sc * floor(nX/sc);
nYc   = sc * floor(nY/sc);


ispupil = handles.whichROIs(1);
if ispupil
    fidp = fopen(handles.pupilfile,'w');
end
% if only pupil, do not write face.bin
isface = 1;
if sum(handles.svdmat(:))==0 && ~handles.whichROIs(2)
    isface = 0;
end
if isface
    fid = fopen(handles.facefile,'w');
end
if isface
    avgframe = zeros(nYc/sc, nXc/sc, 'single');
else
    avgframe = zeros(nY, nX, 'single');
end
avgmotion = avgframe;
fileframes = 1;
nfall=0;
for jf = 1:length(handles.files)
    vr = VideoReader(handles.files{jf});
    % check if movie is RGB
    frame = read(vr,1);
    if size(frame,3) == 3
        isRGB = 1;
    else
        isRGB = 0;
    end
    nframes = vr.NumberOfFrames;
    k=1;
    nf = 1;
    if isRGB
        nt  = 700;
    else
        nt  = 2000;
    end
    while k < nframes
        
        ind0    = k;
        indend  = (k-1) + nt;
        if indend > nframes-1
            indend = Inf;
        end
        fdata   = read(vr, [ind0 indend]);
        fdata   = single(fdata);
       
        if isRGB
            fdata = (0.2989*fdata(:,:,1,:) + 0.5870*fdata(:,:,2,:) ...
               + 0.1140*fdata(:,:,3,:));  
        end
        % write pupil to file
        if ispupil
            fdatap = fdata(handles.rY{1},handles.rX{1},:);
            fdatap = uint8(round(my_conv2(...
                single(fdatap), [1 1 tsc], [1 2 3])));
            fwrite(fidp, reshape(fdatap, [], size(fdatap,3)));
            if ~isface
                avgframe = avgframe + squeeze(sum(single(fdata),4));
            end
        end
        
        % write face to file
        if isface
            % scale in X and Y
            fdata0 = squeeze(mean(mean(reshape(single(fdata(1:nYc,1:nXc,:)),...
                sc,nYc/sc,sc,nXc/sc,size(fdata,4)),1),3));
            % convolve in time
            fdata0 = my_conv2(fdata0, tsc, 3);
            % write to disk
            fdata1 = uint8(round(fdata0));
            fwrite(fid, reshape(fdata1, [], size(fdata1,3)));
            % compute avg frame and motion
            avgframe  = avgframe + sum(fdata0,3);
            if k > 1
                avgmotion = avgmotion + sum(abs(diff(fdata0,1,3)),3);
            end
        end
        fprintf('file %d frameset %d/%d  time %3.2fs\n',...
            jf,nf,round(nframes/(nt)),toc);
        k=k+size(fdata,4);
        nfall=nfall+size(fdata,4);
        nf=nf+1;
    end
    fileframes(jf+1) = nfall;
end
avgframe = avgframe/nfall;
avgmotion = avgmotion/(nfall-length(handles.files));

fclose('all');