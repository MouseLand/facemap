function [fileframes, avgframe, avgmotion] = WriteBinFile(handles)
sc       = handles.sc;
nX       = handles.nX;
nY       = handles.nY;
nXc   = sc * floor(nX/sc);
nYc   = sc * floor(nY/sc);
    
fid = fopen(handles.facefile,'w');
avgframe = zeros(nXc/sc, nYc/sc,'single');
avgmotion = avgframe;
fileframes = 1;
nfall=0;
for jf = 1:length(handles.files)
    vr = VideoReader(handles.files{jf});
    nframes = round(vr.Duration*vr.FrameRate);
    k=0;
    jj=0;
    fdata = zeros(nX, nY, 1000,'single');
    while hasFrame(vr)
        k=k+1;
        nfall=nfall+1;
        jj=jj+1;
        
        fd = readFrame(vr);
        fdata(:,:,jj) = fd;
        
        if mod(k,2000)==0 || k==nframes
            fdata = fdata(:,:,1:jj);
            % scale in X and Y
            fdata = squeeze(mean(reshape(fdata(1:nXc,:,:),sc,nXc/sc,nY,jj),1));
            fdata = squeeze(mean(reshape(fdata(:,1:nYc,:),nXc/sc,sc,nYc/sc,jj),2));
            % convolve in time
            fdata = my_conv2(fdata, 3, 3);
            fdata = uint8(round(fdata));
            fwrite(fid, reshape(fdata, [], size(fdata,3)));
            
            avgframe  = avgframe + sum(single(fdata),3);
            if k > 1
                avgmotion = avgmotion + sum(abs(diff(single(fdata),1,3)),3);
            end
            disp(k)
            jj=0;
            fdata = zeros(nX, nY, 1000,'single');
            toc;
        end
    end
    fileframes(jf+1) = nfall;
end
avgframe = avgframe/nfall;
avgmotion = avgmotion/(nfall-length(handles.files));


fclose('all');
toc;