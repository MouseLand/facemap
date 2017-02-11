function data = ProcessFrames(handles,wroi,wroim)
sc  = handles.sc;
data = ConstructData(handles);
nXc = floor(handles.nX/sc);
nYc = floor(handles.nY/sc);

fileframes = handles.fileframes;
fid = fopen(handles.facefile,'r');
for jf = 1:length(handles.files)
    ntall = 0;
    nf    = 0;
    while ntall < fileframes(jf+1)
        nt   = 500 * sc;
        nt    = min(nt, fileframes(jf+1) - ntall);
        fdata = fread(fid,[nXc*nYc nt]);
        if isempty(fdata)
            disp('frame counting is off! :(');
            break;
        end
        ntall = ntall + size(fdata,2);
        fdata = reshape(fdata, nXc, nYc, size(fdata,2));
        
        % initialize pupil/blink ROIs
        [pbroi,wroi] = InitPupilBlink(handles,wroi,handles.avgframe);
        
        % pupil/blink ROI computation
        if ~isempty(wroi)
            [pup,blink]           = ProcessPupilBlink(handles,fdata,wroi,pbroi);
            if handles.whichROIs(1)
                data(jf).pupil.area   = cat(1,data(jf).pupil.area,pup.area);
                data(jf).pupil.center = cat(1,data(jf).pupil.center,pup.center);
                data(jf).pupil.com    = cat(1,data(jf).pupil.com,pup.com);
            end
            if handles.whichROIs(2)
                data(jf).blink.area   = cat(1, data(jf).blink.area,blink.area);
            end
        end
        % motion ROI computations
        for j = wroim
            datroi = ProcessMoves(fdata, handles, j+2, handles.avgframe, handles.avgmotion);
            data(jf).mroi(j).motion = cat(1, data(jf).mroi(j).motion, datroi{1});
            data(jf).mroi(j).motionSVD = cat(1, data(jf).mroi(j).motionSVD, datroi{2});
            data(jf).mroi(j).movieSVD = cat(1, data(jf).mroi(j).movieSVD, datroi{3});
        end
        nf = nf+1;
        fprintf('file %d frameset %d/%d  time %3.2fs\n',jf,nf,round(fileframes(jf+1)/(500*sc)),toc);
        
    end
end
fclose('all');