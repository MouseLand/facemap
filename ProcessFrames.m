% process ROIs by loading bin file
function data = ProcessFrames(handles,wroi,wroim)

sc  = handles.sc;
data = ConstructData(handles);
nXc = floor(handles.nX/sc);
nYc = floor(handles.nY/sc);

ispupil = handles.whichROIs(1);
if ispupil
    fidp = fopen(handles.pupilfile,'r');
    nXp  = numel(handles.rX{1});
    nYp  = numel(handles.rY{1});
end

% if only pupil, do not load face.bin
isface = 1;
if isempty(wroim) && ~handles.whichROIs(2)
    isface = 0;
end

wroim = reshape(wroim,1,[]);

fileframes = handles.fileframes;
if isface
    fid = fopen(handles.facefile,'r');
end
fdata  = [];
fdatap = [];
ntall = 0;
for jf = 1:length(handles.files)
    nf    = 0;
    while ntall < fileframes(jf+1)
        nt           = 500 * sc;
        nt           = min(nt, fileframes(jf+1) - ntall);
        if isface
            fdata    = fread(fid,[nXc*nYc nt]);
            fdata    = reshape(fdata, nYc, nXc, size(fdata,2));
            nt0      = size(fdata,3);
        end
        if ispupil
            fdatap   = fread(fidp,[nXp*nYp nt]);
            nt0      = size(fdatap,2);
        end
        if isempty(fdata) && isempty(fdatap)
            disp('frame counting is off! :(');
            break;
        end
        ntall = ntall + nt0;
        
        % pupil ROI computation
        if ispupil
            fdatap                    = reshape(fdatap, nYp, nXp, size(fdatap,2));
            [pup]                     = ProcessPupil(handles,fdatap);
            if handles.whichROIs(1)
                data(jf).pupil.area   = cat(1,data(jf).pupil.area,pup.area);
                data(jf).pupil.center = cat(1,data(jf).pupil.center,pup.center);
                data(jf).pupil.com    = cat(1,data(jf).pupil.com,pup.com);
            end
        end
       
        % blink and motion ROI computations
        if isface
            if handles.whichROIs(2)
                sat                   = max(1, (1-handles.saturation(1))*255);
                fmap                  = fdata(handles.rYc{2}, handles.rXc{2}, :) < sat;
                blink.area            = squeeze(sum(sum(fmap,1),2));
                data(jf).blink.area   = cat(1, data(jf).blink.area,blink.area(:));
            end

            for j = wroim
                datroi = ProcessMoves(fdata, handles, j+2, handles.avgframe, handles.avgmotion);
                data(jf).mroi(j).motion   = cat(1, data(jf).mroi(j).motion, datroi{1});
                data(jf).mroi(j).motionSVD = cat(1, data(jf).mroi(j).motionSVD, datroi{2});
                data(jf).mroi(j).movieSVD = cat(1, data(jf).mroi(j).movieSVD, datroi{3});
            end
        end
        nf = nf+1;
        fprintf('file %d frameset %d/%d  time %3.2fs\n',jf,nf,...
            round((fileframes(jf+1)-fileframes(jf))/(500*sc)),toc);
        
    end
end
fclose('all');