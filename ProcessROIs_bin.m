% write face movie to binary file and then load to do SVDs, projections,
% pupil computations
function handles = ProcessROIs_bin(handles)


tstr = {'pupil','blink','whisker','groom','snout','face'};

wroi   = find(handles.whichROIs(1:2))';
wroim  = find(sum(handles.svdmat,2)>0)';
roiall = [wroi(:); wroim(:)+2];
wroiall = false(6,1);
wroiall(roiall) = 1;

if isempty(roiall)
    h=msgbox('no ROIs chosen for processing :(');
elseif sum((wroiall - handles.plotROIs) == 1) > 0
    h=msgbox('you chose to process ROI(s) that aren''t drawn');
else
    handles.svdmat
    svdmot = sum(handles.svdmat(:,2))>0;
    svdmov = sum(handles.svdmat(:,3))>0;
    
    %%%% put data on SSD
    tic;
    vr = VideoReader(handles.files{1});
   
    % X,Y subsampling
    sc    = handles.sc;
    nX    = vr.Width;
    nY    = vr.Height;
    handles.nX = nX;
    handles.nY = nY;
    % subsample chosen ROIs
    for j = 1:6
        rXc{j} = ceil(handles.rX{j}/sc);
        rXc{j} = unique(rXc{j});
        rYc{j} = ceil(handles.rY{j}/sc);
        rYc{j} = unique(rYc{j});
    end
    handles.rXc = rXc;
    handles.rYc = rYc;
    facefile = fullfile(handles.binfolder, 'face.bin');
    pupilfile = fullfile(handles.binfolder, 'pupil.bin');
    handles.facefile = facefile;
    handles.pupilfile = pupilfile;
    
    fprintf('\n----- PROCESSING ROIs: ');
    for j = 1:length(roiall)
        fprintf('%s ',tstr{roiall(j)});
    end
    fprintf('\n');
    
    fprintf('writing face and pupil to binary file\n');
    [fileframes, avgframe, avgmotion] = WriteBinFile(handles);
    handles.fileframes = fileframes;
    handles.avgframe = avgframe;
    handles.avgmotion = avgmotion;
    
    %%%% pass through data to compute SVDs
    for j = 1:4
        handles.movieMask{j}=[];
        handles.motionMask{j}=[];
    end
    if svdmot || svdmov
        fprintf('computing SVDs across all movies\n');
        handles = ComputeSVDMasks(handles);
        toc;
    end
    
    %%%% pass through data to compute pupil/blink/motion and svd projs
    % processing ROIS!
    fprintf('computing pupil/blink and motion and SVD projections\n');
    
    data = ProcessFrames(handles, wroi, wroim);
    
    %%
    % assign ROI's to proc if ROI was processed
    isproc = [wroi (2+find((sum(handles.svdmat,2)>0))')];
    proc = ConstructProc(data, handles, isproc);
    handles.proc = proc;
    
    fprintf('done processing!\n');
end