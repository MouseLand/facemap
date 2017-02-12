function handles = ProcessROIs_bin(handles)

if sum(handles.whichROIs)==0 && sum(handles.svdmat(:))==0
    h=msgbox('no ROIs chosen for processing :(');
else
    svdmot = sum(handles.svdmat(:,2))>0;
    svdmov = sum(handles.svdmat(:,3))>0;
    
    %%%% put data on SSD
    tic;
    vr = VideoReader(handles.files{1});
    nX = vr.Height;
    nY = vr.Width;
    % X,Y subsampling
    sc    = 4;
    handles.sc = 4;
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
    facefile = 'F:\DATA\face.bin';
    handles.facefile = facefile;

    fprintf('writing face file to binary file\n');
    [fileframes, avgframe, avgmotion] = WriteBinFile(handles);
    handles.fileframes = fileframes;
    handles.avgframe = avgframe;
    handles.avgmotion = avgmotion;
    
    %%%% pass through data to compute SVDs
    if svdmot || svdmov
        tic;
        fprintf('computing SVDs across all movies...\n');
        handles = ComputeSVDMasks(handles);
        toc;
        handles.motionMask
        handles.movieMask
    end
    
    %%%% pass through data to compute pupil/blink/motion and svd projs
    % processing ROIS!
    fprintf('computing pupil/blink and motion and SVD projections...\n');
    wroi = find(handles.whichROIs(1:2))';
    wroim = find(sum(handles.svdmat,2)>0)';
    data = ProcessFrames(handles, wroi, wroim);
    
    %%
    % assign ROI's to proc if ROI was processed
    isproc = [wroi (2+find((sum(handles.svdmat,2)>0))')];
    proc = ConstructProc(data, handles, isproc);
    handles.proc = proc;
end