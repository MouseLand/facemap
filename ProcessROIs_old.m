function handles = ProcessROIs(handles)

if sum(handles.whichROIs)==0 && sum(handles.svdmat(:))==0
    h=msgbox('no ROIs chosen for processing :(');
else
    % if SVDs are wanted, run a full pass through of data first
    if (sum(handles.svdmat(:,2))>0 )...% && ~isfield(handles,'motionMask'))...
            || (sum(handles.svdmat(:,3))>0)% && ~isfield(handles,'movieMask'))
        uMov = [];
        uMot = [];
        clear uMov uMot;
        for j=1:4
            uMov{j}=[];
            uMot{j}=[];
            handles.movieMask{j}=[];
            handles.motionMask{j}=[];
        end
            
        fprintf('computing SVDs across all movies...\n');
        tic;
        for jf = 1:length(handles.files)
            vr = VideoReader(handles.files{jf});
            nframes = vr.Duration*vr.FrameRate-1;
            nf = 0;
            Nfr = 1000;
            while hasFrame(vr) 
                fdata = zeros(vr.Height,vr.Width,Nfr,'uint16');
                k=0;
                while hasFrame(vr) && k<Nfr
                    k=k+1;
                    fdata(:,:,k) = readFrame(vr);
                    if mod(k,100)==0 && (((nf+1)*1000)<nframes)
                        nf = nf+1;
                        vr.CurrentTime = double(1/vr.FrameRate*(nf*1000));
                    end
                end
                fdata = fdata(:,:,1:k);
               
                wsvd = find((sum(handles.svdmat(:,2:3),2)>0)');
                
                for j = wsvd
                    if handles.useGPU
                        fr = gpuArray(fdata(handles.rX{j+2},handles.rY{j+2},:));
                    else
                        fr = fdata(handles.rX{j+2},handles.rY{j+2},:);
                    end
                    % convolve in space and time
                    fr = my_conv2(single(fr),[1 1 2],[1 2 3]);
                    if handles.useGPU
                        fr = gather(fr);
                    end
                    fr = reshape(fr,[],size(fr,3));
                    fr2 = bsxfun(@minus,fr,mean(fr,2));
                    if sum(handles.svdmat(j,3))>0
                        [u s v] = svd(fr2'*fr2);
                        % pixels x components
                        nc = 20;
                        uMov{j}    = cat(2,uMov{j},fr2*u(:,1:nc));
                    end
                    clear fr2;
                    if sum(handles.svdmat(j,2))>0
                        % compute motion diff
                        fr   = abs(diff(fr,1,2));
                        fr2 = bsxfun(@minus,fr,mean(fr,2));
                        [u s v] = svd(fr2'*fr2);
                        % pixels x components
                        nc = 20;
                        uMot{j}    = cat(2,uMot{j},fr2*u(:,1:nc));
                    end
                    clear fr2;
                end
                fprintf('file %d frameset %d/%d  time %3.2fs\n',jf,nf,ceil(nframes/1000),toc);
            end
        end
        % take svd of both
        for j = wsvd
            if ~isempty(uMov{j})
                [uMov0 s v] = svd(uMov{j}'*uMov{j});
                % pixels x components
                uMov{j}    = uMov{j}*uMov0(:,1:min(100,size(uMov0,2)));
                % normalize uMov
                uMov{j}    = normc(uMov{j});
                handles.movieMask{j} = (uMov{j});
                switch j
                    case 1
                        for jf = 1:length(handles.files)
                            proc(jf).whisker.movieMask = handles.movieMask{j};
                        end
                    case 2
                        for jf = 1:length(handles.files)
                            proc(jf).groom.movieMask   = handles.movieMask{j};
                        end
                    case 3
                        for jf = 1:length(handles.files)
                            proc(jf).snout.movieMask   = handles.movieMask{j};
                        end
                    case 4
                        for jf = 1:length(handles.files)
                            proc(jf).face.movieMask    = handles.movieMask{j};
                        end
                end
            end
            if ~isempty(uMot{j})
                [uMov0 s v] = svd(uMot{j}'*uMot{j});
                % pixels x components
                uMot{j}    = uMot{j}*uMov0(:,1:min(100,size(uMov0,2)));
                % normalize uMot
                uMot{j}    = normc(uMot{j});
                handles.motionMask{j} = (uMot{j});
                switch j
                    case 1
                        for jf = 1:length(handles.files)
                            proc(jf).whisker.motionMask = handles.motionMask{j};
                        end
                    case 2
                        for jf = 1:length(handles.files)
                            proc(jf).groom.motionMask   = handles.motionMask{j};
                        end
                    case 3
                        for jf = 1:length(handles.files)
                            proc(jf).snout.motionMask   = handles.motionMask{j};
                        end
                    case 4
                        for jf = 1:length(handles.files)
                            proc(jf).face.motionMask    = handles.motionMask{j};
                        end
                end
            end
        end
            
        
    end
    if isfield(handles,'motionMask')
        handles.motionMask
    end
    if isfield(handles,'movieMask')
        handles.movieMask
    end
    % processing ROIS!
    fprintf('computing pupil/blink and motion and SVD projections...\n');
    % initialize data structure
    for jf = 1:length(handles.files)
        for j = find(handles.svdmat(:,1))'
            switch j           
                case 1
                    proc(jf).whisker.motion = [];
                    proc(jf).whisker.motionSVD  = [];
                    proc(jf).whisker.movieSVD  = [];
                case 2
                    proc(jf).groom.motion = [];
                    proc(jf).groom.motionSVD = [];
                    proc(jf).groom.movieSVD = [];
                case 3
                    proc(jf).snout.motion = [];
                    proc(jf).snout.motionSVD = [];
                    proc(jf).snout.movieSVD = [];
                case 4
                    proc(jf).face.motion = [];
                    proc(jf).face.motionSVD = [];
                    proc(jf).face.movieSVD = [];
            end
        end
        for k = find(handles.whichROIs)'
            switch k
                case 1
                    proc(jf).pupil.area   = [];
                    proc(jf).pupil.center = [];
                    proc(jf).pupil.com    = [];
                case 2
                    proc(jf).blink.area   = [];
            end
        end
    end
    
        
    tic;
    for jf = 1:length(handles.files)
        % compute mean image from video
        vr = VideoReader(handles.files{jf});
        handles.vr = vr;
        % take mean from 1000 random frames
        fdata = zeros(vr.Height,vr.Width,1000,'uint16');
        nframes = round(vr.Duration*vr.FrameRate-1);
        
        indf = randperm(nframes,min(1000,nframes));
        for k = 1:min(1000,nframes)
            fdata(:,:,k) = read(vr,indf(k));
        end
        avgframe = median(fdata,3);
        handles.avgframe{jf} = avgframe;
        
        % initialize pupil/blink ROIs
        wroi = find(handles.whichROIs(1:2))';
        [pbroi,wroi] = InitPupilBlink(handles,wroi,avgframe);        
        
        vr = VideoReader(handles.files{jf});
        nf = 0;
        Nfr = 1000; % how many frames to process at a time
        while hasFrame(vr) 
            fdata = zeros(vr.Height,vr.Width,Nfr,'uint16');
            handles.cframe = Nfr*(nf)+1;
            k=0;
            while hasFrame(vr) && k<Nfr
                k=k+1;
                fdata(:,:,k) = readFrame(vr);
            end
            if handles.useGPU
                fdata = gpuArray(fdata(:,:,1:k));
            end
            % convolve in space and time
            fdata = my_conv2(single(fdata),[1 1 2],[1 2 3]);
            if handles.useGPU
                fdata = gather(fdata);
            end
            
            % pupil/blink ROI computation
            if ~isempty(wroi)
                [pup,blink]           = ProcessPupilBlink(handles,fdata,wroi,pbroi);
                if handles.whichROIs(1)
                    proc(jf).pupil.area   = cat(1,proc(jf).pupil.area,pup.area);
                    proc(jf).pupil.center = cat(1,proc(jf).pupil.center,pup.center);
                    proc(jf).pupil.com    = cat(1,proc(jf).pupil.com,pup.com);
                end
                if handles.whichROIs(2)
                    proc(jf).blink.area   = cat(1, proc(jf).blink.area,blink.area);
                end
            end
            % whisker ROI computation
            if sum(handles.svdmat(1,:))>0
                whisk = ProcessMoves(fdata,handles,3);
                max(whisk{1}(:))
                proc(jf).whisker.motion = cat(1, proc(jf).whisker.motion,whisk{1});
                proc(jf).whisker.motionSVD = cat(1, proc(jf).whisker.motionSVD,whisk{2});
                proc(jf).whisker.movieSVD = cat(1, proc(jf).whisker.movieSVD,whisk{3});
            end
            
            % groom ROI computation
            if sum(handles.svdmat(2,:))>0
                groom = ProcessMoves(fdata,handles,4);
                proc(jf).groom.motion = cat(1, proc(jf).groom.motion,groom{1});
                proc(jf).groom.motionSVD = cat(1, proc(jf).groom.motionSVD,groom{2});
                proc(jf).groom.movieSVD = cat(1, proc(jf).groom.movieSVD,groom{3});
            end
            
            % snout ROI computation
            if sum(handles.svdmat(3,:))>0
                snout = ProcessMoves(fdata,handles,5);
                proc(jf).snout.motion = cat(1, proc(jf).snout.motion,snout{1});
                proc(jf).snout.motionSVD = cat(1, proc(jf).snout.motionSVD,snout{2});
                proc(jf).snout.movieSVD = cat(1, proc(jf).snout.movieSVD,snout{3});
            end
            
            % face ROI computation
            if sum(handles.svdmat(4,:))>0
                face = ProcessMoves(fdata,handles,6);
                proc(jf).face.motion = cat(1, proc(jf).face.motion,face{1});
                proc(jf).face.motionSVD = cat(1, proc(jf).face.motionSVD,face{2});
                proc(jf).face.movieSVD = cat(1, proc(jf).face.movieSVD,face{3});
            end
            nf = nf+1;
            fprintf('file %d frameset %d/%d  time %3.2fs\n',jf,nf,round(nframes/1000),toc);
            
        end
    end
     
    % assign ROI's to proc
    % if ROI was processed
    isproc = [wroi (2+find((sum(handles.svdmat,2)>0))')];
    isproc
    rX = handles.rX;
    rY = handles.rY;
    for jf = 1:length(handles.files)
        for j = isproc
            switch j
                case 1
                    proc(jf).pupil.ROI = handles.ROI{j};
                    proc(jf).pupil.saturation = handles.saturation(j);
                    proc(jf).pupil.ROIX = rX{j};
                    proc(jf).pupil.ROIY = rY{j};
                    proc(jf).pupil.nX   = length(rX{j});
                    proc(jf).pupil.nY   = length(rY{j});
                case 2
                    proc(jf).blink.ROI = handles.ROI{j};
                    proc(jf).blink.saturation = handles.saturation(j);
                    proc(jf).blink.ROIX = rX{j};
                    proc(jf).blink.ROIY = rY{j};
                    proc(jf).blink.nX   = length(rX{j});
                    proc(jf).blink.nY   = length(rY{j});
                case 3
                    proc(jf).whisker.ROI = handles.ROI{j};
                    proc(jf).whisker.saturation = handles.saturation(j);
                    proc(jf).whisker.ROIX = rX{j};
                    proc(jf).whisker.ROIY = rY{j};
                    proc(jf).whisker.nX   = length(rX{j});
                    proc(jf).whisker.nY   = length(rY{j});
                case 4
                    proc(jf).groom.ROI = handles.ROI{j};
                    proc(jf).groom.saturation = handles.saturation(j);
                    proc(jf).groom.ROIX = rX{j};
                    proc(jf).groom.ROIY = rY{j};
                    proc(jf).groom.nX   = length(rX{j});
                    proc(jf).groom.nY   = length(rY{j});
                case 5
                    proc(jf).snout.ROI = handles.ROI{j};
                    proc(jf).snout.saturation = handles.saturation(j);
                    proc(jf).snout.ROIX = rX{j};
                    proc(jf).snout.ROIY = rY{j};
                    proc(jf).snout.nX   = length(rX{j});
                    proc(jf).snout.nY   = length(rY{j});
                case 6
                    proc(jf).face.ROI = handles.ROI{j};
                    proc(jf).face.saturation = handles.saturation(j);
                    proc(jf).face.ROIX = rX{j};
                    proc(jf).face.ROIY = rY{j};
                    proc(jf).face.nX   = length(rX{j});
                    proc(jf).face.nY   = length(rY{j});
            end
        end
    end
     handles.proc = proc;

end
