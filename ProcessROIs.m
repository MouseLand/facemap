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
                    fr = my_conv2(single(fr),[2 2 3],[1 2 3]);
                    if handles.useGPU
                        fr = gather(fr);
                    end
                    fr = reshape(fr,[],size(fr,3));
                    fr2 = bsxfun(@minus,fr,mean(fr,2));
                    if sum(handles.svdmat(j,3))>0
                        [u s v] = svd(fr2'*fr2);
                        % pixels x components
                        nc = 20;
                        uMov{j}    = cat(2,uMov{j},normc(fr2*u(:,1:nc)));
                    end
                    clear fr2;
                    if sum(handles.svdmat(j,2))>0
                        % compute motion diff
                        fr   = abs(diff(fr,1,2));
                        [u s v] = svd(fr'*fr);
                        % pixels x components
                        nc = 20;
                        uMot{j}    = cat(2,uMot{j},normc(fr*u(:,1:nc)));
                    end
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
            end
            if ~isempty(uMot{j})
                [uMov0 s v] = svd(uMot{j}'*uMot{j});
                % pixels x components
                uMot{j}    = uMot{j}*uMov0(:,1:min(100,size(uMov0,2)));
                % normalize uMov
                uMot{j}    = normc(uMot{j});
                handles.motionMask{j} = (uMot{j});
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
    proc(length(handles.files)) = struct();
    proc(1).pupil = [];
    proc(1).blink = [];
    for jf = 1:length(handles.files)
        for k = 1:3
            proc(jf).whisker{k} = [];
            proc(jf).groom{k} = [];
            proc(jf).snout{k} = [];
            proc(jf).face{k} = [];
        end
    end
        
    tic;
    for jf = 1:length(handles.files)
        % compute mean image from video
        vr = VideoReader(handles.files{jf});
        handles.vr = vr;
        % take mean from 1000 random frames
        fdata = zeros(vr.Height,vr.Width,1000,'uint16');
        nframes = vr.Duration*vr.FrameRate-1;
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
            fdata = my_conv2(single(fdata),[2 2 3],[1 2 3]);
            if handles.useGPU
                fdata = gather(fdata);
            end
            
            % pupil/blink ROI computation
            if ~isempty(wroi)
                [pup,blink]   = ProcessPupilBlink(handles,fdata,wroi,pbroi);
                proc(jf).pupil = cat(1, proc(jf).pupil,pup');
                proc(jf).blink = cat(1, proc(jf).blink,blink');
            end
            
            % whisker ROI computation
            if sum(handles.svdmat(1,:))>0
                whisk = ProcessMoves(fdata,handles,3);
                for kk = 1:3
                     proc(jf).whisker{kk} = cat(1, proc(jf).whisker{kk},whisk{kk});
                end
            end
            
            % groom ROI computation
            if sum(handles.svdmat(2,:))>0
                groom = ProcessMoves(fdata,handles,4);
                for kk = 1:3
                    proc(jf).groom{kk} = cat(1, proc(jf).groom{kk},groom{kk});
                end
            end
            
            % snout ROI computation
            if sum(handles.svdmat(3,:))>0
                snout = ProcessMoves(fdata,handles,5);
                for kk = 1:3
                    proc(jf).snout{kk} = cat(1, proc(jf).snout{kk},snout{kk});
                end
            end
            
            % face ROI computation
            if sum(handles.svdmat(4,:))>0
                face = ProcessMoves(fdata,handles,6);
                for kk = 1:3
                    proc(jf).face{kk} = cat(1, proc(jf).face{kk},face{kk});
                end
            end
            nf = nf+1;
            fprintf('file %d frameset %d/%d  time %3.2fs\n',jf,nf,round(nframes/1000),toc);
            
        end
    end
    handles.proc = proc;
    
end
