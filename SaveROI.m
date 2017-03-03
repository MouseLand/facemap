% if there are multiple blocks, save the processed data for each block
% separately in its own folder

function handles = SaveROI(handles)

for j = 1:length(handles.files)
    clear proc;
    % folder path
    [savefolder,savefile0,~] = fileparts(handles.files{j});
    
    %proc.suffix = handles.suffix;
    proc.files = handles.files;
    proc.folders = handles.folders;
    proc.filename  = handles.files{j};
    
    if isfield(handles,'proc')
        % save processed data
        proc.data = handles.proc(j);
        proc.avgframe = handles.avgframe;
        proc.avgmotion = handles.avgmotion;
        savefile   = sprintf('%s_proc.mat',savefile0);
        savepath   = fullfile(savefolder,savefile);
        proc.saturation = handles.saturation;
        proc.whichROIs  = handles.whichROIs;
        proc.svdmat     = handles.svdmat;
        proc.sc = handles.sc;
        proc.tsc = handles.tsc;
        proc.thres = handles.thres;
        handles.settings = savepath;
        save(savepath,'proc');
    else
        proc.rX = handles.rX;
        proc.rY = handles.rY;
        proc.nX = handles.nX;
        proc.nY = handles.nY;
        proc.ROI = handles.ROI;
        proc.plotROIs = handles.plotROIs;
        proc.colors   = handles.colors;
        proc.saturation = handles.saturation;
        proc.whichROIs  = handles.whichROIs;
        proc.svdmat     = handles.svdmat;
        proc.sc = handles.sc;
        proc.tsc = handles.tsc;
        proc.thres = handles.thres;
        proc.binfolder = handles.binfolder;
        if isfield(handles,'motionMask')
            proc.motionMask = handles.motionMask;
        end
        if isfield(handles,'movieMask')
            proc.movieMask = handles.movieMask;
        end
        % save settings
        savefile   = sprintf('%s_settings.mat',savefile0);
        savepath   = fullfile(savefolder,savefile);
        handles.settings = savepath;
        save(savepath,'proc');
    end
end