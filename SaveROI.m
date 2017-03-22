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
    end
    
    proc.binfolder = handles.binfolder;
    proc.rX = handles.rX;
    proc.rY = handles.rY;
    proc.nX = handles.nX;
    proc.nY = handles.nY;
    proc.ROI = handles.ROI;
    proc.whichROIs  = handles.whichROIs;
    proc.colors   = handles.colors;
    proc.saturation = handles.saturation;
    proc.svdmat     = handles.svdmat;
    proc.plotROIs   = handles.plotROIs;
    proc.sc = handles.sc;
    proc.tsc = handles.tsc;
    proc.thres = handles.thres;
    
    savefile   = sprintf('%s_proc.mat',savefile0);
    savepath   = fullfile(savefolder,savefile);
    handles.settings = savepath;
    save(savepath,'proc');
    
end