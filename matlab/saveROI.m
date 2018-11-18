% if there are multiple blocks, save the processed data for each block
% separately in its own folder

function saveROI(h)

proc.files = h.files;
proc.nframes = h.nframes;
proc.rootfolder = h.rootfolder;
proc.nX = h.nX;
proc.nY = h.nY;
proc.sc = h.sc;
proc.useGPU = h.useGPU;
proc.ROI = h.ROI;
proc.eROI = h.eROI;
proc.locROI = h.locROI;
proc.ROIfile = h.ROIfile;
proc.plotROIs = h.plotROIs;
proc.saturation = h.saturation;
proc.thres = h.thres;

if isfield(h, 'tpix')
    proc.tpix = h.tpix;
    proc.npix = h.npix;
    proc.wpix = h.wpix;
end

if isfield(h,'motSVD')
    %proc.spix = h.spix;
    proc.avgmot = h.avgmot;
    
    % save processed data
    proc.motSVD   = h.motSVD;
    proc.uMotMask = h.uMotMask;
    
    for j = 2:length(proc.uMotMask)
        if ~isempty(proc.uMotMask{j})
            um = proc.uMotMask{j};
            ROI = round(proc.locROI{j});
            um = reshape(um, ROI(4), ROI(3), size(um,ndims(um)));
            proc.uMotMask{j} = um;
        end
    end
    
    proc.avgframe = h.avgframe;
    proc.avgmotion = h.avgmotion;
end

if isfield(h, 'runSpeed')
    proc.runSpeed = h.runSpeed;
end

if isfield(h, 'pupil')
    for j = 1:length(h.pupil)
        if ~isempty(h.pupil(j).area)
            psmooth = smoothPupil(h.pupil(j).area);
            h.pupil(j).area_raw = h.pupil(j).area(:);
            h.pupil(j).area = psmooth;
        end
    end
    proc.pupil = h.pupil;
end

if isfield(h, 'blink')
	proc.blink = h.blink;
end


%%
[~,fname,~] = fileparts(h.files{1});
fname = [fname '_proc.mat'];

%%
savefile = fname;
%savepath   = fullfile('/media/carsen/SSD/', savefile);
savepath   = fullfile(h.folder_name, savefile);
h.settings = savepath;
save(savepath,'-v7.3','proc');
