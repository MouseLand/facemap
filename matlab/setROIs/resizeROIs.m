% resizes ROIs based on the spatial downsampling set in the GUI
function h = resizeROIs(h, spatscale)

rsc = h.sc / spatscale;

for k = 1:size(h.ROI,1)
    nxS = floor(h.nX{k} / spatscale);
    nyS = floor(h.nY{k} / spatscale);

    for j = 1:numel(h.ROI{k})
        if ~isempty(h.ROI{k}{j})
            h.ROI{k}{j} = h.ROI{k}{j} * rsc;
            h.ROI{k}{j} = onScreenROI(h.ROI{k}{j}, nxS, nyS);
            
        end
    end
    for j = 1:numel(h.eROI{k})
        if ~isempty(h.eROI{k}{j})
            h.eROI{k}{j} = h.eROI{k}{j} * rsc;
            h.eROI{k}{j} = onScreenROI(h.eROI{k}{j}, nxS, nyS);
        end
    end
    for j = 1:numel(h.locROI)
        if ~isempty(h.locROI{j}) && h.ROIfile(j)==k 
            h.locROI{j} = h.locROI{j} * rsc;
            h.locROI{j} = onScreenROI(h.locROI{j}, nxS, nyS);
        end
    end
    
end
