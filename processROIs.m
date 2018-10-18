% write face movie to binary file and then load to do SVDs, projections,
% pupil computations
function h = processROIs(h)

%%
tic;

% compute mean from a subset of frames
h = subsampledMean(h);
h.nframes = int64(h.nframes);

%%
% compute svd of videos ----------------------------- %
h = computeSVDmotion(h);

%% pupil, blink and running ROIs
h.spix{1}=[];
zf = find(h.plotROIs);
zf = zf(ismember(zf,[1 numel(h.plotROIs)-3:numel(h.plotROIs)]));
% these ROIs are not down-sampled in space
% h.iroi are full frame positions
for z = zf(:)'
    if h.plotROIs(z)
        k = h.ROIfile(z);
        nx = floor(h.nX{k});
        ny = floor(h.nY{k});
        h.spix{z} = false(ny, nx);
        iroi = max(1,floor(h.locROI{z} * h.sc));
        pos = round(iroi);
        h.spix{z}(pos(2)-1 + [1:pos(4)], pos(1)-1 + [1:pos(3)]) = 1;
        h.iroi{z} = pos;
    end
end

%%
% get timetraces for U and compute pupil and running ------------ %
h = projectMasks(h);

%%

saveROI(h);

fprintf('done processing!\n');
toc;

