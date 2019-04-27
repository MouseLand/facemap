load('/media/carsen/DATA2/grive/10krecordings/2017-06-25_1_M161025_MP030_eye_proc.mat');
proc2=proc;

%%
nframes = proc2.nframes;

iframes = cumsum([0; nframes]);

for k = 1:length(nframes)

	clear proc;
	proc.data.face.motionSVD = proc2.motSVD{1}(iframes(k)+1:iframes(k+1), :);
	proc.data.face.motionMask = proc2.uMotMask{1};
	proc.data.pupil.area = proc2.pupil(1).area(iframes(k)+1:iframes(k+1));
	proc.data.pupil.com = proc2.pupil(1).com(iframes(k)+1:iframes(k+1), :);
	[~,fname]=fileparts(proc2.files{k});
	save(sprintf('/media/carsen/DATA2/grive/10krecordings/procFaces2/%s_proc.mat', fname), 'proc')
	
end