rootfolder = 'Z:\data\EXP\';

%%
compile_spontdbs;

%%
for j = 1
    if ~isempty(db(j).expts)
        moviefolder = fullfile(rootfolder, db(j).mouse_name, db(j).date,...
            num2str(db(j).expts(1)));
    else
        moviefolder = fullfile(rootfolder, db(j).mouse_name, db(j).date);
    end
    % process all saved _proc files in folder
    procfile = dir(sprintf('%s\\*_proc.mat', moviefolder));
    if ~isempty(procfile)
		clear proc;
		load(fullfile(moviefolder, procfile(1).name));
		
		% open video files
		filename = proc.files;
		proc.vr = [];
		for k = 1:size(filename,1)
			for j = 1:size(filename,2)
				proc.vr{k,j} = VideoReader(filename{k,j});
			end
		end
		% run processing using proc structure
		fprintf('>>>>> processing %s %s\n',db(j).mouse_name, db(j).date);
		fprintf('file %s\n', fullfile(moviefolder, procfile(1).name));
		proc = processROIs(proc);
        
    else
        fprintf('empty %s %s\n',db(j).mouse_name, db(j).date);
    end
        
end