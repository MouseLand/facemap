rootfolder = '\\zserver.cortexlab.net\Data\EyeCamera';

%%
% example db entry
clear db0;
compile_dbs;
db = db0(12);
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
        for k = 1:length(procfile)
            clear proc;
            load(fullfile(moviefolder, procfile(k).name));
            proc.svdmat = [1 1 1; 0 0 0; 0 0 0; 1 1 1];
            % >>> change processing settings here <<<
            %proc.tsc = 1; % changed temporal smoothing constant to 1 frame
            
            % run processing using proc structure
            fprintf('>>>>> processing %s %s\n',db(j).mouse_name, db(j).date);
            fprintf('file %s\n', fullfile(moviefolder, procfile(k).name));
            proc = ProcessROIs_bin(proc);
            
            % saves proc file to _proc.mat (overwriting previous file)
            proc = SaveROI(proc);
        end
    else
        fprintf('empty %s %s\n',db(j).mouse_name, db(j).date);
    end
        
end