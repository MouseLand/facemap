rootfolder = '\\zserver.cortexlab.net\Data\EyeCamera';

%%
% example db entry
clear db;
i=0;
i=i+1;
% one folder down (as a string)
db(i).mouse_name    = 'M150824_MP019';
% next folder down (as a string)
db(i).date          = '2015-12-19';
% next folder down (if empty, looks in 'date' folder)
% this is a number (change moviefolder num2str if it's not)
db(i).expts         = [4];

%%
for j = 1:length(db)
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
            % >>> change processing settings here <<<
            proc.tsc = 1; % changed temporal smoothing constant to 1 frame
            
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