function [filename,folders] = FindBlocks(handles,folder_name)

filename{1} = [];
folders     = {};
fs        = dir(folder_name);
fs        = fs(3:end);

% if it's a folder of folders, descend one to find .mj2's
idate = [];
ik = 1;
if ~isempty(fs)
if fs(1).isdir==1
    blks = questdlg('would you like to process all blocks?');
    switch blks
        case 'Yes'
            ik = 1;
            for j = 1:length(fs)
                fm = dir(fullfile(folder_name,fs(j).name));
                fm = fm(3:end);
                for k = 1:length(fm)
                    [~,~,ext] = fileparts(fm(k).name);
                    ismov = 0;
                    for ie = 1:length(handles.suffix)
                        if strcmp(ext, handles.suffix{ie})          
                            ismov = 1;
                            break;
                        end
                    end
                    if ismov
                        fname = fullfile(folder_name,fs(j).name,fm(k).name);
                        filename{ik} = fname;
                        idate   = [idate; fm(k).datenum];
                        folders = cat(1,folders,fs(j).name);
                        ik = ik+1;
                    end
                end
            end
            
        case 'No'
            ik = 1;
            fstr = {fs.name};
            [folds,didchoose] = listdlg('PromptString','which folders (ctrl for multiple)',...
                'SelectionMode','multiple','ListSize',[160 160],'ListString',fstr);
            if didchoose
                for j = folds
                    fm = dir(fullfile(folder_name,fs(j).name));
                    fm = fm(3:end);
                    for k = 1:length(fm)
                        [~,~,ext] = fileparts(fm(k).name);
                        ismov = 0;
                        for ie = 1:length(handles.suffix)
                            if strcmp(ext, handles.suffix{ie})
                                ismov = 1;
                                break;
                            end
                        end
                        if ismov
                            fname = fullfile(folder_name,fs(j).name,fm(k).name);
                            filename{ik} = fname;
                            idate   = [idate; fm(k).datenum];
                            folders = cat(1,folders,fs(j).name);
                            ik = ik+1;
                            
                        end
                    end
                end
            end
    end
end
end
    % sort files by date
% otherwise, find all .mj2s in current folder
fm = fs;
for k = 1:length(fm)
    [~,~,ext] = fileparts(fm(k).name);
    ismov = 0;
    for ie = 1:length(handles.suffix)
        if strcmp(ext, handles.suffix{ie})
            ismov = 1;
            break;
        end
    end
    if ismov
        fname = fullfile(folder_name,fm(k).name);
        filename{ik} = fname;
        ik = ik+1;
        idate   = [idate; fm(k).datenum];
        folders = cat(1,folders,fm(k).name);
    end
end

% sort files by time recorded
if numel(idate)>1
    [~,idate] = sort(idate);
    for j = 1:numel(idate)
        filename0{j} = filename{idate(j)};
    end
    folders = folders(idate);
    filename=filename0;
    
end

