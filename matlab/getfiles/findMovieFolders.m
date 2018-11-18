% finds all movie files in current directory and all subdirectories
function [filename,folders,namef] = findMovieFolders(h,folder_name)


filename = {};
folders     = {};
namef       = {};
fs        = dir(folder_name);
fs        = fs(3:end);

%% check for files in root folder
if ~isempty(fs)
    isfolderf=[fs.isdir];
    if sum(~isfolderf)
        [file0,~,namef0] = findallmov(folder_name, fs(~isfolderf), h.suffix);
        for k = 1:length(file0)
            filename = cat(1,filename, file0{k});
            namef = cat(1,namef, namef0{k});
            folders = cat(1,folders,' ');
        end
    end
    % the files might be in separate folders
    % go down one folder and find all movies
    %%
    ifold=find(isfolderf);
    if ~isempty(ifold)
        for j = ifold(:)'
            fm = dir(fullfile(folder_name,fs(j).name));
            fm = fm(3:end);
            isfolderm = [fm.isdir];
            fm = fm(~isfolderm);
            
            [file0,~,namef0]=findallmov(fullfile(folder_name,fs(j).name),...
                fm,h.suffix);
            
            if ~isempty(file0{1})
                for k = 1:length(file0)
                    filename = cat(1,filename, file0{k});
                    namef = cat(1,namef, namef0{k});
                    folders = cat(1,folders,fs(j).name);
                end
            end
        end
        
        if length(filename)==1
            filename{1}=[];
        end
    end
else
    filename{1}=[];
end