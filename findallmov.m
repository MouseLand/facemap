function [filename,folders,namef] = findallmov(folder_name,fm,suffix)
filename{1} = [];
folders     = {};
namef       = {};
ik = 1;
for k = 1:length(fm)
    [~,namef0,ext] = fileparts(fm(k).name);
    ismov = 0;
    for ie = 1:length(suffix)
        if strcmp(ext, suffix{ie})
            ismov = 1;
            break;
        end
    end
    if ismov
        fname = fullfile(folder_name,fm(k).name);
        filename{ik} = fname;
        ik = ik+1;
        %idate   = [idate; fm(k).datenum];
        folders = cat(1,folders,fm(k).name);
        namef   = cat(1, namef, namef0);
    end
end
