% choose which movies to process and how to process them
function [filename, folders, namef] = chooseFiles(filename, folders, namef)
filename0{1} = [];
folders0     = {};
namef0       = {};

figpos = get(0,'DefaultFigurePosition');
figunits = get(0,'DefaultFigureUnits');
set(0,'DefaultFigurePosition',[1000 500 500 150],'DefaultFigureUnits','pixels');
wfs = MFquestdlg([.5,.5],'would you like to process all movies?',...
    '','Yes','No','Cancel','Cancel');
switch wfs
    case 'Yes'
        filename0 = filename;
        folders0  = folders;
        namef0    = namef;
        
    case 'No'
        fstr = {};
        for j = 1:numel(folders)
            fstr{j} = [folders{j} '/' namef{j}];
        end
        [fs,didchoose] = listdlg('PromptString','which movies (ctrl for multiple)',...
            'SelectionMode','multiple','ListSize',[300 400],'ListString',fstr);
        if didchoose
            ik = 1;
            for j = fs
                filename0{ik} = filename{j};
                folders0{ik}  = folders{j};
                namef0{ik}    = namef{j};
                ik = ik+1;
            end
        else
            msgbox('no selection - going to take all movies');
        end
    case 'Cancel'
        filename=filename(:);
        folders=folders(:);
        namef=namef(:);
        return;
end

clear filename folders namef;

% ask if user is processing multiple movies simultaneously taken
ismulti = MFquestdlg([.5 .5],'are you processing multiple videos taken simultaneously?',...
    'multi-video?','Yes','No','Cancel','No');

if strcmp(ismulti, 'Yes')
    isblk = MFquestdlg([.5 .5],'are there multiple separate movies of the same view?',...
        '','Yes','No','Cancel','Cancel');
elseif length(filename0) > 1
    isblk = 'Yes';
else
    isblk = 'No';
end

if strcmp(isblk,'Yes')
    filename{1}=[];
    folders = [];
    namef = [];
    
    iview = {};
    if strcmp(ismulti,'Yes')
        % group files by first 4 letters
        for j = 1:length(filename0)
            fbeg=namef0{j}(1:4);
            if sum(~cellfun(@isempty,strfind(iview, fbeg)))==0 && j~=1
                iview = cat(1,iview,fbeg);
            end
        end
        iview
        for k = 1:length(iview)
            ij = 0;
            for j=1:length(filename0)
                if strcmp(iview{k}, namef0{j}(1:4))
                    ij = ij+1;
                    filename{k,ij} = filename0{j};
                    folders{k,ij}  = folders0{j};
                    namef{k,ij}      = namef0{j};
                end
            end
        end
    
        
    else
        for j = 1:length(filename0)
            filename{1,j} = filename0{j};
            folders{1,j}  = folders0{j};
            namef{1,j}      = namef0{j};
        end
    end
    
    
    
    % check that each view has the same number of movies
    for k = 1:size(filename,1)
        for j = 1:size(filename,2)
            if isempty(filename{k,j})
                clear filename;
                filename{1}=[];
                msgbox('error: not all views have the same number of movies!');
                return;
            end
        end
    end
else
    filename = filename0(:);
    folders = folders0(:);
    namef = namef0(:);
end

set(0,'DefaultFigurePosition',figpos,'DefaultFigureUnits',figunits);
