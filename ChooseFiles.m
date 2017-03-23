function [filename0, folders0, namef0] = ChooseFiles(filename, folders, namef)



wfs = questdlg('would you like to process all movies?');
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
            'SelectionMode','multiple','ListSize',[200 160],'ListString',fstr);
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
end