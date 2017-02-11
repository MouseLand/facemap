function data = ConstructData(handles)

data(length(handles.files)).mroi(4).motion = [];
data(length(handles.files)).mroi(4).motionSVD = [];
data(length(handles.files)).mroi(4).movieSVD = [];

for k = find(handles.whichROIs)'
    switch k
        case 1
            data(length(handles.files)).pupil.area   = [];
            data(length(handles.files)).pupil.center = [];
            data(length(handles.files)).pupil.com    = [];
        case 2
            data(length(handles.files)).blink.area   = [];
    end
end