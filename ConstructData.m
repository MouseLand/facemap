% construct data structure to fill up
function data = ConstructData(handles)

data(length(handles.files)).mroi(4).motion = [];
data(length(handles.files)).mroi(4).motionSVD = [];
data(length(handles.files)).mroi(4).movieSVD = [];

for jf = 1:length(handles.files)
    for k = 1:4
        data(jf).mroi(k).motion = [];
        data(jf).mroi(k).motionSVD = [];
        data(jf).mroi(k).movieSVD = [];
    end
    
    for k = 1:2
        switch k
            case 1
                data(jf).pupil.area   = [];
                data(jf).pupil.center = [];
                data(jf).pupil.com    = [];
            case 2
                data(jf).blink.area   = [];
        end
    end
end