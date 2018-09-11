function varargout = MovieGUI(varargin)
% MOVIEGUI MATLAB code for MovieGUI.fig
%      MOVIEGUI, by itself, creates a new MOVIEGUI or raises the existing
%      singleton*.
%
%      H = MOVIEGUI returns the handle to a new MOVIEGUI or the handle to
%      the existing singleton*.
%
%      MOVIEGUI('CALLBACK',hObject,eventData,h,...) calls the local
%      function named CALLBACK in MOVIEGUI.M with the given input arguments.
%
%      MOVIEGUI('Property','Value',...) creates a new MOVIEGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MovieGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MovieGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIh

% Edit the above text to modify the response to help MovieGUI

% Last Modified by GUIDE v2.5 11-Sep-2018 13:13:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @MovieGUI_OpeningFcn, ...
    'gui_OutputFcn',  @MovieGUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before MovieGUI is made visible.
function MovieGUI_OpeningFcn(hObject, eventdata, h, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% h    structure with h and user data (see GUIDATA)
% varargin   command line arguments to MovieGUI (see VARARGIN)

% Choose default command line output for MovieGUI
h.output = hObject;

addpath(genpath('.')); % add all subdirectories here to path

% default filepath for eye camera
h.filepath = '/media/carsen/DATA2/grive/sample_movies/';
h.suffix   = {'.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf'}; % suffix of eye camera file!

% use a GPU? default 1
h.useGPU = 1;

% default smoothing constants
h.sc        = 4;
h.tsc       = 1;

% colors for ROIs
h.colors = [1 0 1;...
           1 .65 0;...
           .85 .3 .1;...
           .8 .8 0;...
           0 1 0;...
           .3 .7 0;
		   0 0.5 0.5;
		   0 0.4 0.8];
% default threshold for pupil
h.thres = 4;

set(h.slider2,'Min',0);
set(h.slider2,'Max',1);
set(h.slider2,'Value',0);
set(h.edit1,'String',num2str(0));
set(h.pupilsigma,'String',num2str(4));
h.thres = 2;
h.saturation = zeros(100,1);
h.framesaturation = zeros(100,1);
h.whichfile = 1;
h.ROIfile = zeros(8,1);
h.plotROIs = false(8,1);
h.indROI = [];
for k = 1:length(h.plotROIs)
    h.locROI{k} = [];
end

axes(h.axes1);
set(gca,'xtick',[],'ytick',[]);
box on;

axes(h.axes4);
set(gca,'xtick',[],'ytick',[]);
box on;

% Update h structure
guidata(hObject, h);

% UIWAIT makes MovieGUI wait for user response (see UIRESUME)
% uiwait(h.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MovieGUI_OutputFcn(hObject, eventdata, h)
% Get default command line output from h structure
varargout{1} = h.output;


%%%%% choose folder to write binary file
function pushbutton18_Callback(hObject, eventdata, h)
folder_name = uigetdir(h.binfolder);
if folder_name ~= 0
    h.binfolder = folder_name;
    set(h.text21,'String',h.binfolder);
end
% Update h structure
guidata(hObject, h);


% ------------ choose folder -- can have multiple blocks!
function folder_Callback(hObject, eventdata, h)
folder_name = uigetdir(h.filepath);
if folder_name ~= 0
    h.rootfolder = folder_name;
    
    [filename,folders,namef] = findMovieFolders(h,folder_name);
    namef
    if isempty(filename{1})
        msgbox('ahh! no movie files found!');
    else
        
        [filename,folders,namef] = chooseFiles(filename,folders,namef);
        
        axes(h.axes1);
        cla;
        axes(h.axes4);
        cla('reset');
        axis off;
                
        h.files = filename;
        h.folders = folders;
        h.vr = [];
        h.nX = [];
        h.nY = [];
        fstr{1}=[];
        for k = 1:size(filename,1)
            for j = 1:size(filename,2)
                h.vr{k,j} = VideoReader(filename{k,j});
                if j==1
                    nX{k}    = h.vr{k}.Width;
                    nY{k}    = h.vr{k}.Height;
                end
                fstr{k,j} = sprintf('%s/%s',folders{k,j},namef{k,j});
            end
            h.whichfile = k;
            h = resetROIs(h);
        end
        h.nX = nX;
        h.nY = nY;
        
        h.wROI = 1;
        h.rcurr = 0;
        h.whichfile = 1;
        h.whichview = 1;
        
        % delete small ROIs
        h.plotROIs(:) = 0;
        h.ROIfile(:) = 0;
        
        % string for movie choosing menu
        set(h.popupmenu6, 'String', fstr);
        set(h.popupmenu6, 'Value', h.whichfile);
                
        % reset ROIs to fit in video
        
        fprintf('displaying \n%s\n',filename{h.whichfile});
        if length(folder_name) > length(h.filepath)
            if strcmp(folder_name(1:length(h.filepath)),h.filepath)
                foldname = folder_name(length(h.filepath)+1:end);
                ns       = strfind(foldname,'\');
                if isempty(ns)
                    ns   = strfind(foldname,'/');
                end
                if ~isempty(ns)
                    ns = ns(1);
                    foldname = sprintf('%s\n%s',foldname(1:ns),foldname(ns+1:end));
                    set(h.text13,'String',foldname);
                else
                    set(h.text13,'String',folder_name);
                end
            else
                set(h.text13,'String',folder_name);
            end
        else
            set(h.text13,'String',folder_name);
        end
        
        h.folder_name = folder_name;
        h.nframes = h.vr{h.whichfile}.Duration*h.vr{h.whichfile}.FrameRate - 1;
        disp(h.nframes+1)
        h.cframe = 1;
        set(h.slider1,'Value',0);
        set(h.slider4,'Value',0);
        set(h.edit3,'String','1');
        
        PlotFrame(h);
    end
end
guidata(hObject,h);

% --------------- choose file to view! -------------------%
function popupmenu6_Callback(hObject, eventdata, h)
wfile0=h.whichfile;
h.whichfile = get(hObject,'Value');
[h.whichview, ~] = ind2sub(size(h.files), h.whichfile);
h.cframe = 1;
h.wROI = 1;
set(h.checkbox16,'Value',1);
axes(h.axes1);
cla;
if wfile0 ~= h.whichfile
    axes(h.axes4);
    cla('reset');
    h.indROI=[];
    box off;
    axis off;
end
PlotFrame(h);
fprintf('displaying \n%s\n',h.files{h.whichfile});
h.nframes = h.vr{h.whichfile}.Duration*h.vr{h.whichfile}.FrameRate-1;
set(h.edit1,'String',sprintf('%1.2f',h.saturation(h.whichfile)));
set(h.slider5,'Value',h.framesaturation(h.whichfile));
guidata(hObject,h);


% SET WHETHER OR NOT TO VIEW AREAS
function checkbox16_Callback(hObject, eventdata, h)
wc = get(hObject,'Value');
h.wROI = wc;
PlotFrame(h);
guidata(hObject,h);

% ------- DRAW INCLUDED ROIs FOR MULTIVIDEO SVD ----------------------- %
function keepROI_Callback(hObject, eventdata, h)
nxS = floor(h.nX{h.whichview} / h.sc);
nyS = floor(h.nY{h.whichview} / h.sc);
if isempty(h.ROI{h.whichview}{1})
    ROI0 = [1 1 nxS nyS];
else
    ROI0 = [nxS*.25 nyS*.25 nxS*.5 nyS*.5];
end
ROI = drawROI(h,ROI0);
ROI = onScreenROI(ROI, nxS, nyS);
if isempty(h.ROI{h.whichview}{1})
    h.ROI{h.whichview}{1} = ROI;
else
    h.ROI{h.whichview}{end+1} = ROI;
end
h.rcurr = 0;

PlotFrame(h);

guidata(hObject, h);

% ------- DRAW EXCLUDED ROIs FOR MULTIVIDEO SVD ----------------------- %
function excludeROI_Callback(hObject, eventdata, h)
nxS = floor(h.nX{h.whichview} / h.sc);
nyS = floor(h.nY{h.whichview} / h.sc);
ROI0 = [nxS*.4 nyS*.4 nxS*.3 nyS*.3];
ROI = drawROI(h,ROI0);
ROI = onScreenROI(ROI, nxS, nyS);
if isempty(h.eROI{h.whichview}{1})
    h.eROI{h.whichview}{1} = ROI;
else
    h.eROI{h.whichview}{end+1} = ROI;
end
h.rcurr = 1;

PlotFrame(h);

guidata(hObject, h);

% --- delete all multivideo SVD ROIs ---- %
function deleteall_Callback(hObject, eventdata, h)
h = resetROIs(h);
PlotFrame(h);
guidata(hObject, h);

% --- delete last drawn multivideo SVD ROI ---- %
function deleteone_Callback(hObject, eventdata, h)
if h.rcurr
    if numel(h.eROI{h.whichview}) > 1
        h.eROI{h.whichview} = h.eROI{h.whichview}(1:end-1);
    else
        h.eROI{h.whichview}{1} = [];
    end
else
    if numel(h.ROI{h.whichview}) > 1
        h.ROI{h.whichview} = h.ROI{h.whichview}(1:end-1);
    else
        h.ROI{h.whichview}{1} = [];
    end
end
PlotFrame(h);
guidata(hObject, h);

% use a GPU?
function checkbox17_Callback(hObject, eventdata, h)
wc = get(hObject,'Value');
h.useGPU = wc;
guidata(hObject,h);

% --- SLIDER FOR CHOOSING DISPLAYED FRAME ------------------ %
function slider1_CreateFcn(hObject, eventdata, h)
set(hObject,'Interruptible','On');
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.75 .75 .75]);
end

function slider1_Callback(hObject, eventdata, h)
set(hObject,'Interruptible','On');
set(hObject,'BusyAction','cancel');
v = get(hObject,'Value');
smin = get(hObject,'Min');
smax = get(hObject,'Max');
cframe = min(h.nframes-1,max(1,round(v/(smax-smin) * h.nframes)));
h.cframe = cframe;
set(h.edit3,'String',num2str(cframe));
set(h.slider4,'Value',h.cframe/(h.nframes-1));
PlotFrame(h);
PlotROI(h);
guidata(hObject,h);

% --- FINESCALE SLIDER ------------------ %
function slider4_Callback(hObject, eventdata, h)
set(hObject,'Interruptible','On');
set(hObject,'BusyAction','cancel');
set(hObject,'SliderStep',[1/double(h.nframes-1) 2/double(h.nframes-1)]);
v = get(hObject,'Value');
cframe = min(h.nframes-1,max(1,round((v)*h.nframes)));%%/ h.nframes)));
h.cframe = cframe;
set(h.edit3,'String',num2str(cframe));
set(h.slider1,'Value',h.cframe/h.nframes);
PlotFrame(h);
PlotROI(h);
guidata(hObject,h);

function slider4_CreateFcn(hObject, eventdata, h)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
set(hObject,'Min',0);
set(hObject,'Max',1);

% --- PLAY button
function togglebutton1_Callback(hObject, eventdata, h)
while get(hObject, 'value') && h.cframe+1 < h.nframes
    h.cframe = h.cframe+1;
    set(h.edit3,'String',num2str(h.cframe));
    set(h.slider1,'Value',h.cframe/h.nframes);
    PlotFrame(h);
    PlotROI(h);
end
set(h.slider4,'Value',h.cframe/(h.nframes-1));
guidata(hObject,h);

% --- Executes during object creation, after setting all properties.
function slider5_CreateFcn(hObject, eventdata, h)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% ***************** SATURATION SETTINGS ****************%
% --- SLIDER FOR FULL FRAME SATURATION ---------------------%
function slider5_Callback(hObject, eventdata, h)
set(hObject,'Interruptible','On');
set(hObject,'BusyAction','cancel');
sats = get(hObject,'Value');
h.framesaturation(h.whichview) = sats;
PlotFrame(h);
guidata(hObject,h);

% --- SLIDER FOR SELECTED ROI SATURATION -------------------%
function slider2_Callback(hObject, eventdata, h)
set(hObject,'Interruptible','On');
set(hObject,'BusyAction','cancel');
sats = get(hObject,'Value');
h.saturation(h.indROI) = sats;
set(h.edit1,'String',sprintf('%1.2f',sats));
%PlotFrame(h);
axes(h.axes4);
cla;
PlotROI(h);
guidata(hObject, h);

function slider2_CreateFcn(hObject, eventdata, h)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% saturation edit
function edit1_Callback(hObject, eventdata, h)
sval = get(hObject,'String');
set(h.slider2,'Value',str2num(sval));
h.saturation(h.whichfile) = str2num(sval);
PlotFrame(h);
guidata(hObject,h);

function edit1_CreateFcn(hObject, eventdata, h)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white','String',num2str(h.pupLow));
end

% ****** Save ROI settings and keep list of saved folders ----- %
function savesettings_Callback(hObject, eventdata,h)
saveROI(h);

guidata(hObject,h);


% ****** ROIs will be processed across expts -------------------- %
function processROIs_Callback(hObject, eventdata, h)
h = processROIs(h);
guidata(hObject,h);


% **** initialize list of files box
function popupmenu6_CreateFcn(hObject, eventdata, h)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
set(hObject,'String','first choose folder');
set(hObject,'Value',1);
if isfield(h,'folders')
    set(hObject,'String',folders);
end


%%%%%% ***************** TEXT BOXES *************** %%%%%%%
%%%%% frame number edit box
function edit3_Callback(hObject, eventdata, h)
cframe = get(hObject,'String');
if iscell(cframe)
    cframe = cframe{1};
end
h.cframe = max(1,min(h.nframes-1,round(str2num(cframe))));
set(hObject,'String',sprintf('%d',h.cframe));
set(h.slider1,'Value',h.cframe/h.nframes);
set(h.slider4,'Value',h.cframe/h.nframes);
PlotFrame(h);
guidata(hObject,h);

function edit3_CreateFcn(hObject, eventdata, h)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%%%%% SPATIAL SMOOTHING BOX
function edit4_Callback(hObject, eventdata, h)
spatscale = get(hObject,'String');
if iscell(spatscale)
    spatscale = spatscale{1};
end
spatscale = max(1, min(50, round(str2num(spatscale))));
% resize ROIs
h = resizeROIs(h, spatscale);

h.sc    = spatscale;
set(hObject, 'String', sprintf('%d', h.sc));
axes(h.axes1);
cla;
PlotFrame(h);
axes(h.axes4);
cla;
PlotROI(h);
guidata(hObject, h);

function edit4_CreateFcn(hObject, eventdata, h)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%%%%%% pupil threshold edit box ---------------------- %
function pupilsigma_Callback(hObject, eventdata, h)
thres = get(hObject,'String');
if iscell(thres)
    thres = thres{1};
end
h.thres = max(1, min(10, (str2num(thres))));
set(hObject, 'String', sprintf('%1.1f', h.thres));
PlotROI(h);
guidata(hObject,h);


function pupilsigma_CreateFcn(hObject, eventdata, h)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%% temporal smoothing constant box --------------------- %
function edit5_Callback(hObject, eventdata, h)
tempscale = get(hObject,'String');
if iscell(tempscale)
    tempscale = tempscale{1};
end
h.tsc    = max(1, min(50, round(str2num(tempscale))));
set(hObject, 'String', sprintf('%d', h.tsc));
PlotFrame(h);
guidata(hObject, h);

function edit5_CreateFcn(hObject, eventdata, h)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- SMALLER ROIs AND PUPILS ----------------------------------- %
function running_Callback(hObject, eventdata, h)
h.indROI = 1;

h = drawSmallROI(h);
guidata(hObject, h);

function whisker_Callback(hObject, eventdata, h)
h.indROI = 2;

h = drawSmallROI(h);
guidata(hObject, h);

function snout_Callback(hObject, eventdata, h)
h.indROI = 3;

h = drawSmallROI(h);
guidata(hObject, h);

function otherROI_Callback(hObject, eventdata, h)
h.indROI = 4;

h = drawSmallROI(h);
guidata(hObject, h);

function pupil1_Callback(hObject, eventdata, h)
h.indROI = 5;

h = drawSmallROI(h);
guidata(hObject, h);

function pupil2_Callback(hObject, eventdata, h)
h.indROI = 6;

h = drawSmallROI(h);
guidata(hObject, h);

function blink1_Callback(hObject, eventdata, h)
h.indROI = 7;

h = drawSmallROI(h);
guidata(hObject, h);

function blink2_Callback(hObject, eventdata, h)
h.indROI = 8;

h = drawSmallROI(h);
guidata(hObject, h);

% --------- DELETE ROIs -------- %

function runningdelete_Callback(hObject, eventdata, h)
h.plotROIs(1) = 0;
h.ROIfile(1) = 0;
PlotFrame(h);
axes(h.axes4);
cla;
title('');
guidata(hObject,h);

function whiskerdelete_Callback(hObject, eventdata, h)
h.plotROIs(2) = 0;
h.ROIfile(2) = 0;
PlotFrame(h);
axes(h.axes4);
cla;
title('');
guidata(hObject,h);

function snoutdelete_Callback(hObject, eventdata, h)
h.plotROIs(3) = 0;
h.ROIfile(3) = 0;
PlotFrame(h);
axes(h.axes4);
cla;
title('');
guidata(hObject,h);

function otherdelete_Callback(hObject, eventdata, h)
h.plotROIs(4) = 0;
h.ROIfile(4) = 0;
PlotFrame(h);
axes(h.axes4);
cla;
title('');
guidata(hObject,h);

function pupil1delete_Callback(hObject, eventdata, h)
h.plotROIs(5) = 0;
h.ROIfile(5) = 0;
PlotFrame(h);
axes(h.axes4);
cla;
title('');
guidata(hObject,h);

function pupil2delete_Callback(hObject, eventdata, h)
h.plotROIs(6) = 0;
h.ROIfile(6) = 0;
PlotFrame(h);
axes(h.axes4);
cla;
title('');
guidata(hObject,h);

function blink1delete_Callback(hObject, eventdata, h)
h.plotROIs(7) = 0;
h.ROIfile(7) = 0;
PlotFrame(h);
axes(h.axes4);
cla;
title('');
guidata(hObject,h);

function blink2delete_Callback(hObject, eventdata, h)
h.plotROIs(8) = 0;
h.ROIfile(8) = 0;
PlotFrame(h);
axes(h.axes4);
cla;
title('');
guidata(hObject,h);

function pushbutton23_Callback(hObject, eventdata, h)
