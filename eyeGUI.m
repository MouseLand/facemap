function varargout = eyeGUI(varargin)
% EYEGUI MATLAB code for eyeGUI.fig
%      EYEGUI, by itself, creates a new EYEGUI or raises the existing
%      singleton*.
%
%      H = EYEGUI returns the handle to a new EYEGUI or the handle to
%      the existing singleton*.
%
%      EYEGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EYEGUI.M with the given input arguments.
%
%      EYEGUI('Property','Value',...) creates a new EYEGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before eyeGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to eyeGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help eyeGUI

% Last Modified by GUIDE v2.5 13-Feb-2017 20:19:14

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @eyeGUI_OpeningFcn, ...
    'gui_OutputFcn',  @eyeGUI_OutputFcn, ...
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


% --- Executes just before eyeGUI is made visible.
function eyeGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to eyeGUI (see VARARGIN)

% Choose default command line output for eyeGUI
handles.output = hObject;

% default filepath for eye camera
handles.filepath = '\\zserver.cortexlab.net\Data\EyeCamera\';
handles.suffix   = {'.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf'}; % suffix of eye camera file!

% default filepath to write binary file (ideally an SSD)
handles.binfolder = 'F:\DATA';

% default file size
handles.nX       = 640;
handles.nY       = 480;

handles = ResetROIs(handles);

handles.whichROIs = false(2,1);
handles.svdmat    = false(4,3);
handles.colors    = [.2 .6 1; 0 1 0; 1 .4 .2; 1 .8 .2; .8 0 .8; .7 .6 1];

% default smoothing constants
handles.sc        = 4;
handles.tsc       = 3;

% default stds for eye ROIs
handles.thres     = [4 6];

set(handles.slider2,'Min',0);
set(handles.slider2,'Max',1);
set(handles.slider2,'Value',0);
set(handles.edit1,'String',num2str(0));
handles.saturation = zeros(6,1);

axes(handles.axes1);
set(gca,'xtick',[],'ytick',[]);
box on;
handles.roiaxes = handles.axesPupil.Position;
axes(handles.axesPupil);
set(gca,'xtick',[],'ytick',[]);
box on;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes eyeGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = eyeGUI_OutputFcn(hObject, eventdata, handles)
% Get default command line output from handles structure
varargout{1} = handles.output;


%%%%% choose folder to write binary file
function pushbutton18_Callback(hObject, eventdata, handles)
folder_name = uigetdir(handles.binfolder);
if folder_name ~= 0
    handles.binfolder = folder_name;
    set(handles.text21,'String',handles.binfolder);
end
% Update handles structure
guidata(hObject, handles);


% ------------ choose folder -- can have multiple blocks!
function folder_Callback(hObject, eventdata, handles)
folder_name = uigetdir(handles.filepath);
if folder_name ~= 0
    handles.rootfolder = folder_name;
    [filename,folders,namef] = FindBlocks(handles,folder_name);
    
    if isempty(filename{1})
        msgbox('ahh! no movie files found!');
    else
        [filename,folders,namef] = ChooseFiles(filename,folders,namef);
    
        handles.files = filename;
        handles.folders = folders;
        handles.whichfile = 1;
        
        set(handles.popupmenu6,'String',folders);
        set(handles.popupmenu6,'Value',handles.whichfile);
        handles.vr = VideoReader(filename{handles.whichfile});
        nX    = handles.vr.Width;
        nY    = handles.vr.Height;
        handles.nX = nX;
        handles.nY = nY;
        % reset ROIs to fit in video
        handles = ResetROIs(handles);

        fprintf('displaying \n%s\n',filename{handles.whichfile});
        if length(folder_name) > length(handles.filepath)
            if strcmp(folder_name(1:length(handles.filepath)),handles.filepath)
                foldname = folder_name(length(handles.filepath)+1:end);
                ns       = strfind(foldname,'\');
                if isempty(ns)
                    ns   = strfind(foldname,'/');
                end
                if ~isempty(ns)
                    ns = ns(1);
                    foldname = sprintf('%s\n%s',foldname(1:ns),foldname(ns+1:end));
                    set(handles.text13,'String',foldname);
                else
                    set(handles.text13,'String',folder_name);
                end
            else
                set(handles.text13,'String',folder_name);
            end
        else
            set(handles.text13,'String',folder_name);
        end
        
        handles.folder_name = folder_name;
        handles.cframe = 1;
        set(handles.slider1,'Value',0);
        set(handles.slider4,'Value',0);
        set(handles.edit3,'String','1');
        handles.nframes = handles.vr.Duration*handles.vr.FrameRate-1;
        PlotEye(handles);
    end
end
guidata(hObject,handles);

% --------------- choose file to view! -------------------%
function popupmenu6_Callback(hObject, eventdata, handles)
handles.whichfile = get(hObject,'Value');
handles.vr = VideoReader(handles.files{handles.whichfile});
fprintf('displaying \n%s\n',handles.files{handles.whichfile});
handles.cframe = 1;
set(handles.slider1,'Value',0);
set(handles.slider4,'Value',0);
set(handles.edit3,'String','1');
PlotEye(handles);
handles.nframes = handles.vr.Duration*handles.vr.FrameRate-1;
guidata(hObject,handles);


% --- DRAW ROI BUTTONS -------------------------------------------%
function roi0 = OnScreenROI(ROI,nX,nY)
roi0 = ROI;
roi0(1) = min(nX,max(1,ROI(1)));
roi0(2) = min(nY,max(1,ROI(2)));
roi0(3) = min(nX-roi0(1),ROI(3));
roi0(4) = min(nY-roi0(2),ROI(4));

function drawpupilROI_Callback(hObject, eventdata, handles)
j = 1;
handles.lastROI=false(6,1);
ROI = handles.ROI{j};
ROI = DrawROI(handles,ROI);
ROI = OnScreenROI(ROI,handles.nX,handles.nY);
handles.ROI{j} = ROI;
handles.rX{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rY{j}   = floor(ROI(2)-1+[1:ROI(4)]);
handles.whichROIs(j) = 1;
handles.plotROIs(j) = 1;
handles.lastROI(j) = 1;
PlotEye(handles);
set(handles.checkbox1,'Value',1);
set(handles.slider2,'Value',handles.saturation(j));
set(handles.edit1,'String',num2str(handles.saturation(j)));
guidata(hObject, handles);

function drawblinkROI_Callback(hObject, eventdata, handles)
j = 2;
handles.lastROI=false(6,1);
ROI = handles.ROI{j};
ROI = DrawROI(handles,ROI);
ROI = OnScreenROI(ROI,handles.nX,handles.nY);
handles.ROI{j} = ROI;
handles.rX{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rY{j}   = floor(ROI(2)-1+[1:ROI(4)]);
handles.whichROIs(j) = 1;
handles.plotROIs(j) = 1;
handles.lastROI(j) = 1;
PlotEye(handles);
set(handles.checkbox2,'Value',1);
set(handles.slider2,'Value',handles.saturation(j));
set(handles.edit1,'String',num2str(handles.saturation(j)));
guidata(hObject, handles);

function drawwhiskerROI_Callback(hObject, eventdata, handles)
j = 3;
handles.lastROI=false(6,1);
ROI = handles.ROI{j};
ROI = DrawROI(handles,ROI);
ROI = OnScreenROI(ROI,handles.nX,handles.nY);
handles.ROI{j} = ROI;
handles.rX{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rY{j}   = floor(ROI(2)-1+[1:ROI(4)]);
handles.plotROIs(j) = 1;
handles.lastROI(j) = 1;
PlotEye(handles);
if sum(handles.svdmat(j-2,:))==0
    handles.svdmat(j-2,1) = 1;
    set(handles.checkbox3,'Value',1);
end
set(handles.slider2,'Value',handles.saturation(j));
set(handles.edit1,'String',num2str(handles.saturation(j)));
guidata(hObject, handles);

function drawgroomROI_Callback(hObject, eventdata, handles)
j = 4;
handles.lastROI=false(6,1);
ROI = handles.ROI{j};
ROI = DrawROI(handles,ROI);
ROI = OnScreenROI(ROI,handles.nX,handles.nY);
handles.ROI{j} = ROI;
handles.rX{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rY{j}   = floor(ROI(2)-1+[1:ROI(4)]);
handles.plotROIs(j) = 1;
handles.lastROI(j) = 1;
PlotEye(handles);
if sum(handles.svdmat(j-2,:))==0
    handles.svdmat(j-2,1) = 1;
    set(handles.checkbox4,'Value',1);
end
set(handles.slider2,'Value',handles.saturation(j));
set(handles.edit1,'String',num2str(handles.saturation(j)));
guidata(hObject, handles);

function drawsnoutROI_Callback(hObject, eventdata, handles)
j = 5;
handles.lastROI=false(6,1);
ROI = handles.ROI{j};
ROI = DrawROI(handles,ROI);
ROI = OnScreenROI(ROI,handles.nX,handles.nY);
handles.ROI{j} = ROI;
handles.rX{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rY{j}   = floor(ROI(2)-1+[1:ROI(4)]);
handles.plotROIs(j) = 1;
handles.lastROI(j) = 1;
PlotEye(handles);
if sum(handles.svdmat(j-2,:))==0
    handles.svdmat(j-2,1) = 1;
    set(handles.checkbox5,'Value',1);
end
set(handles.slider2,'Value',handles.saturation(j));
set(handles.edit1,'String',num2str(handles.saturation(j)));
guidata(hObject, handles);

function drawfaceROI_Callback(hObject, eventdata, handles)
j = 6;
handles.lastROI=false(6,1);
ROI = handles.ROI{j};
ROI = DrawROI(handles,ROI);
ROI = OnScreenROI(ROI,handles.nX,handles.nY);
handles.ROI{j} = ROI;
handles.rX{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rY{j}   = floor(ROI(2)-1+[1:ROI(4)]);
handles.plotROIs(j) = 1;
handles.lastROI(j) = 1;
PlotEye(handles);
if sum(handles.svdmat(j-2,:))==0
    handles.svdmat(j-2,1) = 1;
    set(handles.checkbox6,'Value',1);
end
set(handles.slider2,'Value',handles.saturation(j));
set(handles.edit1,'String',num2str(handles.saturation(j)));
guidata(hObject, handles);

% --- SLIDER FOR CHOOSING DISPLAYED FRAME ------------------ %
function slider1_CreateFcn(hObject, eventdata, handles)
% Hint: slider controls usually have a light gray background.
set(hObject,'Interruptible','On');
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.75 .75 .75]);
end

function slider1_Callback(hObject, eventdata, handles)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
set(hObject,'Interruptible','On');
set(hObject,'BusyAction','cancel');
v = get(hObject,'Value');
smin = get(hObject,'Min');
smax = get(hObject,'Max');
cframe = min(handles.nframes,max(1,round(v/(smax-smin) * handles.nframes)));
handles.cframe = cframe;
set(handles.edit3,'String',num2str(cframe));
set(handles.slider4,'Value',handles.cframe/handles.nframes);
PlotEye(handles);

guidata(hObject,handles);


% --- FINESCALE SLIDER ------------------ %
function slider4_Callback(hObject, eventdata, handles)
set(hObject,'Interruptible','On');
set(hObject,'BusyAction','cancel');
set(hObject,'SliderStep',[1/double(handles.nframes) 2/double(handles.nframes)]);
v = get(hObject,'Value');
cframe = min(handles.nframes,max(1,round((v)*handles.nframes)));%%/ handles.nframes)));
handles.cframe = cframe;
set(handles.edit3,'String',num2str(cframe));
set(handles.slider1,'Value',handles.cframe/handles.nframes);
PlotEye(handles);
guidata(hObject,handles);

function slider4_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
set(hObject,'Min',0);
set(hObject,'Max',1);


% --- PLAY button
function togglebutton1_Callback(hObject, eventdata, handles)
while get(hObject, 'value') && handles.cframe < handles.nframes
    handles.cframe = handles.cframe+4;
    set(handles.edit3,'String',num2str(handles.cframe));
    set(handles.slider1,'Value',handles.cframe/handles.nframes);
    PlotEye(handles);
end
set(handles.slider4,'Value',handles.cframe/handles.nframes);
guidata(hObject,handles);


% --- SLIDERS FOR CONTRAST IN PUPIL WINDOW -------------------%
function slider2_Callback(hObject, eventdata, handles)
set(hObject,'Interruptible','On');
set(hObject,'BusyAction','cancel');
sats = get(hObject,'Value');
if ~isempty(handles.lastROI)
    handles.saturation(handles.lastROI) =sats;
end
set(handles.edit1,'String',sprintf('%1.2f',sats));
PlotEye(handles);
guidata(hObject, handles);

function slider2_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function edit1_Callback(hObject, eventdata, handles)
sval = get(hObject,'String');
set(handles.slider2,'Value',str2num(sval));
if ~isempty(handles.lastROI)
    handles.saturation(handles.lastROI) = str2num(sval);
end
PlotEye(handles);
guidata(hObject,handles);

function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white','String',num2str(handles.pupLow));
end


% ------ Save ROI settings and keep list of saved folders ----- %
function savesettings_Callback(hObject, eventdata, handles)
handles = SaveROI(handles);
if ~isfield(handles,'multifiles')
    ik = 1;
else
    ik = length(handles.multifiles)+1;
end
handles.multifiles{ik} = handles.settings;
if strcmp(handles.settings(1:length(handles.filepath)),handles.filepath)
    foldname = handles.settings(length(handles.filepath)+1:end);
else
    foldname = handles.settings;
end
handles.multifilelabel{ik} = foldname;

guidata(hObject,handles);


% ----- ROIs will be processed across expts -------------------- %
function processROIs_Callback(hObject, eventdata, handles)
handles = ProcessROIs_bin(handles);
handles = SaveROI(handles);
guidata(hObject,handles);


% ----- BATCH PROCESS ROIs -------------------- %
function pushbutton17_Callback(hObject, eventdata, handles)
% make multi-file folder list to choose from

[folds,didchoose] = listdlg('PromptString','which folders (ctrl for multiple)',...
    'SelectionMode','multiple','ListSize',[260 160],'ListString',handles.multifilelabel);
for j = folds
    load(handles.multifiles{j})
    fprintf('file %s\n', handles.multifiles{j});
    proc.axesPupil = handles.axesPupil;
    proc.axes1     = handles.axes1;
    proc = ProcessROIs_bin(proc);
    proc = SaveROI(proc);
end
guidata(hObject,handles);

%%%% list of files box
function popupmenu6_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
set(hObject,'String','first choose folder');
set(hObject,'Value',1);
if isfield(handles,'folders')
    set(hObject,'String',folders);
end

%%%%% frame number edit box
function edit3_Callback(hObject, eventdata, handles)
cframe = get(hObject,'String');
if iscell(cframe)
    cframe = cframe{1};
end
handles.cframe = max(1,min(handles.nframes,round(str2num(cframe))));
set(hObject,'String',sprintf('%d',handles.cframe));
set(handles.slider1,'Value',handles.cframe/handles.nframes);
set(handles.slider4,'Value',handles.cframe/handles.nframes);
PlotEye(handles);
guidata(hObject,handles);

function edit3_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%% spatial smoothing constant box
function edit4_Callback(hObject, eventdata, handles)
spatscale = get(hObject,'String');
if iscell(spatscale)
    spatscale = spatscale{1};
end
handles.sc    = max(1, min(50, round(str2num(spatscale))));
set(hObject, 'String', sprintf('%d', handles.sc));
PlotEye(handles);
guidata(hObject, handles);

function edit4_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%%%%% temporal smoothing constant box
function edit5_Callback(hObject, eventdata, handles)
tempscale = get(hObject,'String');
if iscell(tempscale)
    tempscale = tempscale{1};
end
handles.tsc    = max(1, min(50, round(str2num(tempscale))));
set(hObject, 'String', sprintf('%d', handles.tsc));
PlotEye(handles);
guidata(hObject, handles);

function edit5_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%%%% pupil sigma -- how many stds off to draw pupil ROI!
function edit6_Callback(hObject, eventdata, handles)
thres = get(hObject,'String');
if iscell(thres)
    thres = thres{1};
end
handles.thres(1)  = max(0.5, min(10, str2num(thres)));
set(hObject, 'String', sprintf('%1.1f', handles.thres(1)));
PlotEye(handles);
guidata(hObject, handles);

function edit6_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- WHICH ROIs SHOULD BE PROCESSED --------------------------- %
function checkbox1_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.whichROIs(1) = wc;
guidata(hObject,handles);

function checkbox2_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.whichROIs(2) = wc;
guidata(hObject,handles);

function checkbox3_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(1,1)  = wc;
guidata(hObject,handles);

function checkbox4_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(2,1)  = wc;
guidata(hObject,handles);

% --- Executes on button press in checkbox5.
function checkbox5_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(3,1)  = wc;
guidata(hObject,handles);

function checkbox6_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(4,1)  = wc;
guidata(hObject,handles);

function checkbox7_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(1,2)  = wc;
guidata(hObject,handles);


function checkbox8_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(2,2)  = wc;
guidata(hObject,handles);

function checkbox9_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(3,2)  = wc;
guidata(hObject,handles);

function checkbox10_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(4,2)  = wc;
guidata(hObject,handles);

function checkbox11_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(1,3)  = wc;
guidata(hObject,handles);

function checkbox12_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(2,3)  = wc;
guidata(hObject,handles);

function checkbox13_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(3,3)  = wc;
guidata(hObject,handles);

function checkbox14_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.svdmat(4,3)  = wc;
guidata(hObject,handles);

