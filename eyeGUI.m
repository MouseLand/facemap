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

% Last Modified by GUIDE v2.5 06-Sep-2016 22:48:19

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
handles.filepath = '\\zserver.ioo.ucl.ac.uk\Data\EyeCamera\';
handles.suffix   = '.mj2'; % suffix of eye camera file!
handles.nX       = 640;
handles.nY       = 480;
for j = 1:6
    handles.ROI{j} = [handles.nX/4 handles.nY/4 handles.nX/4 handles.nY/4]; 
    if j == 6
        handles.ROI{j} = [2 2 handles.nX-4 handles.nY-4];
    end
    ROI = handles.ROI{j};
    handles.rY{j}   = floor(ROI(1)-1+[1:ROI(3)]);
    handles.rX{j}   = floor(ROI(2)-1+[1:ROI(4)]);
end
handles.plotROIs  = false(6,1);
handles.lastROI   = false(6,1);
handles.whichROIs = false(2,1);
handles.svdmat    = false(4,3);
handles.colors    = [.2 .6 1; 0 1 0; 1 .4 .2; 1 .8 .2; .8 0 .8; .7 .6 1];
handles.fitellipse = [0 1]; 
handles.roif(2) = struct();
% by default fit a circle to the pupil
% and an ellipse to the eye area

set(handles.slider2,'Min',0);
set(handles.slider2,'Max',1);
set(handles.slider2,'Value',0);
set(handles.checkbox15,'Value',1); % use GPU by default
handles.useGPU = 1;
set(handles.edit1,'String',num2str(0));
handles.saturation = zeros(6,1);

axes(handles.axes1);
set(gca,'xtick',[],'ytick',[]);
box on;
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


% ---- do you have a gpu? -----------%
function checkbox15_Callback(hObject, eventdata, handles)
wc = get(hObject,'Value');
handles.useGPU  = wc;
guidata(hObject,handles);


% ------------ choose folder -- can have multiple blocks!
function folder_Callback(hObject, eventdata, handles)
folder_name = uigetdir(handles.filepath);
handles.rootfolder = folder_name;
[filename,folders] = FindBlocks(handles,folder_name);
if isempty(filename{1})
    msgbox('ahh! no movie files found!');
else
    handles.files = filename;
    handles.folders = folders;
    % which block do you want to view
%     if length(filename)>1
%         [folds,didchoose] = listdlg('PromptString','choose single file to view',...
%             'SelectionMode','single','ListSize',[160 160],'ListString',folders);
%         if didchoose
%             handles.whichfile = folds;
%         else
%             handles.whichfile = 1;
%         end
%     else
        handles.whichfile = 1;
    %end
    set(handles.popupmenu6,'String',folders);
    set(handles.popupmenu6,'Value',handles.whichfile);
    handles.vr = VideoReader(filename{handles.whichfile});
    fprintf('displaying \n%s\n',filename{handles.whichfile});
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
        end
    else
        set(handles.text13,'String',folder_name);
    end
    handles.folder_name = folder_name;
    handles.cframe = 1;
    handles.nframes = handles.vr.Duration*handles.vr.FrameRate-1;
    PlotEye(handles);
    
end
guidata(hObject,handles);

% --------------- choose file to view! -------------------%
function popupmenu6_Callback(hObject, eventdata, handles)
handles.whichfile = get(hObject,'Value');
handles.vr = VideoReader(handles.files{handles.whichfile});
fprintf('displaying \n%s\n',handles.files{handles.whichfile});
handles.cframe = 1;
set(handles.slider1,'Value',0);
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
handles.rY{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rX{j}   = floor(ROI(2)-1+[1:ROI(4)]);
handles.whichROIs(j) = 1;
handles.plotROIs(j) = 1;
handles.lastROI(j) = 1;
if ~isfield(handles,'roif')
    handles.roif(2) = struct();
end
handles = InitPlot(handles,j);
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
handles.rY{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rX{j}   = floor(ROI(2)-1+[1:ROI(4)]);
handles.whichROIs(j) = 1;
handles.plotROIs(j) = 1;
handles.lastROI(j) = 1;
handles = InitPlot(handles,j);
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
handles.rY{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rX{j}   = floor(ROI(2)-1+[1:ROI(4)]);
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
handles.rY{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rX{j}   = floor(ROI(2)-1+[1:ROI(4)]);
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
handles.rY{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rX{j}   = floor(ROI(2)-1+[1:ROI(4)]);
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
handles.rY{j}   = floor(ROI(1)-1+[1:ROI(3)]);
handles.rX{j}   = floor(ROI(2)-1+[1:ROI(4)]);
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
PlotEye(handles);

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
ns = strfind(foldname,'\');
if isempty(ns)
    ns = strfind(foldname,'/');
end
if ~isempty(ns)
    foldname = foldname(1:ns(end));
end
handles.multifilelabel{ik} = foldname;

guidata(hObject,handles);


% ----- ROIs will be processed in blocks -------------------- %
function processROIs_Callback(hObject, eventdata, handles)
handles = ProcessROIs(handles);
handles = SaveROI(handles);

guidata(hObject,handles);


% ----- batch process ROIs -------------------- %
function pushbutton17_Callback(hObject, eventdata, handles)
% make multi-file folder list to choose from
[folds,didchoose] = listdlg('PromptString','which folders (ctrl for multiple)',...
    'SelectionMode','multiple','ListSize',[240 160],'ListString',handles.multifilelabel);

for j = folds
    load(handles.multifiles{j})
    proc.axesPupil = handles.axesPupil;
    proc.axes1     = handles.axes1;
    proc.useGPU    = handles.useGPU;
    proc = ProcessROIs(proc);
    proc = SaveROI(proc);
end


guidata(hObject,handles);



                
    

% ---save function
function pushbutton2_Callback(hObject, eventdata, handles)
keyboard;
for idx = 1:2:1000
    vr = VideoReader(handles.files{1});
    rimg = read(vr,idx*2);
    clf
    axes('position',[.05 .2 .9 .7]);
    imagesc(rimg(1:2:end,1:2:end));
    colormap('gray')
    axes('position',[.05 .05 .9 .1]);
    pf = zscore(handles.face{2}(1:5,:)');
    plot(pf)
    hold all;
    plot(idx,pf(idx,:),'k*');
    axis tight;
    drawnow
    pause(.5);
end

save('f1.mat','handles');


% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.


% --- Executes on button press in pushbutton15.


% --- Executes on button press in checkbox6.


% --- Executes on selection change in popupmenu6.

% --- Executes during object creation, after setting all properties.
function popupmenu6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
set(hObject,'String','first choose folder');
set(hObject,'Value',1);
if isfield(handles,'folders')
    set(hObject,'String',folders);
end


% --- Executes on button press in pushbutton16.
% hObject    handle to pushbutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in checkbox15.
% hObject    handle to checkbox15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox15



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cframe = get(hObject,'String');
if iscell(cframe)
    cframe = cframe{1};
end
handles.cframe = max(1,min(handles.nframes,round(str2num(cframe))));
%keyboard;
set(hObject,'String',sprintf('%d',handles.cframe));
set(handles.slider1,'Value',handles.cframe/handles.nframes);
PlotEye(handles);
guidata(hObject,handles);

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
