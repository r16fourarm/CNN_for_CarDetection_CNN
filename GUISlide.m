function varargout = GUISlide(varargin)
% GUISLIDE MATLAB code for GUISlide.fig
%      GUISLIDE, by itself, creates a new GUISLIDE or raises the existing
%      singleton*.
%
%      H = GUISLIDE returns the handle to a new GUISLIDE or the handle to
%      the existing singleton*.
%
%      GUISLIDE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUISLIDE.M with the given input arguments.
%
%      GUISLIDE('Property','Value',...) creates a new GUISLIDE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUISlide_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUISlide_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUISlide

% Last Modified by GUIDE v2.5 20-Dec-2016 11:48:42

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUISlide_OpeningFcn, ...
                   'gui_OutputFcn',  @GUISlide_OutputFcn, ...
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


% --- Executes just before GUISlide is made visible.
function GUISlide_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUISlide (see VARARGIN)

% Choose default command line output for GUISlide
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUISlide wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUISlide_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in bLoad.
function bLoad_Callback(hObject, eventdata, handles)
% hObject    handle to bLoad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = uigetfile('.jpg');
axes(handles.gambar1);
imshow(file);


% --- Executes on button press in bProses.
function bProses_Callback(hObject, eventdata, handles)
% hObject    handle to bProses (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    mkdir('out');
    m = getimage(handles.gambar1);
    size(m)
    
    stride = 56;
    
    slideWidth = size(m,2) - 168
    slideHeight = size(m,1) - 168
    timesSlideRight = ((slideWidth - mod(slideWidth,stride)) / 56) - 1
    timesSlideDown = ((slideHeight - mod(slideHeight,stride)) / 56) - 1
    
    numberOfUniqueWindows = ((timesSlideRight + 1) * (timesSlideDown + 1))
    windowColle = zeros(224,224,3,numberOfUniqueWindows,'uint8');
    example = zeros(224,224,3,'uint8');
    
    windowCount = 0;
    for row = 112:+stride:((timesSlideDown * stride) + 112)
        
        for col = 112:+stride:((timesSlideRight * stride) + 112)
            
            windowCount = windowCount + 1;
            colleRow = 0;
            for windowOffsetFromCenterRow = (row - 111):+1:(row + 112)
                colleRow = colleRow + 1;
                colleCol = 0;
                for windowOffsetFromCenterCol = (col - 111):+1:(col + 112)
                    colleCol = colleCol + 1;
                    %windowColle(colleRow,colleCol,:,windowCount) = window(windowOffsetFromCenter,windowOffsetFromCenter,:);
                    example(colleRow,colleCol,:) = m(windowOffsetFromCenterRow, windowOffsetFromCenterCol,:);
                end
            end
            %imshow(example);
            imwrite(example,['out\window' num2str(windowCount) '.jpg']);
        end
    end
    
    %%%
    %cmd/python code here 
    %%%
%     dos('G:')
%     dos('cd cnn\fix') 
    status=dos(['python smalcnn.py' ' ' num2str(windowCount)])
    if status==0
        warndlg('done')
    end
    
    delete('out\*.jpg');
    rmdir('out');


% --- Executes on button press in bSquare.
function bSquare_Callback(hObject, eventdata, handles)
% hObject    handle to bSquare (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%mkdir('out');
square = true(224,224);

for row = 9:1:216
    for col = 9:1:216
        square(row,col) = 0;
    end
end

carProb = csvread('CarProbability.csv')

m = getimage(handles.gambar1);
stride = 56;
    
slideWidth = size(m,2) - 168;
slideHeight = size(m,1) - 168;
timesSlideRight = ((slideWidth - mod(slideWidth,stride)) / 56) - 1;
timesSlideDown = ((slideHeight - mod(slideHeight,stride)) / 56) - 1;

numberOfUniqueWindows = ((timesSlideRight + 1) * (timesSlideDown + 1));
windowColle = zeros(224,224,3,numberOfUniqueWindows,'uint8');
example = zeros(128,128,3,'uint8');
  
windowCount = 1;
carAmount = 0;
for row = 112:+stride:((timesSlideDown * stride) + 112)
    
    for col = 112:+stride:((timesSlideRight * stride) + 112)
        carAmount = carAmount + 1;
         if ((carProb(windowCount,1) > carProb(windowCount,2)) && (carProb(windowCount,1) >= 0.9))
         %if ((carProb(windowCount,1) > carProb(windowCount,2)))
         %if ((carProb(windowCount)) == 0)   
            colleRow = 0;
            for windowOffsetFromCenterRow = (row - 111):+1:(row + 112)
                colleRow = colleRow + 1;
                colleCol = 0;
                for windowOffsetFromCenterCol = (col - 111):+1:(col + 112)
                    colleCol = colleCol + 1;
                    if ((square(colleRow, colleCol) == 1))
                        m(windowOffsetFromCenterRow, windowOffsetFromCenterCol,1) = 255;
                        m(windowOffsetFromCenterRow, windowOffsetFromCenterCol,2) = 0;
                        m(windowOffsetFromCenterRow, windowOffsetFromCenterCol,3) = 0;
                    end
                end
            end
        end
        windowCount = windowCount + 1;
        %imshow(example);
    end
end

%carAmount
imshow(m);
delete('CarProbability.csv');
