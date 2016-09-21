handles.filepath = '\\zserver.ioo.ucl.ac.uk\Data\EyeCamera\';
handles.suffix   = '.mj2'; % suffix of eye camera file!
handles.nX       = 640;
handles.nY       = 480;

folder_name = uigetdir(handles.filepath);
[filename,folders] = FindBlocks(handles,folder_name);
for k = 1:length(folders)
    procfile{k} = filename{k}(1:end-length(handles.suffix));
    procfile{k} = [procfile{k} '_proc.mat'];
end

%%
k = 1;
vr = VideoReader(filename{k});

load(procfile{k});
fsvd = proc.data.face{2};

%%
close all;
figure('position',[0 0 320 320]);
%%
vr.CurrentTime = 3550;
tpt0 = vr.CurrentTime*vr.FrameRate;
tpt = tpt0;
tf = 7500;
mr = size(runSpeed,1)/size(fsvd,1);


nframes = 1000;
F(nframes) = struct('cdata',[],'colormap',[]);
    
savedir='D:\DATA\F\M160525_MD030\mov1';

tframes = 0;
while hasFrame(vr) && tframes<377*4
    tpt = tpt+1;
    vf = readFrame(vr);
    if mod(tpt,5) == 0
        tframes = tframes+1;
        clf
        axes('position',[0.05 .35 .9 .6]);
        imagesc(vf,[0 200])
        colormap('gray');
        axis off;
        
        axes('position',[0.05 .1 .9 .23]);
        nc = 10;
        zf = zscore(fsvd(:,1:nc),1,1);
        plot(zf(tpt0+[0:tf],end:-1:1),'linewidth',1);
        text(0,10,'motion SVDs');
        %hold all;
        %plot((tpt-tpt0)*ones(1,nc),zf(tpt,:),'k.','markersize',16);
        axis tight;
        axis off;
        
        axes('position',[0.05 .03 .9 .38]);
        plot(runSpeed(round(tpt0*mr)+[0:tf*mr]),'k','linewidth',1);
        hold all;
        plot((tpt-tpt0)*mr*ones(2,1),[-5 65],'k','linewidth',1);
        axis off;
        axis tight;
        ylim([-10 100]);
        text(0,-8,'running');
        
        
        drawnow;
        %print(sprintf('%s/frame%d',savedir,tframes),'-dpng')
        %pause(.25);
        
       
    end
end