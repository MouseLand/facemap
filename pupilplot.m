clf
tm = 1.84e4;
plot([1:tm]/(60*30),proc.pupil(1).area_raw(1:tm),'color',[0 .6 0],...
    'linewidth',1.5)
hold all;
plot([1:tm]/(60*30),proc.pupil(1).area(1:tm),'k','linewidth',.5)
axis tight;
box off;
ylim([0 400])
xlabel('time (minutes)')
ylabel('area (pixels^2)');
text(.78,.65,'raw trace','color',[0 .6 0],'units','normalized');
text(.73,.55,'processed trace','color',[0 0 0],'units','normalized');