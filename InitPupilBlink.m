function [roif,wroi] = InitPupilBlink(handles,wroi,avgframe)

roif(2) = struct();
for j = wroi
    roif(j).rX       = handles.rXc{j};
    roif(j).rY       = handles.rYc{j};
    roif(j).nX       = numel(roif(j).rX);
    roif(j).nY       = numel(roif(j).rY);
    nX = roif(j).nX;
    nY = roif(j).nY;
    roif(j).avgframe = avgframe(roif(j).rX,roif(j).rY);
    
    roif(j).sats     = (1-handles.saturation(j))*255;
    avgf             = roif(j).avgframe < roif(j).sats;
 
    if j==1
        roif(j).boxfact = 1.2;
    else
        roif(j).boxfact = 1.4;
    end
    roif(j).ccenter = round([nX/2 nY/2]);
    if j == 1
        roif(j).cradius = roif(j).boxfact*[nX/5 nY/5];
    else
        roif(j).cradius = roif(j).boxfact*[nX/3 nY/3];
    end
    % box around pupil
    [binds,extinds,boxX,boxY] = MakeBox(roif(j));
    roif(j).boxinds = binds;
    roif(j).boxext  = extinds;
    roif(j).boxX    = boxX;
    roif(j).boxY    = boxY;
    roif(j).fitellipse = handles.fitellipse(j);
    
    if sum(avgf(:))<20
        wroi(find(wroi==j)) = [];
        if j==1
            disp('no pupil area found :( ... decrease saturation');
        else
            disp('no blink area found :( ... decrease saturation')
        end
    end
end