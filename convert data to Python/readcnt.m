function [S,event] = readcnt(cntfile)

HDR = sopen(cntfile,'r',[],'32bit');
[S,HDR] = sread(HDR,HDR.SPR/HDR.SampleRate,0);
%reorganize the events to delete the repeating events. GZH 04/12/08
firstEventPos = HDR.EVENT.POS(1);
eventPos = HDR.EVENT.POS-firstEventPos+1;
eventSeq = zeros(1,eventPos(end));
eventSeq(eventPos) = HDR.EVENT.TYP;
eventPos = find(eventSeq~=0);
eventType = eventSeq(eventPos);
eventPos = eventPos+firstEventPos-1;
output_event.type = eventType';
output_event.pos = eventPos';

event = output_event;

end