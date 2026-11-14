function nmse=cal_Weber_NMSE(orgSig,recSig,varargin)
if isempty(varargin)
    boun = 0;
else boun = varargin{1};
end

if size(orgSig,2)==1       % if signal is 1-D
    orgSig = orgSig(boun+1:end-boun,:);
    recSig = recSig(boun+1:end-boun,:);
else                       % if signal is 2-D or 3-D
    orgSig = orgSig(boun+1:end-boun,boun+1:end-boun,:);
    recSig = recSig(boun+1:end-boun,boun+1:end-boun,:);
end

[size1,a] = size(recSig(recSig>0));
avg_org = sum(orgSig(orgSig>0))/size1;
avg_sig = sum(recSig(recSig>0))/size1;
nmse=norm(orgSig(:)/avg_org-recSig(:)/avg_sig)^2/size1;


% %mse=norm(orgSig(:)-recSig(:),2)^2/length(orgSig(:));
% mse=norm(orgSig(:)-recSig(:))^2;
% %sigEner=norm(orgSig(:));
% sigEner=norm(orgSig(:))^2;

%nmse=(mse/sigEner);