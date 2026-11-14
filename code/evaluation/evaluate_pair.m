function r = evaluate_pair(prediction, reference)
% Same-scale XYZ arrays; phantom crop only, not clinical map evaluation.
assert(isequal(size(prediction),size(reference)));
p=single(prediction(20:100,20:100,1:60));
t=single(reference(20:100,20:100,1:60));
r.NMSE=cal_Weber_NMSE(t,p);
r.SSIM=ssim(t,p);
end
