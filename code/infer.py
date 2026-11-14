"""Portable CDR inference without normalization, filtering or resampling."""
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from architecture32 import U_Net
from preprocessing import prepare

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','input','output','weights-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--strategy',choices=['CDR'],default='CDR')
    p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    a=p.parse_args();c=json.loads(a.config.read_text())
    manifest=(a.config.parent/c['manifest']).resolve()
    with manifest.open(encoding='utf-8-sig',newline='') as f:r=[x for x in csv.DictReader(f) if x['method']=='DLAIO-CDR' and int(x['fold'])==a.fold]
    if len(r)!=2 or {x['stage'] for x in r}!={'1','2'} or len({x['pairing'] for x in r})!=1:raise ValueError('Invalid stage pairing')
    r.sort(key=lambda x:int(x['stage']));weights=[]
    for x in r:
        rel=Path(x['package_relative_path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid weight path')
        w=a.weights_root/rel
        if digest(w)!=x['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
        weights.append(w)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.set_num_threads(c['torch_threads']);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(c['sitk_threads'])
    torch.backends.cudnn.allow_tf32=c['cudnn_allow_tf32'];torch.backends.cuda.matmul.allow_tf32=c['matmul_allow_tf32']
    im=sitk.ReadImage(str(a.input));v=sitk.GetArrayFromImage(im)
    if v.shape!=(128,128,128) or not np.isfinite(v).all():raise ValueError('Require finite128-cubed input; no implicit padding/cropping')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(command=sys.argv,python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),device=a.device,device_name=torch.cuda.get_device_name(0) if a.device=='cuda' else platform.processor(),fold=a.fold,input_sha256=digest(a.input),weights_sha256=[x['sha256'] for x in r],manifest_sha256=digest(manifest),config=c,input_grid=dict(size_xyz=im.GetSize(),spacing_xyz=im.GetSpacing(),origin_xyz=im.GetOrigin(),direction=im.GetDirection()),metadata_policy='Copy input header; no physical calibration/resampling',outputs=[])
    with torch.inference_mode():
        for stage,w in enumerate(weights,1):
            net=U_Net(1,1).to(a.device,memory_format=torch.channels_last_3d).eval()
            net.load_state_dict(torch.load(w,map_location='cpu',weights_only=True),strict=True)
            x=prepare(v).to(a.device)
            if a.device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();v=net(x)[0,0].cpu().numpy();elapsed=time.perf_counter()-start
            if not np.isfinite(v).all():raise ValueError('Nonfinite output')
            out=sitk.GetImageFromArray(v);out.CopyInformation(im);path=a.output/f'stage{stage}.nii.gz';sitk.WriteImage(out,str(path))
            report['outputs'].append(dict(stage=stage,shape_zyx=list(v.shape),finite=True,minimum=float(v.min()),maximum=float(v.max()),sha256=digest(path),forward_readback_seconds=elapsed))
            del net
    (a.output/'run_record.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
